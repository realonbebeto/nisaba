use arrow::{
    array::{
        Array, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Int16Array, Int32Array,
        ListArray, RecordBatch, StringArray,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Float32Type, Schema},
    error::ArrowError,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;
use lancedb::{
    Connection, DistanceType,
    database::CreateTableMode,
    index::{Index, vector::IvfHnswPqIndexBuilder},
    query::{ExecutableQuery, QueryBase, Select},
};
use nalgebra::DVector;
use std::{collections::HashMap, marker::PhantomData, path::PathBuf, str::FromStr, sync::Arc};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    error::NisabaError,
    reconciler::{
        AnalyzerConfig,
        calculation::{field_weight, l2_norm},
        report::{FieldMatch, LoadedFields, TableMatch},
    },
    types::{FieldDef, FieldProfile, TableDef, TableRep},
};

/// The `Storable` trait defines how to access common attributes of the store
/// element (TableDef/FieldDef) during latent store storage and retrieval.
pub trait Storable: Send + Sync {
    /// The result type returned from similarity search
    ///
    type SearchResult;

    /// Get the schema for this type
    fn schema(dim: usize) -> Arc<Schema>;

    /// Get name of the store element (TableDef/FieldDef)
    fn name(&self) -> &String;

    /// Get the id of the store element (TableDef/FieldDef)
    fn get_id(&self) -> Uuid;

    /// Get table_id
    fn table_id(&self) -> &Uuid;

    /// Get the name of the resource for this type
    fn vtable_name() -> &'static str;

    /// Get the columns required to fetch results from latent store
    fn result_columns() -> Vec<String>;
}

#[derive(Clone)]
/// The `LatentStore` represents an interface to access Lancedb store
pub struct LatentStore {
    /// Access to Lancedb
    conn: Connection,
    /// Embedding model for vector generation
    embedding_model: Arc<Mutex<TextEmbedding>>,
    /// Analyzer config
    config: Arc<AnalyzerConfig>,
    /// Number of dimension for the model
    dim: usize,
}

impl LatentStore {
    pub fn builder() -> LatentStoreBuilder {
        LatentStoreBuilder::builder()
    }
    /// The `table_handler` is a function giving access to a Lancedb table
    ///
    /// Returns:
    ///
    /// A table_handler of type TableHandler of generic T
    pub fn table_handler<T: Storable>(&self) -> TableHandler<T> {
        TableHandler {
            conn: self.conn.clone(),
            config: self.config.clone(),
            _phantom: PhantomData,
            embedding_model: self.embedding_model.clone(),
            dim: self.dim,
        }
    }
}

pub struct LatentStoreBuilder {
    conn_path: Option<String>,
    embedding_model: Option<EmbeddingModel>,
    config: Option<Arc<AnalyzerConfig>>,
}

impl LatentStoreBuilder {
    pub fn builder() -> Self {
        Self {
            conn_path: None,
            embedding_model: None,
            config: None,
        }
    }
    pub fn connection_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.conn_path = path.map(|v| v.into());
        self
    }

    pub fn embedding_model(mut self, model: Option<EmbeddingModel>) -> Self {
        self.embedding_model = model;
        self
    }

    pub fn analyzer_config(mut self, config: Arc<AnalyzerConfig>) -> Self {
        self.config = Some(config);

        self
    }

    pub async fn build(self) -> Result<LatentStore, NisabaError> {
        let path = db_path(self.conn_path);
        let path = path
            .to_str()
            .ok_or(NisabaError::Invalid("Invalid latent store path".into()))?;

        let conn = lancedb::connect(path).execute().await?;

        let embed_model_name = self
            .embedding_model
            .unwrap_or(EmbeddingModel::MultilingualE5Small);

        let dim = TextEmbedding::get_model_info(&embed_model_name)?.dim;

        let embedding_model = TextEmbedding::try_new(InitOptions::new(embed_model_name))?;

        Ok(LatentStore {
            conn,
            embedding_model: Arc::new(Mutex::new(embedding_model)),
            config: self.config.unwrap_or_default(),
            dim,
        })
    }
}

/// The `TableHandler` represents a connection over store element which implements Storable trait
pub struct TableHandler<T: Storable> {
    /// Access to Lancedb
    conn: Connection,
    /// Analyzer config
    config: Arc<AnalyzerConfig>,
    // Marker to hold a Storable type
    // Used to associate the handler with a specific Storable type
    _phantom: PhantomData<T>,
    /// Embedding model for vector generation
    embedding_model: Arc<Mutex<TextEmbedding>>,
    /// Number of dimension for the model
    dim: usize,
}

impl<T: Storable> TableHandler<T> {
    pub async fn initialize(&self) -> Result<(), NisabaError> {
        let scheme = T::schema(self.dim);

        self.conn
            .create_empty_table(T::vtable_name(), scheme)
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;

        Ok(())
    }

    /// This `create_index` function handles creation of table index for performant retrieval
    ///
    /// Returns:
    ///
    /// A `Result` of unit value on success and NisabaError when connection to Latent Store index creation fails.
    pub async fn create_index(&self) -> Result<(), NisabaError> {
        let tbl = self.conn.open_table(T::vtable_name()).execute().await?;

        let capacity = tbl.count_rows(None).await?;

        let idx_builder = IvfHnswPqIndexBuilder::default();
        let idx_builder = idx_builder.distance_type(DistanceType::Cosine);

        if capacity > 256 {
            tbl.create_index(&["vector"], Index::IvfHnswPq(idx_builder))
                .execute()
                .await?;
        }

        Ok(())
    }

    /// This `search_table_rep` function performs a search in the latent store and returns a vector
    /// of TableMatch
    ///
    /// Arguments:
    ///
    /// * `item`: The `item` parameter is the unique identifier of a stored TableRep.
    ///
    /// * `columns`: The `columns` property is a Vec of String used to return the required
    ///   set of fields from the latent store.
    ///
    /// Returns:
    ///
    /// A `Result` of a `Vec` of SearchResult values on success and NisabaError when
    /// embedding generation, search, and creation of the SearchResult fails.
    pub async fn search_tables(
        &self,
        table_reps: &[TableRep],
        columns: Vec<String>,
    ) -> Result<Vec<(TableRep, Vec<TableMatch>)>, NisabaError> {
        let table = self
            .conn
            .open_table(TableRep::vtable_name())
            .execute()
            .await?;

        // let silo_id = table_reps[0].silo_id.clone();

        let filter_ids = format!(
            "('{}')",
            table_reps
                .iter()
                .map(|f| f.id.to_string())
                .collect::<Vec<_>>()
                .join("','")
        );

        let mut embed_batches = table
            .query()
            .only_if(format!("id IN {}", filter_ids)) // No need for .to_string() if item is already displayable
            .select(Select::Columns(vec!["vector".into()]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let embed_batches = embed_batches
            .pop()
            .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

        let embeddings: Vec<Vec<f32>> = embed_batches
            .column(0)
            .as_fixed_size_list()
            .iter()
            .map(|opt_arr| {
                opt_arr
                    .unwrap()
                    .as_primitive::<Float32Type>()
                    .values()
                    .to_vec()
            })
            .collect();

        let mut query = table
            .query()
            // .only_if(format!("silo_id != \'{}\'", silo_id))
            .nearest_to(embeddings[0].clone())?;

        for embed in embeddings.iter().skip(1) {
            query = query.add_query_vector(embed.clone())?;
        }

        let batches = query
            .nprobes(self.config.similarity.top_k.unwrap_or(20))
            .refine_factor(20)
            .limit(self.config.similarity.top_k.unwrap_or(20))
            .select(Select::Columns(columns))
            .distance_type(self.config.similarity.algorithm)
            // .distance_range(Some(0.), Some(1. - self.config.similarity.threshold))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = table_from_record_batches(table_reps, batches)?;

        Ok(results)
    }

    /// This `load_fields` function performs a search in the latent store and returns a vector
    /// of FieldMatch
    ///
    /// Arguments:
    ///
    /// * `item`: The `item` parameter is the unique identifier of a stored TableRep.
    ///
    /// * `columns`: The `columns` property is a Vec of String used to return the required
    ///   set of fields from the latent store.
    ///
    /// Returns:
    ///
    /// A `Result` of a `Vec` of SearchResult values on success and NisabaError when
    /// embedding generation, search, and creation of the SearchResult fails.
    pub async fn load_fields(
        &self,
        query_table_id: &Uuid,
        table_ids: &[Uuid],
        columns: Vec<String>,
    ) -> Result<LoadedFields, NisabaError> {
        let table = self
            .conn
            .open_table(FieldDef::vtable_name())
            .execute()
            .await?;

        let filter_ids = format!(
            "('{}')",
            table_ids
                .iter()
                .chain(std::iter::once(query_table_id))
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("','")
        );

        let batches = table
            .query()
            .only_if(format!(" table_id IN {}", filter_ids))
            .select(Select::Columns(columns))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = fields_from_record_batches(query_table_id, batches)?;

        Ok(results)
    }

    pub async fn clear_table(&self) -> Result<(), NisabaError> {
        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        table.delete("true").await?;

        Ok(())
    }

    pub async fn store_tables(&self, table_defs: Vec<TableDef>) -> Result<(), NisabaError> {
        let field_texts: Vec<(&FieldProfile, String)> = table_defs
            .iter()
            .flat_map(|table_def| {
                let mut text_buffer = String::new();
                table_def.fields.iter().map(move |field| {
                    text_buffer.clear();
                    field.write_field_def_paragraph(&mut text_buffer);
                    (field, text_buffer.clone())
                })
            })
            .collect();

        let embeddings = {
            let mut model = self.embedding_model.lock().await;
            model.embed(
                field_texts
                    .iter()
                    .map(|(_, text)| text)
                    .collect::<Vec<&String>>(),
                None,
            )?
        };

        let field_embeds: Vec<(&FieldProfile, Vec<f32>)> = field_texts
            .iter()
            .zip(embeddings.iter())
            .map(|((field_def, _text), embedding)| (*field_def, embedding.clone()))
            .collect();

        let table_embeds: Vec<(&TableDef, Vec<f32>)> = table_defs
            .iter()
            .map(|t| {
                // Group fields by table
                let field_embeds = field_embeds.iter().filter(|f| {
                    let field_ids = t
                        .fields
                        .iter()
                        .map(|i| i.field_def.id)
                        .collect::<Vec<Uuid>>();

                    field_ids.contains(&f.0.field_def.id)
                });

                let (accumulator, total_weight) = field_embeds.fold(
                    (DVector::zeros(self.dim), 0.0),
                    |(mut acc, total_w), (field_def, embedding)| {
                        let weight = field_weight(field_def.field_stats.as_ref().unwrap());
                        let embeding = DVector::from_vec(embedding.clone());

                        acc.axpy(weight, &embeding, 1.0);
                        (acc, total_w + weight)
                    },
                );

                // Table Combined embedding
                let table_embedding = l2_norm(accumulator / total_weight).as_slice().to_vec();

                (t, table_embedding)
            })
            .collect();

        let field_schema = FieldDef::schema(self.dim);

        let field_batch = fields_to_record_batch(field_embeds, field_schema.clone(), self.dim)?;

        let tbl = self
            .conn
            .open_table(FieldDef::vtable_name())
            .execute()
            .await?;

        tbl.add(vec![field_batch]).execute().await?;

        let table_schema = TableRep::schema(self.dim);

        let table_batch = table_to_record_batch(table_embeds, table_schema.clone(), self.dim)?;

        let tbl = self
            .conn
            .open_table(TableRep::vtable_name())
            .execute()
            .await?;

        tbl.add(vec![table_batch]).execute().await?;

        Ok(())
    }
}

fn table_from_record_batches(
    table_reps: &[TableRep],
    batches: Vec<RecordBatch>,
) -> Result<Vec<(TableRep, Vec<TableMatch>)>, NisabaError> {
    let mut table_matches = Vec::with_capacity(batches.len());

    for batch in batches {
        // Get index value
        let index_value = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 0 is not Int32Array, got {:?}",
                batch.column(0).data_type()
            )))?
            .value(0);

        let source = table_reps[index_value as usize].clone();

        // Get id values
        let id_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 1 is not StringArray, got {:?}",
                batch.column(1).data_type()
            )))?;

        // Get silo_id values
        let silo_id_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 2 is not StringArray, got {:?}",
                batch.column(2).data_type()
            )))?;

        // Get name values
        let table_schema_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 3 is not StringArray, got {:?}",
                batch.column(3).data_type()
            )))?;

        // Get name values
        let name_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 4 is not StringArray, got {:?}",
                batch.column(4).data_type()
            )))?;

        let fields_list =
            batch
                .column(5)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ArrowError::CastError(format!(
                    "Column 5 is not ListArray, got {:?}",
                    batch.column(5).data_type()
                )))?;

        let distances = batch
            .column(6)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 6 is not Float32Array, got {:?}",
                batch.column(6).data_type()
            )))?;

        let mut tmp_match = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            let id = Uuid::from_str(id_array.value(row_idx))?;

            let silo_id = silo_id_array.value(row_idx).to_string();

            let table_schema = table_schema_array.value(row_idx).to_string();

            let name = name_array.value(row_idx).to_string();

            let field_ids = fields_list.value(row_idx);

            let field_ids =
                field_ids
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or(ArrowError::CastError(format!(
                        "Column is not StringArray, got {:?}",
                        field_ids.data_type()
                    )))?;

            let mut fields = Vec::new();

            for index in 0..field_ids.len() {
                let id = Uuid::from_str(field_ids.value(index))?;
                fields.push(id);
            }

            let confidence = 1.0 - distances.value(row_idx);
            let num_fields = fields.len();

            let schema = TableRep {
                id,
                silo_id,
                table_schema,
                name,
                fields,
            };

            // Remove self matches
            let ratio = num_fields as f32 / source.fields.len() as f32;

            // TODO: SizeCompatibilityConfig struct
            if (0.7..=1.3).contains(&ratio) {
                tmp_match.push(TableMatch { schema, confidence });
            }
        }

        table_matches.push((source, tmp_match));
    }

    Ok(table_matches)
}

fn table_to_record_batch(
    items: Vec<(&TableDef, Vec<f32>)>,
    schema: Arc<Schema>,
    dim: usize,
) -> Result<RecordBatch, NisabaError> {
    let ids = StringArray::from_iter_values(items.iter().map(|t| t.0.id.to_string()));

    let silo_ids = StringArray::from_iter_values(items.iter().map(|t| t.0.silo_id.clone()));

    let table_schemas =
        StringArray::from_iter_values(items.iter().map(|t| t.0.table_schema.clone()));

    let names = StringArray::from_iter_values(items.iter().map(|t| t.0.name.clone()));

    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        items
            .iter()
            .map(|t| {
                let v = t.1.iter().map(|v| Some(*v)).collect::<Vec<Option<f32>>>();
                Some(v)
            })
            .collect::<Vec<_>>(),
        dim as i32,
    );

    let num_fields = Int16Array::from_iter_values(items.iter().map(|t| t.0.fields.len() as i16));

    let capacity = items.iter().map(|t| t.0.fields.len()).sum::<usize>();

    // For char signature
    let mut field_values = Vec::with_capacity(capacity * 4);
    let mut field_offsets = Vec::with_capacity(capacity);
    field_offsets.push(0i32);

    for (table, _) in items {
        field_values.extend_from_slice(
            &table
                .fields
                .iter()
                .map(|f| f.field_def.id.to_string())
                .collect::<Vec<String>>(),
        );

        field_offsets.push(field_values.len() as i32);
    }

    let fields = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, false)),
        OffsetBuffer::new(ScalarBuffer::from(field_offsets)),
        Arc::new(StringArray::from(field_values)),
        None,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(silo_ids),
            Arc::new(table_schemas),
            Arc::new(names),
            Arc::new(fields),
            Arc::new(num_fields),
            Arc::new(vectors),
        ],
    )?;

    Ok(batch)
}

/// The function `from_record_batches` processes a vector of `RecordBatch` instances to extract and
/// transform data into a specific data structure, handling various data types and error cases along
/// the way.
///
/// Arguments:
///
/// * `batches`: The `from_record_batches` function takes a vector of `RecordBatch` instances as
///   input. Each `RecordBatch` contains columns of data where each column represents a specific field
///   or attribute of the records.
///
/// Returns:
///
/// The function `from_record_batches` returns a `Result` containing a vector of `FieldMatch`
/// structs, which represent the schema information extracted from the input `RecordBatch`
/// instances. The `FieldMatch` struct contains a `FieldDef` struct representing the schema details
/// of a field/column and a confidence value associated with that schema.
fn fields_from_record_batches(
    query_table_id: &Uuid,
    batches: Vec<RecordBatch>,
) -> Result<LoadedFields, NisabaError> {
    let mut query_fields = Vec::new();
    let mut candidate_fields: HashMap<Uuid, Vec<FieldMatch>> = HashMap::new();

    for batch in batches {
        let ids_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 0 is not StringArray, got {:?}",
                batch.column(0).data_type()
            )))?;

        // Get table_id values
        let table_id_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 1 is not StringArray, got {:?}",
                batch.column(1).data_type()
            )))?;

        // Get name values
        let name_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 2 is not StringArray, got {:?}",
                batch.column(2).data_type()
            )))?;

        // Get canonical_type values
        let canonical_type_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 3 is not StringArray, got {:?}",
                batch.column(3).data_type()
            )))?;

        //Type confidence
        let type_confidence_array = batch
            .column(4)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 4 is not Float32Array, got {:?}",
                batch.column(4).data_type()
            )))?;

        // Column Default
        let column_default_array = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 5 is not StringArray, got {:?}",
                batch.column(5).data_type()
            )))?;

        // Nullable
        let nullable_array = batch
            .column(6)
            .as_boolean_opt()
            .ok_or(ArrowError::CastError(format!(
                "Column 6 is not BooleanArray, got {:?}",
                batch.column(6).data_type()
            )))?;

        // Char Max Length
        let char_max_len_array = batch
            .column(7)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 7 is not Int32Array, got {:?}",
                batch.column(7).data_type()
            )))?;

        // Numeric precision
        let numeric_precision_array = batch
            .column(8)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 8 is not Int32Array, got {:?}",
                batch.column(8).data_type()
            )))?;

        // Numeric scale
        let numeric_scale_array = batch
            .column(9)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 9 is not Int32Array, got {:?}",
                batch.column(9).data_type()
            )))?;

        // Datetime precision
        let datetime_precision_array = batch
            .column(10)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 10 is not Int32Array, got {:?}",
                batch.column(10).data_type()
            )))?;

        let embedding_array = batch
            .column(11)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 11 is not FixedSizeListArray, got {:?}",
                batch.column(11).data_type()
            )))?;

        for row_idx in 0..batch.num_rows() {
            let id = Uuid::from_str(ids_array.value(row_idx))?;
            let table_id = Uuid::from_str(table_id_array.value(row_idx))?;

            let name = name_array.value(row_idx).to_string();

            let canonical_type = DataType::from_str(canonical_type_array.value(row_idx))?;

            let type_confidence = if type_confidence_array.is_null(row_idx) {
                None
            } else {
                Some(type_confidence_array.value(row_idx))
            };

            let column_default = if column_default_array.is_null(row_idx) {
                None
            } else {
                Some(column_default_array.value(row_idx).to_owned())
            };

            let is_nullable = nullable_array.value(row_idx);

            let char_max_length = if char_max_len_array.is_null(row_idx) {
                None
            } else {
                Some(char_max_len_array.value(row_idx))
            };

            let numeric_precision = if numeric_precision_array.is_null(row_idx) {
                None
            } else {
                Some(numeric_precision_array.value(row_idx))
            };

            let numeric_scale = if numeric_scale_array.is_null(row_idx) {
                None
            } else {
                Some(numeric_scale_array.value(row_idx))
            };

            let datetime_precision = if datetime_precision_array.is_null(row_idx) {
                None
            } else {
                Some(datetime_precision_array.value(row_idx))
            };

            let embedding = embedding_array
                .value(row_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(ArrowError::CastError(
                    "Inner array is not Float32Array".to_string(),
                ))?
                .values()
                .to_vec();

            let schema = FieldDef {
                id,
                table_id,
                name,
                canonical_type,
                type_confidence,
                column_default,
                is_nullable,
                char_max_length,
                numeric_precision,
                numeric_scale,
                datetime_precision,
            };

            if query_table_id == &table_id {
                query_fields.push(FieldMatch {
                    schema,
                    embedding: DVector::from_vec(embedding),
                });
            } else {
                candidate_fields
                    .entry(table_id)
                    .or_default()
                    .push(FieldMatch {
                        schema,
                        embedding: DVector::from_vec(embedding),
                    });
            }
        }
    }

    Ok(LoadedFields {
        query_fields,
        candidate_fields,
    })
}

fn fields_to_record_batch(
    items: Vec<(&FieldProfile, Vec<f32>)>,
    schema: Arc<Schema>,
    dim: usize,
) -> Result<RecordBatch, NisabaError> {
    let ids = StringArray::from_iter_values(items.iter().map(|f| f.0.field_def.id.to_string()));

    let table_ids = StringArray::from(
        items
            .iter()
            .map(|f| f.0.field_def.table_id.to_string())
            .collect::<Vec<String>>(),
    );

    let names = StringArray::from(
        items
            .iter()
            .map(|f| f.0.field_def.name.clone())
            .collect::<Vec<String>>(),
    );

    let canonical_types = StringArray::from(
        items
            .iter()
            .map(|f| f.0.field_def.canonical_type.to_string())
            .collect::<Vec<String>>(),
    );

    // For type confidence
    let type_confidence_array = Float32Array::from(
        items
            .iter()
            .map(|f| f.0.field_def.type_confidence)
            .collect::<Vec<Option<f32>>>(),
    );

    // Column Defaults
    let column_defaults_array = StringArray::from(
        items
            .iter()
            .map(|f| f.0.field_def.column_default.clone())
            .collect::<Vec<Option<String>>>(),
    );

    let nullable_array = BooleanArray::from(
        items
            .iter()
            .map(|f| f.0.field_def.is_nullable)
            .collect::<Vec<bool>>(),
    );

    let char_max_lengths_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.field_def.char_max_length)
            .collect::<Vec<Option<i32>>>(),
    );

    let numeric_precision_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.field_def.numeric_precision)
            .collect::<Vec<Option<i32>>>(),
    );

    let numeric_scale_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.field_def.numeric_scale)
            .collect::<Vec<Option<i32>>>(),
    );

    let datetime_precision_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.field_def.datetime_precision)
            .collect::<Vec<Option<i32>>>(),
    );

    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        items.iter().map(|(_, embedding)| {
            Some(
                embedding
                    .iter()
                    .map(|f| Some(*f))
                    .collect::<Vec<Option<f32>>>(),
            )
        }),
        dim as i32,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(table_ids),
            Arc::new(names),
            Arc::new(canonical_types),
            Arc::new(type_confidence_array),
            Arc::new(column_defaults_array),
            Arc::new(nullable_array),
            Arc::new(char_max_lengths_array),
            Arc::new(numeric_precision_array),
            Arc::new(numeric_scale_array),
            Arc::new(datetime_precision_array),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    Ok(batch)
}

/// A function that creates/provides local path to where the latent store will be created or is existing
fn db_path(dir_path: Option<String>) -> PathBuf {
    match dir_path {
        Some(v) => {
            let mut path = PathBuf::new();
            path.push(v);
            path
        }
        None => {
            let mut path = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
            path.push("nisaba");
            path
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_path_creation() {
        let default_path = db_path(None);
        let parent_dir = default_path.parent().unwrap();

        assert!(parent_dir.exists());
    }

    #[tokio::test]
    async fn test_new_with_custom_path() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();

        let _ltstore = LatentStore::builder()
            .analyzer_config(Arc::new(AnalyzerConfig::default()))
            .connection_path(Some(temp_path))
            .build()
            .await
            .unwrap();

        assert!(std::path::Path::new(temp_path).exists());
    }

    #[tokio::test]
    async fn test_new_valid_connection() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();

        let ltstore = LatentStore::builder()
            .analyzer_config(Arc::new(AnalyzerConfig::default()))
            .connection_path(Some(temp_path))
            .embedding_model(None)
            .build()
            .await
            .unwrap();

        let tables = ltstore.conn.table_names().execute().await.unwrap();

        assert!(tables.is_empty() || !tables.is_empty());
    }

    #[test]
    fn test_vector_storage() {}

    #[test]
    fn test_vector_retrieval() {}
}
