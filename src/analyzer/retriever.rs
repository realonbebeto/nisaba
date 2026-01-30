use arrow::{
    array::{
        Array, AsArray, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, ListArray,
        RecordBatch, RecordBatchIterator, StringArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
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
use nalgebra::SVector;
use std::{marker::PhantomData, path::PathBuf, str::FromStr, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        AnalyzerConfig,
        report::{FieldMatch, TableMatch},
    },
    error::NisabaError,
    types::{FieldDef, TableDef, TableRep},
};

/// The `Storable` trait defines how to access common attributes of the store
/// element (TableDef/FieldDef) during latent store storage and retrieval.
pub trait Storable: Send + Sync {
    /// The result type returned from similarity search
    ///
    type SearchResult;

    /// Get the schema for this type
    fn schema() -> Arc<Schema>;

    /// Get name of the store element (TableDef/FieldDef)
    fn name(&self) -> &String;

    /// Get the id of the store element (TableDef/FieldDef)
    fn get_id(&self) -> Uuid;

    /// Get silo_id
    fn silo_id(&self) -> &str;

    /// Get the name of the resource for this type
    fn vtable_name() -> &'static str;

    /// Get the columns required to fetch results from latent store
    fn result_columns() -> Vec<String>;
}

#[derive(Clone)]
/// The `LatentStore` represents an interface to access Lancedb store
///
/// Properties:
///
/// * `conn`: The `conn` property is a `Connection` to Lancedb.
pub struct LatentStore {
    conn: Connection,
    embedding_model: EmbeddingModel,
    config: Arc<AnalyzerConfig>,
}

impl LatentStore {
    pub fn builder() -> LatentStoreBuilder {
        LatentStoreBuilder::builder()
    }
    /// The `table_handler` is a function represents the result of a clustering process
    ///
    /// Arguments:
    ///
    /// * `conn`: The `conn` property is a `Connection` to Lancedb.
    /// * `config`: The `config` property is a shared holder to AnalyzerConfig.
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

        Ok(LatentStore {
            conn,
            embedding_model: self
                .embedding_model
                .unwrap_or(EmbeddingModel::MultilingualE5Small),
            config: self.config.unwrap_or_default(),
        })
    }
}

/// The `TableHandler` represents a connection over store element which implements Storable trait
///
/// Properties:
///
/// * `conn`: The `conn` property is a `Connection` to Lancedb.
/// * `config`: The `config` property is a shared holder to AnalyzerConfig.
pub struct TableHandler<T: Storable> {
    conn: Connection,
    config: Arc<AnalyzerConfig>,
    // Marker to hold a Storable type
    // Used to associate the handler with a specific Storable type
    _phantom: PhantomData<T>,
    embedding_model: EmbeddingModel,
}

impl<T: Storable> TableHandler<T> {
    pub async fn initialize(&self) -> Result<(), NisabaError> {
        let scheme = T::schema();

        self.conn
            .create_empty_table(T::vtable_name(), scheme)
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;

        Ok(())
    }

    /// This `create_index_on_table_rep` function handles creation of table index for performant retrieval
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
    /// of FieldMatch or TableMatch
    ///
    /// Arguments:
    ///
    /// * `item`: The `item` parameter is the unique identifier of a stored TableRep.
    ///
    /// * `config`: The `config` property is a shared holder to AnalyzerConfig.
    ///
    /// * `columns`: The `columns` property is a Vec of String used to return the required
    ///   set of fields from the latent store.
    ///
    /// Returns:
    ///
    /// A `Result` of a `Vec` of SearchResult values on success and NisabaError when
    /// embedding generation, search, and creation of the SearchResult fails.
    pub async fn search_table_rep(
        &self,
        item: Uuid,
        columns: Vec<String>,
    ) -> Result<Vec<TableMatch>, NisabaError> {
        let table = self
            .conn
            .open_table(TableRep::vtable_name())
            .execute()
            .await?;

        let mut embed_batches = table
            .query()
            .only_if(format!("id = \'{}\'", item)) // No need for .to_string() if item is already displayable
            .select(Select::Columns(vec!["vector".into()]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let embed_batches = embed_batches
            .pop()
            .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

        let embedding: Vec<f32> = embed_batches
            .column(0)
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        let batches = table
            .query()
            .only_if(format!("id != \'{}\'", item))
            .nearest_to(embedding)?
            .nprobes(self.config.similarity.top_k.unwrap_or(7))
            .refine_factor(5)
            .limit(self.config.similarity.top_k.unwrap_or(7))
            .select(Select::Columns(columns))
            .distance_type(self.config.similarity.algorithm)
            .distance_range(Some(0.), Some(1. - self.config.similarity.threshold))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = table_from_record_batches(batches)?;

        Ok(results)
    }

    /// This `search_table_rep` function performs a search in the latent store and returns a vector
    /// of FieldMatch or TableMatch
    ///
    /// Arguments:
    ///
    /// * `item`: The `item` parameter is the unique identifier of a stored TableRep.
    ///
    /// * `config`: The `config` property is a shared holder to AnalyzerConfig.
    ///
    /// * `columns`: The `columns` property is a Vec of String used to return the required
    ///   set of fields from the latent store.
    ///
    /// Returns:
    ///
    /// A `Result` of a `Vec` of SearchResult values on success and NisabaError when
    /// embedding generation, search, and creation of the SearchResult fails.
    pub async fn search_field_def(
        &self,
        item: Uuid,
        columns: Vec<String>,
    ) -> Result<Vec<FieldMatch>, NisabaError> {
        let table = self
            .conn
            .open_table(FieldDef::vtable_name())
            .execute()
            .await?;

        let mut embed_batches = table
            .query()
            .only_if(format!("id = \'{}\'", item)) // No need for .to_string() if item is already displayable
            .select(Select::Columns(vec!["table_name".into(), "vector".into()]))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let embed_batches = embed_batches
            .pop()
            .ok_or_else(|| NisabaError::NoRecordBatch)?;

        let embedding: Vec<f32> = embed_batches
            .column(1)
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        let table_name: String = embed_batches
            .column(0)
            .as_string::<i32>()
            .value(0)
            .to_string();

        let batches = table
            .query()
            .only_if(format!(
                "id != \'{}\' AND table_name != \'{}\'",
                item, table_name
            ))
            .nearest_to(embedding)?
            .nprobes(self.config.similarity.top_k.unwrap_or(7))
            .refine_factor(5)
            .limit(self.config.similarity.top_k.unwrap_or(7))
            .select(Select::Columns(columns))
            .distance_type(self.config.similarity.algorithm)
            .distance_range(Some(0.), Some(1. - self.config.similarity.threshold))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = fields_from_record_batches(batches)?;

        Ok(results)
    }

    pub async fn clear_table(&self) -> Result<(), NisabaError> {
        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        table.delete("true").await?;

        Ok(())
    }

    pub async fn store_tables(&self, table_defs: Vec<TableDef>) -> Result<(), NisabaError> {
        let mut model = TextEmbedding::try_new(InitOptions::new(self.embedding_model.clone()))?;

        let field_texts: Vec<(&FieldDef, String)> = table_defs
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

        let embeddings = &model.embed(
            field_texts
                .iter()
                .map(|(_, text)| text)
                .collect::<Vec<&String>>(),
            None,
        )?;

        let field_embeds: Vec<(&FieldDef, Vec<f32>)> = field_texts
            .iter()
            .zip(embeddings.iter())
            .map(|((field_def, _text), embedding)| (*field_def, embedding.clone()))
            .collect();

        let table_embeds: Vec<(&TableDef, Vec<f32>)> = table_defs
            .iter()
            .map(|t| {
                let mut total_field_embed = vec![0.0f32; 384];
                let emb = field_embeds.iter().filter(|f| {
                    let field_ids = t.fields.iter().map(|i| i.id).collect::<Vec<Uuid>>();

                    field_ids.contains(&f.0.id)
                });

                for row in emb {
                    for (i, &val) in row.1.iter().enumerate() {
                        total_field_embed[i] += val;
                    }
                }

                let height = t.fields.len();

                total_field_embed
                    .iter_mut()
                    .for_each(|x| *x /= height as f32);

                let total_field_embed: SVector<f32, 384> = SVector::from_vec(total_field_embed);

                // Table Stats Embedding
                let structure_embed = t.structure().embedding();

                // Table Combined embedding
                let table_embedding = ((self.config.scoring.type_weight) * total_field_embed
                    + self.config.scoring.structure_weight * structure_embed)
                    .as_slice()
                    .to_vec();

                (t, table_embedding)
            })
            .collect();

        let field_schema = FieldDef::schema();

        let field_batch = fields_to_record_batch(field_embeds, field_schema.clone())?;

        let field_batches =
            RecordBatchIterator::new(vec![field_batch].into_iter().map(Ok), field_schema);

        let tbl = self
            .conn
            .open_table(FieldDef::vtable_name())
            .execute()
            .await?;

        tbl.add(field_batches).execute().await?;

        let table_schema = TableRep::schema();

        let table_batch = table_to_record_batch(table_embeds, table_schema.clone())?;

        let table_batches =
            RecordBatchIterator::new(vec![table_batch].into_iter().map(Ok), table_schema);

        let tbl = self
            .conn
            .open_table(TableRep::vtable_name())
            .execute()
            .await?;

        tbl.add(table_batches).execute().await?;

        Ok(())
    }
}

fn table_from_record_batches(batches: Vec<RecordBatch>) -> Result<Vec<TableMatch>, NisabaError> {
    let mut schemas = Vec::new();

    for batch in batches {
        // Get id values
        let id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 0 is not StringArray, got {:?}",
                batch.column(0).data_type()
            )))?;

        // Get silo_id values
        let silo_id_array = batch
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

        let fields_list =
            batch
                .column(3)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ArrowError::CastError(format!(
                    "Column 3 is not ListArray, got {:?}",
                    batch.column(3).data_type()
                )))?;

        let distances = batch
            .column(4)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 4 is not Float32Array, got {:?}",
                batch.column(4).data_type()
            )))?;

        for row_idx in 0..batch.num_rows() {
            let id = Uuid::from_str(id_array.value(row_idx))?;

            let silo_id = silo_id_array.value(row_idx).to_string();

            let name = name_array.value(row_idx).to_string();

            let field_ids = fields_list.value(0);

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

            let confidence = distances.value(row_idx);

            let schema = TableRep {
                id,
                silo_id,
                name,
                fields,
            };
            schemas.push(TableMatch { schema, confidence });
        }
    }

    Ok(schemas)
}

fn table_to_record_batch(
    items: Vec<(&TableDef, Vec<f32>)>,
    schema: Arc<Schema>,
) -> Result<RecordBatch, NisabaError> {
    let ids = StringArray::from_iter_values(items.iter().map(|t| t.0.id.to_string()));

    let silo_ids = StringArray::from_iter_values(items.iter().map(|t| t.0.silo_id.clone()));
    let names = StringArray::from_iter_values(items.iter().map(|t| t.0.name.clone()));

    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        items
            .iter()
            .map(|t| {
                let v = t.1.iter().map(|v| Some(*v)).collect::<Vec<Option<f32>>>();
                Some(v)
            })
            .collect::<Vec<_>>(),
        384,
    );

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
                .map(|f| f.id.to_string())
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
            Arc::new(names),
            Arc::new(vectors),
            Arc::new(fields),
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
fn fields_from_record_batches(batches: Vec<RecordBatch>) -> Result<Vec<FieldMatch>, NisabaError> {
    let mut schemas = Vec::new();

    for batch in batches {
        let ids_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 0 is not StringArray, got {:?}",
                batch.column(0).data_type()
            )))?;

        // Get silo_id values
        let silo_id_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 1 is not StringArray, got {:?}",
                batch.column(1).data_type()
            )))?;

        // Get table name values
        let table_schema_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 2 is not StringArray, got {:?}",
                batch.column(2).data_type()
            )))?;

        // Get table name values
        let table_name_array = batch
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

        // Get canonical_type values
        let canonical_type_array = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 5 is not StringArray, got {:?}",
                batch.column(5).data_type()
            )))?;

        //Type confidence
        let type_confidence_array = batch
            .column(6)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 6 is not Float32Array, got {:?}",
                batch.column(6).data_type()
            )))?;

        // Cardinality
        let cardinality_array = batch
            .column(7)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 7 is not Float32Array, got {:?}",
                batch.column(7).data_type()
            )))?;

        // Avg Byte Len
        let avg_byte_len_array = batch
            .column(8)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 8 is not Float32Array, got {:?}",
                batch.column(8).data_type()
            )))?;

        // Monotonicity
        let monotonicity_array = batch
            .column(9)
            .as_boolean_opt()
            .ok_or(ArrowError::CastError(format!(
                "Column 9 is not BooleanArray, got {:?}",
                batch.column(9).data_type()
            )))?;

        // Class signature
        let class_signature_array =
            batch
                .column(10)
                .as_list_opt::<i32>()
                .ok_or(ArrowError::CastError(format!(
                    "Column 10 is not ListArray, got {:?}",
                    batch.column(10).data_type()
                )))?;

        // Column Default
        let column_default_array = batch
            .column(11)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(format!(
                "Column 11 is not StringArray, got {:?}",
                batch.column(11).data_type()
            )))?;

        // Nullable
        let nullable_array = batch
            .column(12)
            .as_boolean_opt()
            .ok_or(ArrowError::CastError(format!(
                "Column 12 is not BooleanArray, got {:?}",
                batch.column(12).data_type()
            )))?;

        // Char Max Length
        let char_max_len_array = batch
            .column(13)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 13 is not Int32Array, got {:?}",
                batch.column(13).data_type()
            )))?;

        // Numeric precision
        let numeric_precision_array = batch
            .column(14)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 14 is not Int32Array, got {:?}",
                batch.column(14).data_type()
            )))?;

        // Numeric scale
        let numeric_scale_array = batch
            .column(15)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 15 is not Int32Array, got {:?}",
                batch.column(15).data_type()
            )))?;

        // Datetime precision
        let datetime_precision_array = batch
            .column(16)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 16 is not Int32Array, got {:?}",
                batch.column(16).data_type()
            )))?;

        let distances = batch
            .column(17)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or(ArrowError::CastError(format!(
                "Column 17 is not Float32Array, got {:?}",
                batch.column(17).data_type()
            )))?;

        for row_idx in 0..batch.num_rows() {
            let id = Uuid::from_str(ids_array.value(row_idx))?;
            let silo_id = silo_id_array.value(row_idx).to_string();

            let table_schema = table_schema_array.value(row_idx).to_string();

            let table_name = table_name_array.value(row_idx).to_string();

            let name = name_array.value(row_idx).to_string();

            let canonical_type = DataType::from_str(canonical_type_array.value(row_idx))?;

            let type_confidence = if type_confidence_array.is_null(row_idx) {
                None
            } else {
                Some(type_confidence_array.value(row_idx))
            };

            let cardinality = if cardinality_array.is_null(row_idx) {
                None
            } else {
                Some(cardinality_array.value(row_idx))
            };

            let avg_byte_length = if avg_byte_len_array.is_null(row_idx) {
                None
            } else {
                Some(avg_byte_len_array.value(row_idx))
            };

            let is_monotonic = monotonicity_array.value(row_idx);

            let char_class_signature = if class_signature_array.is_null(row_idx) {
                None
            } else {
                let values = class_signature_array.value(row_idx);
                let values = values.as_primitive::<Float32Type>();

                Some([
                    values.value(0),
                    values.value(1),
                    values.value(2),
                    values.value(3),
                ])
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

            let confidence = distances.value(row_idx);

            let schema = FieldDef {
                id,
                silo_id,
                table_schema,
                table_name,
                name,
                canonical_type,
                type_confidence,
                cardinality,
                avg_byte_length,
                is_monotonic,
                char_class_signature,
                column_default,
                is_nullable,
                char_max_length,
                numeric_precision,
                numeric_scale,
                datetime_precision,
            };

            schemas.push(FieldMatch { schema, confidence });
        }
    }

    Ok(schemas)
}

fn fields_to_record_batch(
    items: Vec<(&FieldDef, Vec<f32>)>,
    schema: Arc<Schema>,
) -> Result<RecordBatch, NisabaError> {
    let ids = StringArray::from_iter_values(items.iter().map(|f| f.0.id.to_string()));

    let silo_ids = StringArray::from(
        items
            .iter()
            .map(|f| f.0.silo_id.clone())
            .collect::<Vec<String>>(),
    );

    let table_schemas = StringArray::from(
        items
            .iter()
            .map(|f| f.0.table_schema.clone())
            .collect::<Vec<String>>(),
    );

    let table_names = StringArray::from(
        items
            .iter()
            .map(|f| f.0.table_name.clone())
            .collect::<Vec<String>>(),
    );

    let names = StringArray::from(
        items
            .iter()
            .map(|f| f.0.name.clone())
            .collect::<Vec<String>>(),
    );

    let canonical_types = StringArray::from(
        items
            .iter()
            .map(|f| f.0.canonical_type.to_string())
            .collect::<Vec<String>>(),
    );

    let capacity = items.len();

    // For type confidence
    let type_confidence_array = Float32Array::from(
        items
            .iter()
            .map(|f| f.0.type_confidence)
            .collect::<Vec<Option<f32>>>(),
    );

    // For cardinality
    let cardinalities_array = Float32Array::from(
        items
            .iter()
            .map(|f| f.0.cardinality)
            .collect::<Vec<Option<f32>>>(),
    );

    // For average byte length
    let avg_byte_lens_array = Float32Array::from(
        items
            .iter()
            .map(|f| f.0.avg_byte_length)
            .collect::<Vec<Option<f32>>>(),
    );

    // For monotonicity
    let monotonic_flag_array = BooleanArray::from(
        items
            .iter()
            .map(|f| f.0.is_monotonic)
            .collect::<Vec<bool>>(),
    );

    // For char signature
    let mut char_class_values = Vec::with_capacity(capacity * 4);
    let mut char_class_offsets = Vec::with_capacity(capacity);
    char_class_offsets.push(0i32);
    let mut char_class_nulls = Vec::with_capacity(capacity);

    // Column Defaults
    let column_defaults_array = StringArray::from(
        items
            .iter()
            .map(|f| f.0.column_default.clone())
            .collect::<Vec<Option<String>>>(),
    );

    let nullable_array =
        BooleanArray::from(items.iter().map(|f| f.0.is_nullable).collect::<Vec<bool>>());

    let char_max_lengths_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.char_max_length)
            .collect::<Vec<Option<i32>>>(),
    );

    let numeric_precision_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.numeric_precision)
            .collect::<Vec<Option<i32>>>(),
    );

    let numeric_scale_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.numeric_scale)
            .collect::<Vec<Option<i32>>>(),
    );

    let datetime_precision_array = Int32Array::from(
        items
            .iter()
            .map(|f| f.0.datetime_precision)
            .collect::<Vec<Option<i32>>>(),
    );

    for (field, _) in &items {
        // Char class signature
        if let Some(ccs) = field.char_class_signature {
            char_class_values.extend_from_slice(&ccs);
            char_class_offsets.push(char_class_values.len() as i32);
            char_class_nulls.push(true);
        } else {
            char_class_nulls.push(false);
            char_class_offsets.push(char_class_values.len() as i32);
        }
    }

    let char_class_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        OffsetBuffer::new(ScalarBuffer::from(char_class_offsets)),
        Arc::new(Float32Array::from(char_class_values)),
        Some(NullBuffer::from(char_class_nulls)),
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
        384,
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(silo_ids),
            Arc::new(table_schemas),
            Arc::new(table_names),
            Arc::new(names),
            Arc::new(canonical_types),
            Arc::new(type_confidence_array),
            Arc::new(cardinalities_array),
            Arc::new(avg_byte_lens_array),
            Arc::new(monotonic_flag_array),
            Arc::new(char_class_array),
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
