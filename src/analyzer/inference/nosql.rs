use arrow::{
    array::{
        ArrayRef, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array,
        NullArray, RecordBatch, StringArray, TimestampMillisecondArray,
    },
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
};
use futures::TryStreamExt;
use mongodb::bson::{Bson, Document, doc};

use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

use crate::{
    analyzer::{
        datastore::{Extra, Source},
        inference::{SchemaInferenceEngine, SourceField, convert_into_table_defs},
        probe::InferenceStats,
    },
    error::NisabaError,
    types::{TableDef, TableRep},
};

// =================================================
// NoSQL-Like Inference Engine
// =================================================

/// NoSQL inference engine for MongoDB
#[derive(Debug)]
/// The NoSQLInferenceEngine struct has a field sample_size of type u32.
///
/// Properties:
///
/// * `sample_size`: The `sample_size` property in `NoSQLInferenceEngine` represents the size
///   of the sample data that will be used by the inference engine for analysis.
pub struct NoSQLInferenceEngine {
    sample_size: usize,
}

impl Default for NoSQLInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl NoSQLInferenceEngine {
    pub fn new(sample_size: Option<usize>) -> Self {
        NoSQLInferenceEngine {
            sample_size: sample_size.unwrap_or(1000),
        }
    }

    /// The function `infer_from_mongodb` asynchronously infers table definitions from MongoDB
    /// collections based on the provided configuration.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter in the `infer_from_mongodb` function is of type
    ///   `&StorageConfig`, which contains information about the storage configuration such as the backend
    ///   type, connection string, and database name.
    ///
    /// Returns:
    ///
    /// The function `mongodb_store_infer` returns a `Result` containing a vector of `TableDef` structs
    /// or a `NisabaError`.
    pub async fn mongodb_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        on_table: Arc<F>,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let client = source
            .client
            .as_mongo()
            .ok_or(NisabaError::Missing("No mongodb client provided".into()))?;

        let db = client.default_database().ok_or(NisabaError::Missing(
            "No database for Mongo Client was provided".into(),
        ))?;

        let db_name =
            source
                .metadata
                .extra
                .get(&Extra::MongoDatabase)
                .ok_or(NisabaError::Missing(
                    "No database for Mongo provided".into(),
                ))?;

        let collections = db.list_collection_names().await?;

        let mut table_reps_results: Vec<Result<TableRep, NisabaError>> = Vec::new();

        for collection_name in collections {
            let result = async {
                let collection = db.collection::<Document>(&collection_name);
                let cursor = collection
                    .find(doc! {})
                    .limit(self.sample_size as i64)
                    .await?;

                let docs: Vec<Document> = cursor.try_collect().await?;

                let (result, schema) = self.mongo_collection_infer(
                    db_name,
                    &collection_name,
                    &docs,
                    &source.metadata.silo_id,
                )?;

                let mut batch = self.docs_to_record_batch(&docs, schema.clone())?;

                let table_def = convert_into_table_defs(result)?;

                {
                    let mut stats = infer_stats.lock().await;
                    stats.tables_found += table_def.len();
                }

                let mut table_def = table_def
                    .into_iter()
                    .next()
                    .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

                // Promotion
                self.enrich_table_def(&mut table_def, &mut batch)?;

                on_table(vec![table_def.clone()]).await?;

                Ok::<TableRep, NisabaError>((&table_def).into())
            }
            .await;

            table_reps_results.push(result);
        }

        let mut errors = Vec::new();
        let mut table_reps = Vec::new();

        for result in table_reps_results {
            match result {
                Ok(rep) => {
                    table_reps.push(rep);
                }
                Err(e) => errors.push(e.to_string()),
            }
        }

        {
            let mut stats = infer_stats.lock().await;
            stats.errors.extend(errors);

            stats.tables_inferred += table_reps.len();

            stats.fields_inferred += table_reps.iter().map(|t| t.fields.len()).sum::<usize>();
        }

        Ok(table_reps)
    }

    /// The `mongo_collection_infer` function infers field types from MongoDB documents and
    /// returns source fields and a schema.
    ///
    /// Arguments:
    ///
    /// * `db_name`: The `db_name` parameter refers to the name of the database where the collection
    ///   is located in MongoDB.
    ///
    /// * `collection_name`: The `collection_name` parameter refers to the name of the MongoDB collection
    ///   for which schema inference needs to be performed on.
    ///
    /// * `docs`: The `docs` parameter in the `mongo_cpllection_infer` function refers to the slice of MongoDB Documents.
    ///
    /// * `silo_id`: The `silo_id` parameter represents the identifier for the data silo where the collection is located.
    ///
    /// Returns:
    ///
    /// A tuple is being returned containing a vector of `SourceField` structs and an `Arc<Schema>`
    /// instance.
    fn mongo_collection_infer(
        &self,
        db_name: &str,
        collection_name: &str,
        docs: &[Document],
        silo_id: &str,
    ) -> Result<(Vec<SourceField>, Arc<Schema>), NisabaError> {
        let mut field_types: HashMap<String, DataType> = HashMap::new();

        // Safe assumption to use the first document to infer types
        self.mongo_document_infer(&docs[0], &mut field_types);

        let source_fields: Vec<SourceField> = field_types
            .clone()
            .into_iter()
            .map(|(name, dtype)| SourceField {
                silo_id: silo_id.into(),
                table_schema: db_name.into(),
                table_name: collection_name.into(),
                column_name: name,
                column_default: None,
                is_nullable: "yes".into(),
                data_type: format!("{}", dtype),
                character_maximum_length: None,
                udt_name: format!("{}", dtype),
                numeric_precision: None,
                numeric_scale: None,
                datetime_precision: None,
            })
            .collect();

        let schema_fields: Vec<Field> = field_types
            .into_iter()
            .map(|(n, t)| Field::new(n, t, true))
            .collect();

        let schema = Arc::new(Schema::new(schema_fields));

        Ok((source_fields, schema))
    }

    /// The function `mongo_document_infer` iterates through a MongoDB document to infer data types for
    /// each field and update a HashMap with the inferred types.
    ///
    /// Arguments:
    ///
    /// * `doc`: The `doc` parameter is a reference to `Document` object, which represents a document in a MongoDB collection.
    ///
    /// * `field_types`: The `field_types` parameter is a mutable reference to a `HashMap` that stores
    ///   the data types inferred for each field in a MongoDB document.
    fn mongo_document_infer(&self, doc: &Document, field_types: &mut HashMap<String, DataType>) {
        for (key, value) in doc.iter() {
            let field_name = key.trim().to_lowercase();

            let inferred_type = bson_to_arrow_type(value);

            field_types
                .entry(field_name.clone())
                .and_modify(|existing| *existing = merge_types(existing, &inferred_type))
                .or_insert(inferred_type);

            // TODO: Possible handling of nested documents which should help in picking max_depth/nesting
        }
    }

    /// The function `docs_to_record_batch` converts a vector of documents into a RecordBatch based on a
    /// given schema.
    ///
    /// Arguments:
    ///
    /// * `docs`: The `docs` parameter is a slice of `Document` instances that contain the data to be
    ///   converted into a `RecordBatch`.
    ///
    /// * `schema`: The `schema` parameter represents the schema of the data that will be converted into a `RecordBatch`.
    ///
    /// Returns:
    ///
    /// The function `docs_to_record_batch` returns a `Result` containing a `RecordBatch` or a
    /// `NisabaError`.
    fn docs_to_record_batch(
        &self,
        docs: &[Document],
        schema: Arc<Schema>,
    ) -> Result<RecordBatch, NisabaError> {
        let mut columns: Vec<ArrayRef> = Vec::new();

        for field in schema.fields() {
            let field_name = field.name();

            match field.data_type() {
                DataType::Null => {
                    columns.push(Arc::new(NullArray::new(docs.len())));
                }
                DataType::Boolean => {
                    let values: Vec<Option<bool>> = docs
                        .iter()
                        .map(|doc| doc.get(field_name).and_then(|v| v.as_bool()))
                        .collect();

                    columns.push(Arc::new(BooleanArray::from(values)));
                }
                DataType::Int32 => {
                    let values: Vec<Option<i32>> = docs
                        .iter()
                        .map(|doc| doc.get(field_name).and_then(|v| v.as_i32()))
                        .collect();

                    columns.push(Arc::new(Int32Array::from(values)));
                }
                DataType::Int64 => {
                    let values: Vec<Option<i64>> = docs
                        .iter()
                        .map(|doc| doc.get(field_name).and_then(|v| v.as_i64()))
                        .collect();

                    columns.push(Arc::new(Int64Array::from(values)));
                }

                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    let values: Vec<Option<i64>> = docs
                        .iter()
                        .map(|doc| {
                            doc.get(field_name)
                                .and_then(|v| v.as_datetime().map(|v| v.timestamp_millis()))
                        })
                        .collect();

                    columns.push(Arc::new(TimestampMillisecondArray::from(values)));
                }
                DataType::Decimal128(_, _) => {
                    let values: Vec<Option<i128>> = docs
                        .iter()
                        .map(|doc| {
                            doc.get(field_name).and_then(|v| match v {
                                Bson::Decimal128(d) => Some(i128::from_le_bytes(d.bytes())),
                                _ => None,
                            })
                        })
                        .collect();

                    columns.push(Arc::new(Decimal128Array::from(values)));
                }
                DataType::Binary => {
                    let values: Vec<Option<Vec<u8>>> = docs
                        .iter()
                        .map(|doc| {
                            doc.get(field_name).and_then(|v| match v {
                                Bson::Binary(b) => Some(b.bytes.clone()),
                                _ => None,
                            })
                        })
                        .collect();

                    columns.push(Arc::new(BinaryArray::from_iter(
                        values.iter().map(|v| v.as_deref()),
                    )));
                }
                DataType::Float64 => {
                    let values: Vec<Option<f64>> = docs
                        .iter()
                        .map(|doc| doc.get(field_name).and_then(|v| v.as_f64()))
                        .collect();

                    columns.push(Arc::new(Float64Array::from(values)));
                }
                _ => {
                    let values: Vec<Option<String>> = docs
                        .iter()
                        .map(|doc| {
                            let val = doc.get(field_name).and_then(|v| v.as_str());

                            val.map(|v| v.to_owned())
                        })
                        .collect();

                    columns.push(Arc::new(StringArray::from(values)));
                }
            }
        }

        Ok(RecordBatch::try_new(schema, columns)?)
    }
}

impl SchemaInferenceEngine for NoSQLInferenceEngine {
    fn engine_name(&self) -> &str {
        "mongodb"
    }
}

/// The function `bson_to_arrow_type` converts BSON data types to Arrow data types.
///
/// Arguments:
///
/// * `data_type`: The function `bson_to_arrow_type` takes a reference to a `Bson` enum and converts it
///   into an Arrow `DataType`.
///
/// Returns:
///
/// The function `bson_to_arrow_type` returns a `DataType` enum based on the input `Bson` enum. The
/// returned `DataType` corresponds to the Arrow data type that best represents the given BSON data
/// type.
fn bson_to_arrow_type(data_type: &Bson) -> DataType {
    match data_type {
        Bson::Array(arr) => {
            if arr.is_empty() {
                DataType::List(Arc::new(Field::new("item", DataType::Null, true)))
            } else {
                let elem_type = arr
                    .iter()
                    .map(bson_to_arrow_type)
                    .fold(DataType::Null, |acc, t| merge_types(&acc, &t));
                DataType::List(Arc::new(Field::new("item", elem_type, true)))
            }
        }
        Bson::Binary(_) => DataType::Binary,
        Bson::Boolean(_) => DataType::Boolean,
        Bson::DateTime(_) => DataType::Timestamp(TimeUnit::Millisecond, None),
        Bson::DbPointer(_) => DataType::Utf8,
        Bson::Decimal128(_) => DataType::Decimal128(38, 10),
        Bson::Document(_) => {
            DataType::Struct(Fields::from(vec![Field::new("item", DataType::Utf8, true)]))
        }
        Bson::Double(_) => DataType::Float64,
        Bson::Int32(_) => DataType::Int32,
        Bson::Int64(_) => DataType::Int64,
        Bson::JavaScriptCode(_) => DataType::Utf8,
        Bson::JavaScriptCodeWithScope(_) => DataType::Utf8,
        Bson::MaxKey => DataType::Utf8,
        Bson::MinKey => DataType::Utf8,
        Bson::Null => DataType::Null,
        Bson::ObjectId(_) => DataType::Utf8,
        Bson::RegularExpression(_) => DataType::Utf8,
        Bson::String(_) => DataType::Utf8,
        Bson::Symbol(_) => DataType::Utf8,
        Bson::Timestamp(_) => DataType::Timestamp(TimeUnit::Millisecond, None),
        Bson::Undefined => DataType::Utf8,
    }
}

/// The `merge_types` function merges two data types based on specific rules and returns the
/// resulting data type.
///
/// Arguments:
///
/// * `type_a`: `&DataType` - a reference to the first data type to be merged.
///
/// * `type_b`: `&DataType` - a reference to the second data type to be merged.
///
/// Returns:
///
/// The `merge_types` function returns a `DataType` value based on the input types `type_a` and
/// `type_b`. The function matches the input types with various patterns and returns the corresponding
/// merged data type. If none of the specific patterns match, it defaults to returning `DataType::Utf8`.
fn merge_types(type_a: &DataType, type_b: &DataType) -> DataType {
    match (type_a, type_b) {
        (a, b) if a == b => a.clone(),
        (DataType::Null, other) | (other, DataType::Null) => other.clone(),
        (DataType::Decimal128(p, s), DataType::Int32)
        | (DataType::Int32, DataType::Decimal128(p, s)) => DataType::Decimal128(*p, *s),
        (DataType::Decimal128(p, s), DataType::Int64)
        | (DataType::Int64, DataType::Decimal128(p, s)) => DataType::Decimal128(*p, *s),
        (DataType::Int32, DataType::Int64) | (DataType::Int64, DataType::Int32) => DataType::Int64,
        (DataType::Int32, DataType::Float64) | (DataType::Float64, DataType::Int32) => {
            DataType::Float64
        }
        (DataType::Decimal128(p, s), DataType::Float32)
        | (DataType::Float32, DataType::Decimal128(p, s)) => DataType::Decimal128(*p, *s),
        (DataType::Decimal128(p, s), DataType::Float64)
        | (DataType::Float64, DataType::Decimal128(p, s)) => DataType::Decimal128(*p, *s),
        (DataType::Int32, DataType::Float32) | (DataType::Float32, DataType::Int32) => {
            DataType::Float32
        }
        (DataType::List(f1), DataType::List(f2)) => {
            let elem_type = merge_types(f1.data_type(), f2.data_type());
            DataType::List(Arc::new(Field::new("item", elem_type, true)))
        }
        _ => DataType::Utf8,
    }
}

#[cfg(test)]
mod tests {

    use crate::{AnalyzerConfig, LatentStore, analyzer::datastore::Source, types::FieldDef};

    use super::*;

    #[tokio::test]
    async fn test_mongo_inference() {
        let source = Source::mongodb()
            .auth("mongodb", "mongodb")
            .host("localhost")
            .database("mongo_store")
            .pool_size(5)
            .port(27017)
            .build()
            .unwrap();

        let stats = Arc::new(Mutex::new(InferenceStats::default()));

        let latent_store = Arc::new(
            LatentStore::builder()
                .analyzer_config(Arc::new(AnalyzerConfig::default()))
                .build()
                .await
                .unwrap(),
        );

        let table_handler = latent_store.table_handler::<TableRep>();

        table_handler.initialize().await.unwrap();

        let field_handler = latent_store.table_handler::<FieldDef>();

        field_handler.initialize().await.unwrap();

        let nosql_inference = NoSQLInferenceEngine::default();

        let result = nosql_inference
            .mongodb_store_infer(
                &source,
                stats,
                Arc::new(|table_defs| async {
                    table_handler.store_tables(table_defs).await?;
                    Ok(())
                }),
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 9);
    }
}
