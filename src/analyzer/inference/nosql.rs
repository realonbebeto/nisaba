use arrow::{
    array::{
        ArrayRef, BinaryArray, BooleanArray, Decimal128Array, Float64Array, Int32Array, Int64Array,
        NullArray, RecordBatch, StringArray, TimestampMillisecondArray,
    },
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
};
use futures::{TryStreamExt, executor::block_on};
use mongodb::{
    Client,
    bson::{Bson, Document, doc},
};
use tokio::runtime::Runtime;

use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        catalog::{StorageBackend, StorageConfig},
        inference::{
            SchemaInferenceEngine, SourceField, compute_field_metrics, convert_into_table_defs,
            promote::{ColumnStats, TypeLatticeResolver, cast_utf8_column},
        },
    },
    error::NisabaError,
    types::TableDef,
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
    sample_size: u32,
}

impl Default for NoSQLInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl NoSQLInferenceEngine {
    pub fn new() -> Self {
        NoSQLInferenceEngine::default()
    }

    /// The `with_sample_size` function sets the sample size for a data structure and returns
    /// the modified structure.
    ///
    /// Arguments:
    ///
    /// * `size`: The `size` parameter in the `with_sample_size` function represents the sample size
    ///   that you want to set for the object. This parameter allows you to specify the size of the samples
    ///   that will be used in inference.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after updating the `sample_size` field with the provided
    /// `size` value.
    pub fn with_sample_size(mut self, size: u32) -> Self {
        self.sample_size = size;
        self
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
    /// The function `infer_from_mongodb` returns a `Result` containing a vector of `TableDef` structs
    /// or a `NisabaError`.
    async fn infer_from_mongodb(
        &self,
        config: &StorageConfig,
    ) -> Result<Vec<TableDef>, NisabaError> {
        let silo_id = format!("{}-{}", config.backend, Uuid::now_v7());
        let conn_str = config.connection_string()?;
        let db_name = config.clone().database.unwrap();

        let rt = Runtime::new()?;

        let table_defs: Result<Vec<TableDef>, NisabaError> = rt.block_on(async {
            let client = Client::with_uri_str(conn_str).await?;

            let db = client.database(&db_name);
            let collections = db.list_collection_names().await?;

            let mut table_defs: Vec<TableDef> = Vec::new();

            for collection_name in collections {
                let collection = db.collection::<Document>(&collection_name);
                let cursor = collection
                    .find(doc! {})
                    .limit(i64::from(self.sample_size))
                    .await?;

                let docs: Vec<Document> = cursor.try_collect().await?;

                let (result, schema) =
                    self.mongo_collection_infer(&db_name, &collection_name, &docs, &silo_id)?;

                let mut batch = self.docs_to_record_batch(&docs, schema.clone())?;

                let mut table_def = convert_into_table_defs(result)?;

                // Promotion

                for (index, field) in schema.fields().iter().enumerate() {
                    let column = batch.column(index);

                    match column.data_type() {
                        DataType::LargeUtf8
                        | DataType::Utf8
                        | DataType::Int32
                        | DataType::Int64 => {
                            let stats = ColumnStats::new(column);
                            let resolver = TypeLatticeResolver::new();
                            let resolved_result = resolver.promote(column.data_type(), &stats)?;

                            if let Some(ff) = table_def[0]
                                .fields
                                .iter_mut()
                                .find(|f| f.name == *field.name())
                            {
                                ff.type_confidence = Some(resolved_result.confidence);

                                // Update char_max_length
                                match (&ff.canonical_type, &resolved_result.dest_type) {
                                    (DataType::Utf8, DataType::Utf8)
                                    | (DataType::LargeUtf8, DataType::LargeUtf8)
                                    | (DataType::Utf8View, DataType::Utf8View) => {
                                        ff.char_max_length =
                                            resolved_result.character_maximum_length;
                                    }

                                    (_, _) => {}
                                }

                                // Update type related signals when there is a mismatch on types
                                if ff.canonical_type != resolved_result.dest_type {
                                    ff.canonical_type = resolved_result.dest_type;
                                    ff.type_confidence = Some(resolved_result.confidence);
                                    ff.is_nullable = resolved_result.nullable;
                                    ff.char_max_length = resolved_result.character_maximum_length;
                                    ff.numeric_precision = resolved_result.numeric_precision;
                                    ff.numeric_scale = resolved_result.numeric_scale;
                                    ff.datetime_precision = resolved_result.datetime_precision;

                                    // Very important for field values in batch to be updated
                                    cast_utf8_column(&mut batch, &ff.name, &ff.canonical_type)?;
                                }
                            }
                        }
                        _ => {}
                    }
                }

                let metrics = compute_field_metrics(&batch)?;

                let _ = table_def[0].fields.iter_mut().map(|f| {
                    let fmetrics = metrics.get(&f.name);

                    if let Some(m) = fmetrics {
                        f.char_class_signature = Some(m.char_class_signature);
                        f.is_monotonic = m.monotonicity;
                        f.cardinality = Some(m.cardinality);
                        f.avg_byte_length = m.avg_byte_length
                    }
                });

                table_defs.extend(table_def);
            }
            Ok(table_defs)
        });

        table_defs
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
    /// The `infer_schema` function infers table schema from a MongoDB backend.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter is of type `StorageConfig`, which contains
    ///   configuration settings for the storage backend being used.
    ///
    /// Returns:
    ///
    /// A `Result` containing either a vector of `TableDef` or a `NisabaError` is being returned.
    fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        match config.backend {
            StorageBackend::MongoDB => block_on(self.infer_from_mongodb(config)),
            _ => Err(NisabaError::Unsupported(format!(
                "{:?} NoSQL store provided unsupported by NoSQL engine",
                config.backend
            ))),
        }
    }

    /// The function `can_handle` checks if the provided `StorageBackend` is `MongoDB`.
    ///
    /// Arguments:
    ///
    /// * `backend`: The `backend` parameter is of type `StorageBackend`, which is an enum representing
    ///   different types of storage backends.
    ///
    /// Returns:
    ///
    /// The function `can_handle` returns a boolean value indicating whether the provided
    /// `StorageBackend` is equal to `StorageBackend::MongoDB`.
    fn can_handle(&self, backend: &StorageBackend) -> bool {
        matches!(backend, StorageBackend::MongoDB)
    }

    fn engine_name(&self) -> &str {
        "NoSQL-MongDB"
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
    use super::*;

    #[test]
    fn test_mongo_inference() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::MongoDB,
            "localhost",
            27017,
            "mongo_store",
            "mongodb",
            "mongodb",
            None::<String>,
        )
        .unwrap();

        let nosql_inference = NoSQLInferenceEngine::default();

        let result = block_on(nosql_inference.infer_from_mongodb(&config)).unwrap();

        assert_eq!(result.len(), 1);

        // release_date is read in as Integer but parquet seems to store the semantic type
        let release_date = result
            .iter()
            .find(|t| t.name == "albums")
            .unwrap()
            .fields
            .iter()
            .find(|f| f.name == "release_date")
            .unwrap();

        assert!(matches!(release_date.canonical_type, DataType::Date32));
    }
}
