use arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use futures::{TryStreamExt, executor::block_on};
use mongodb::{
    Client,
    bson::{Bson, Document, doc},
    options::ClientOptions,
};

use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        catalog::{DataLocation, DataStoreType},
        inference::{SchemaInferenceEngine, SourceField, convert_into_table_defs},
    },
    error::NError,
    types::TableDef,
};

// =================================================
// NoSQL-Like Inference Engine
// =================================================

/// NoSQL inference engine for MongoDB
#[derive(Debug)]
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

    pub fn with_sample_size(mut self, size: u32) -> Self {
        self.sample_size = size;
        self
    }

    async fn infer_from_mongodb(
        &self,
        location: &DataLocation,
    ) -> Result<Vec<SourceField>, NError> {
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());
        let conn_str = location.connection_string()?;
        let client_options = ClientOptions::parse(&conn_str).await?;
        let db_name = client_options.clone().default_database.unwrap();

        let client = Client::with_options(client_options)?;
        let db = client.database(&db_name);
        let collections = db.list_collection_names().await?;

        let mut source_fields: Vec<SourceField> = Vec::new();

        for collection_name in collections {
            let collection = db.collection::<Document>(&collection_name);
            let mut cursor = collection
                .find(doc! {})
                .limit(i64::from(self.sample_size))
                .await?;

            let result = self
                .mongo_collection_infer(&db_name, &collection_name, &mut cursor, &silo_id)
                .await?;

            source_fields.extend(result);
        }

        Ok(source_fields)
    }

    async fn mongo_collection_infer(
        &self,
        db_name: &str,
        collection_name: &str,
        cursor: &mut mongodb::Cursor<Document>,
        silo_id: &str,
    ) -> Result<Vec<SourceField>, NError> {
        let mut field_types: HashMap<String, DataType> = HashMap::new();

        while let Some(doc) = cursor.try_next().await? {
            self.mongo_document_infer(&doc, &mut field_types);
        }

        let fields: Vec<SourceField> = field_types
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

        Ok(fields)
    }

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
}

impl SchemaInferenceEngine for NoSQLInferenceEngine {
    fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let source_fields = match location.store_type {
            DataStoreType::MongoDB => block_on(self.infer_from_mongodb(location)),
            _ => Err(NError::Unsupported(format!(
                "{:?} NoSQL store provided unsupported by NoSQL engine",
                location.store_type
            ))),
        };

        convert_into_table_defs(source_fields?)
    }
    fn can_handle(&self, store_type: &DataStoreType) -> bool {
        matches!(store_type, DataStoreType::MongoDB)
    }
    fn engine_name(&self) -> &str {
        "NoSQL-MongDB"
    }
}

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
