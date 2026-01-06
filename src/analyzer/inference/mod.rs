use arrow::{
    array::{
        Array, AsArray, GenericStringArray, LargeBinaryArray, LargeStringArray, RecordBatch,
        StringArray,
    },
    datatypes::{DataType, Field, Int64Type, Schema, TimeUnit, TimestampNanosecondType},
    error::ArrowError,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use uuid::Uuid;

mod file;
mod nosql;
mod promote;
mod sql;

pub use file::FileInferenceEngine;
pub use nosql::NoSQLInferenceEngine;
pub use sql::SQLInferenceEngine;

use crate::{
    analyzer::catalog::{StorageBackend, StorageConfig},
    error::NError,
    types::{FieldDef, TableDef},
};

// =================================================
// Inference Engine Registry
// =================================================
/// Registry of available inference engines
///
#[allow(dead_code)]
#[derive(Debug)]
pub struct InferenceEngineRegistry {
    engines: HashMap<String, Box<dyn SchemaInferenceEngine>>,
}

impl Default for InferenceEngineRegistry {
    fn default() -> Self {
        let file_engine = FileInferenceEngine::new();
        let sql_engine = SQLInferenceEngine::new();
        let nosql_engine = NoSQLInferenceEngine::new();

        let mut engines: HashMap<String, Box<dyn SchemaInferenceEngine>> = HashMap::new();

        engines.insert(
            file_engine.engine_name().to_lowercase(),
            Box::new(file_engine),
        );

        engines.insert(
            sql_engine.engine_name().to_lowercase(),
            Box::new(sql_engine),
        );

        engines.insert(
            nosql_engine.engine_name().to_lowercase(),
            Box::new(nosql_engine),
        );

        Self { engines }
    }
}

impl InferenceEngineRegistry {
    pub fn new() -> Self {
        InferenceEngineRegistry::default()
    }

    pub fn size(&self) -> usize {
        self.engines.len()
    }

    /// The function `add_engine` adds a schema inference engine to a collection based on its engine
    /// name. Can be used to replace an existing engine
    ///
    /// Arguments:
    ///
    /// * `engine`: The `engine` parameter is a `Box` containing a trait object of type `SchemaInferenceEngine`.
    pub fn add_engine(&mut self, engine: Box<dyn SchemaInferenceEngine>) {
        self.engines
            .insert(engine.engine_name().to_lowercase(), engine);
    }

    /// The function `get_engine` returns an optional reference to a `SchemaInferenceEngine` based on
    /// the provided `DataStoreType`.
    ///
    /// Arguments:
    ///
    /// * `store_type`: The `store_type` parameter is a reference to a `DataStoreType` enum that is used
    ///   to identify the type of data store for which you want to retrieve the schema inference engine.
    ///
    /// Returns:
    ///
    /// The `get_engine` function returns an `Option` containing a reference to a `dyn
    /// SchemaInferenceEngine` trait object based on the provided `DataStoreType`.
    pub fn get_engine(&self, backend: &StorageBackend) -> Option<&dyn SchemaInferenceEngine> {
        self.engines
            .values()
            .find(|eng| eng.can_handle(backend))
            .map(|eng| eng.as_ref())
    }

    /// The `infer_schema` function infers table schemas based on the data location using the
    /// specified engine.
    ///
    /// Arguments:
    ///
    /// * `location`: The `location` parameter in the `infer_schema` function represents the location of
    ///   the data from which you want to infer the schema. It includes information such as the type of
    ///   store where the data is located (e.g., file system, database), the path or connection details to
    ///   access the data,
    ///
    /// Returns:
    ///
    /// A Result containing a vector of TableSchema objects or an NError if there is an issue with the
    /// operation.
    pub fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NError> {
        let engine = self.get_engine(&config.backend).ok_or_else(|| {
            NError::Unsupported(format!(
                "No engine available for store type: {:?}",
                config.backend
            ))
        })?;

        let table_defs = engine.infer_schema(config)?;

        Ok(table_defs)
    }

    pub fn discover_ecosystem(&self, configs: Vec<StorageConfig>) -> Result<Vec<TableDef>, NError> {
        let mut table_defs = Vec::new();

        for config in configs {
            table_defs.extend(self.infer_schema(&config)?);
        }

        Ok(table_defs)
    }
}

/// Trait for schema inference engines
pub trait SchemaInferenceEngine: std::fmt::Debug + Send + Sync {
    /// Infer schema from a data source
    fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NError>;

    /// Check if this engine can handle the given data source
    fn can_handle(&self, backend: &StorageBackend) -> bool;

    /// Get the name of the inference engine
    fn engine_name(&self) -> &str;
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct SourceField {
    pub silo_id: String,
    pub table_schema: String,
    pub table_name: String,
    pub column_name: String,
    pub column_default: Option<String>,
    pub is_nullable: String,
    pub data_type: String,
    pub numeric_precision: Option<i32>,
    pub numeric_scale: Option<i32>,
    pub datetime_precision: Option<i32>,
    pub character_maximum_length: Option<i32>,
    pub udt_name: String,
}

pub fn convert_into_table_defs(schemas: Vec<SourceField>) -> Result<Vec<TableDef>, NError> {
    if schemas.is_empty() {
        return Ok(Vec::new());
    }

    let fields: Vec<FieldDef> = schemas
        .into_iter()
        .map(|v| {
            let is_nullable = v.is_nullable.trim().eq_ignore_ascii_case("yes");

            FieldDef {
                id: Uuid::now_v7(),
                silo_id: v.silo_id,
                name: v.column_name,
                table_name: v.table_name,
                canonical_type: sql_to_arrow_type(
                    &v.udt_name,
                    v.numeric_precision,
                    v.numeric_scale,
                    v.datetime_precision,
                ),
                type_confidence: None,
                cardinality: None,
                avg_byte_length: None,
                is_monotonic: false,
                char_class_signature: None,
                column_default: v.column_default,
                is_nullable,
                char_max_length: v.character_maximum_length,
                numeric_precision: v.numeric_precision,
                numeric_scale: v.numeric_scale,
                datetime_precision: v.datetime_precision,
            }
        })
        .collect();

    let mut map: HashMap<String, TableDef> = HashMap::new();

    for f in fields {
        map.entry(f.table_name.clone())
            .and_modify(|v| v.fields.push(f.clone()))
            .or_insert(TableDef {
                silo_id: f.silo_id.clone(),
                id: Uuid::now_v7(),
                name: f.table_name.clone(),
                fields: vec![f],
            });
        // TODO: nested and max-depth will have to be updated later
    }

    let map: Vec<TableDef> = map.into_values().collect();

    Ok(map)
}

fn sql_to_arrow_type(
    data_type: &str,
    numeric_precision: Option<i32>,
    numeric_scale: Option<i32>,
    datetime_precision: Option<i32>,
) -> DataType {
    match data_type.to_lowercase().trim() {
        // Signed Integer Types
        "tinyint" => DataType::Int8,
        "integer" | "int" | "int4" | "serial4" | "serial" => DataType::Int32,
        "smallint" | "int2" | "serial2" | "smallserial" => DataType::Int16,

        "bigint" => DataType::Int64,
        // TODO: For array, the subtype can be picked
        "array" | "vector" | "tsvector" => {
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false)))
        }

        //Boolean
        "bool" | "boolean" => DataType::Boolean,
        "numeric" | "decimal" => {
            let precision = numeric_precision.unwrap_or(38) as u8;
            let scale = numeric_scale.unwrap_or(10) as i8;

            DataType::Decimal128(precision, scale)
        }

        "float2" => DataType::Float16,
        "real" | "float4" => DataType::Float32,
        "double" | "float" | "float8" => DataType::Float64,
        "date" => DataType::Date32,
        "bytea" => DataType::Binary,
        "uuid" => DataType::FixedSizeBinary(16),
        "jsonb" => DataType::LargeBinary,
        "json" | "text" => DataType::LargeUtf8,
        "time" | "timetz" => {
            if let Some(p) = datetime_precision {
                match p {
                    0 => DataType::Time32(TimeUnit::Second),
                    1..=3 => DataType::Time32(TimeUnit::Millisecond),
                    4..=6 => DataType::Time64(TimeUnit::Microsecond),
                    _ => DataType::Time32(TimeUnit::Microsecond),
                }
            } else {
                DataType::Time32(TimeUnit::Microsecond)
            }
        }

        "timestamp" | "timestamptz" => {
            if let Some(p) = datetime_precision {
                match p {
                    0 => DataType::Timestamp(TimeUnit::Second, None),
                    1..=3 => DataType::Timestamp(TimeUnit::Millisecond, None),
                    4..=6 => DataType::Timestamp(TimeUnit::Microsecond, None),
                    _ => DataType::Timestamp(TimeUnit::Microsecond, None),
                }
            } else {
                DataType::Timestamp(TimeUnit::Microsecond, None)
            }
        }

        // Unsigned Integer Types
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        // TODO: it probably maps to Decimal
        "uint128" => DataType::UInt64,
        "character" | "char" | "character varying" | "varchar" | "string" => DataType::Utf8,
        "null" => DataType::Null,
        _ => DataType::Utf8,
    }
}

// ===============================
// Metric Computation Functions
// ===============================
#[derive(Debug, Clone)]
pub struct FieldMetrics {
    pub char_class_signature: [f32; 4],
    pub monotonicity: bool,
    pub avg_byte_length: Option<f32>,
    pub cardinality: f32,
}

pub fn compute_field_metrics(batch: &RecordBatch) -> Result<HashMap<String, FieldMetrics>, NError> {
    let mut metrics = HashMap::new();

    let schema = batch.schema();

    for (index, field) in schema.fields().iter().enumerate() {
        let column = batch.column(index);

        let char_class_signature = compute_char_class_signature(column);
        let monotonicity = detect_monotonicity(column);
        let avg_byte_length = compute_avg_byte_length(column)?;
        let cardinality = compute_cardinality(column)?;

        metrics.insert(
            field.name().clone(),
            FieldMetrics {
                char_class_signature,
                monotonicity,
                avg_byte_length,
                cardinality,
            },
        );
    }

    Ok(metrics)
}

fn compute_char_class_signature(samples: &dyn Array) -> [f32; 4] {
    match samples.data_type() {
        DataType::LargeUtf8 | DataType::Utf8 => {
            let arr = samples.as_any().downcast_ref::<GenericStringArray<i64>>();

            match arr {
                Some(ar) => {
                    let mut totals = [0f32; 4];
                    let mut char_count = 0f32;

                    let _ = (0..ar.len()).filter(|&i| !ar.is_null(i)).map(|i| {
                        let r = ar.value(i);

                        for ch in r.chars() {
                            char_count += 1.0;

                            if ch.is_ascii_digit() {
                                totals[0] += 1.0;
                            } else if ch.is_alphabetic() {
                                totals[1] += 1.0;
                            } else if ch.is_ascii_whitespace() {
                                totals[2] += 1.0
                            } else {
                                totals[3] += 1.0
                            }
                        }
                    });

                    if char_count == 0.0 {
                        return [0.0; 4];
                    }

                    [
                        totals[0] / char_count,
                        totals[1] / char_count,
                        totals[2] / char_count,
                        totals[3] / char_count,
                    ]
                }
                None => [0.0; 4],
            }
        }
        _ => [0.0; 4],
    }
}

fn detect_monotonicity(samples: &dyn Array) -> bool {
    if samples.is_empty() {
        return true;
    }

    if samples.len() < 2 {
        return false;
    }

    match samples.data_type() {
        DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let samples = samples.as_primitive::<Int64Type>();

            let mut values: Vec<i64> = (0..samples.len())
                .filter(|&i| !samples.is_null(i))
                .map(|i| samples.value(i))
                .collect();

            values.sort_unstable();
            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Timestamp { .. } => {
            let timestamps = samples.as_primitive::<TimestampNanosecondType>();

            let mut values: Vec<i64> = (0..timestamps.len())
                .filter(|&i| !timestamps.is_null(i))
                .map(|i| timestamps.value(i))
                .collect();

            values.sort_unstable();
            values.windows(2).all(|w| w[1] > w[0])
        }
        _ => false,
    }
}

fn compute_cardinality(samples: &dyn Array) -> Result<f32, NError> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    if samples.len() < 2 {
        return Ok(1.0);
    }

    let values: HashSet<_> = samples.as_string::<i64>().iter().flatten().collect();

    let cc = values.len() as f32 / samples.len() as f32;

    Ok(cc)
}

fn compute_avg_byte_length(samples: &dyn Array) -> Result<Option<f32>, NError> {
    match samples.data_type() {
        DataType::Utf8 => {
            let string_arr =
                samples
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or(ArrowError::CastError(
                        "Failed to cast array to string".into(),
                    ))?;
            let total_len: usize = string_arr.iter().flatten().map(|s| s.len()).sum();

            let count = string_arr.len() - string_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f32 / count as f32))
        }
        DataType::LargeUtf8 => {
            let string_arr = samples.as_any().downcast_ref::<LargeStringArray>().ok_or(
                ArrowError::CastError("Failed to cast array to large string".into()),
            )?;

            let total_len: usize = string_arr.iter().flatten().map(|s| s.len()).sum();

            let count = string_arr.len() - string_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f32 / count as f32))
        }
        DataType::Binary | DataType::LargeBinary => {
            let binary_arr = samples.as_any().downcast_ref::<LargeBinaryArray>().ok_or(
                ArrowError::CastError("Failed to cast array to binary array".into()),
            )?;

            let total_len: usize = binary_arr.iter().flatten().map(|s| s.len()).sum();

            let count = binary_arr.len() - binary_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f32 / count as f32))
        }

        _ => Ok(None),
    }
}

fn table_def_to_arrow_schema(table_def: &TableDef) -> Arc<Schema> {
    let fields: Vec<Field> = table_def
        .fields
        .iter()
        .map(|v| Field::new(v.name.clone(), v.canonical_type.clone(), v.is_nullable))
        .collect();

    Arc::new(Schema::new(fields))
}
