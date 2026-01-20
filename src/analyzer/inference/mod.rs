use arrow::{
    array::{
        Array, AsArray, Date32Array, Date64Array, FixedSizeBinaryArray, GenericStringArray,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{
        DataType, Field, Int16Type, Int32Type, Int64Type, Schema, TimeUnit,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    },
    error::ArrowError,
};
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
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
    error::NisabaError,
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
    /// A Result containing a vector of TableSchema objects or an NisabaError if there is an issue with the
    /// operation.
    pub fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        let engine = self.get_engine(&config.backend).ok_or_else(|| {
            NisabaError::Unsupported(format!(
                "No engine available for store type: {:?}",
                config.backend
            ))
        })?;

        let table_defs = engine.infer_schema(config)?;

        Ok(table_defs)
    }

    pub fn discover_ecosystem(
        &self,
        configs: Vec<StorageConfig>,
    ) -> Result<Vec<TableDef>, NisabaError> {
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
    fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError>;

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

pub fn convert_into_table_defs(schemas: Vec<SourceField>) -> Result<Vec<TableDef>, NisabaError> {
    if schemas.is_empty() {
        return Ok(Vec::new());
    }

    let fields: Result<Vec<FieldDef>, NisabaError> = schemas
        .into_iter()
        .map(|v| {
            let is_nullable = v.is_nullable.trim().eq_ignore_ascii_case("yes");

            Ok(FieldDef {
                id: Uuid::now_v7(),
                silo_id: v.silo_id,
                name: v.column_name,
                table_name: v.table_name,
                canonical_type: sql_to_arrow_type(
                    &v.udt_name,
                    v.numeric_precision,
                    v.numeric_scale,
                    v.datetime_precision,
                )?,
                type_confidence: Some(0.98),
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
            })
        })
        .collect();

    let fields = fields?;

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
) -> Result<DataType, NisabaError> {
    let normalized = data_type.to_lowercase();
    let normalized = &normalized.trim();
    match *normalized {
        // Signed Integer Types
        "tinyint" | "int8" => Ok(DataType::Int8),
        "integer" | "int" | "int4" | "serial4" | "serial" | "int32" => Ok(DataType::Int32),
        "smallint" | "int2" | "serial2" | "smallserial" | "int16" => Ok(DataType::Int16),
        "bigint" | "int64" => Ok(DataType::Int64),
        // TODO: For array, the subtype can be picked
        "array" | "vector" | "tsvector" => Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            false,
        )))),

        //Boolean
        "bool" | "boolean" => Ok(DataType::Boolean),
        "numeric" | "decimal" => {
            let precision = numeric_precision.unwrap_or(38) as u8;
            let scale = numeric_scale.unwrap_or(10) as i8;

            Ok(DataType::Decimal128(precision, scale))
        }

        "float2" | "float16" => Ok(DataType::Float16),
        "float4" | "float32" => Ok(DataType::Float32),
        "real" | "double" | "float" | "float8" | "float64" => Ok(DataType::Float64),
        "date" | "date32" => Ok(DataType::Date32),
        "date64" => Ok(DataType::Date64),
        "bytea" | "binary" | "blob" => Ok(DataType::Binary),
        "uuid" => Ok(DataType::FixedSizeBinary(16)),
        "jsonb" | "largebinary" => Ok(DataType::LargeBinary),
        "json" | "text" => Ok(DataType::Utf8),
        "time" | "timetz" => {
            if let Some(p) = datetime_precision {
                match p {
                    0 => Ok(DataType::Time32(TimeUnit::Second)),
                    1..=3 => Ok(DataType::Time32(TimeUnit::Millisecond)),
                    4..=6 => Ok(DataType::Time64(TimeUnit::Microsecond)),
                    _ => Ok(DataType::Time32(TimeUnit::Microsecond)),
                }
            } else {
                Ok(DataType::Time32(TimeUnit::Microsecond))
            }
        }

        "timestamp" | "timestamptz" => {
            if let Some(p) = datetime_precision {
                match p {
                    0 => Ok(DataType::Timestamp(TimeUnit::Second, None)),
                    1..=3 => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
                    4..=6 => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
                    _ => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
                }
            } else {
                Ok(DataType::Timestamp(TimeUnit::Microsecond, None))
            }
        }

        // Unsigned Integer Types
        "uint8" => Ok(DataType::UInt8),
        "uint16" => Ok(DataType::UInt16),
        "uint32" => Ok(DataType::UInt32),
        "uint64" => Ok(DataType::UInt64),
        // TODO: it probably maps to Decimal
        "uint128" => Ok(DataType::UInt64),
        "character" | "char" | "character varying" | "varchar" | "string" | "utf8" => {
            Ok(DataType::Utf8)
        }
        "null" => Ok(DataType::Null),

        // Complex Arrow String to Arrow
        "Utf8View" => Ok(DataType::Utf8View),
        s if s.starts_with("timestamp(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("decimal256(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("decimal128") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("decimal64(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("decimal32(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("list(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("largelist(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("largelistview(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("listview(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("fixedsizelist(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("time64(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("time32(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("duration(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("interval(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("struct(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("union(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("dictionary(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("map(") => Ok(DataType::from_str(data_type)?),
        s if s.starts_with("runendencoded(") => Ok(DataType::from_str(data_type)?),
        _ => Ok(DataType::Utf8),
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

pub fn compute_field_metrics(
    batch: &RecordBatch,
) -> Result<HashMap<String, FieldMetrics>, NisabaError> {
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
        DataType::Int16 => {
            let samples = samples.as_primitive::<Int16Type>();

            let mut values: Vec<i16> = (0..samples.len())
                .filter(|&i| !samples.is_null(i))
                .map(|i| samples.value(i))
                .collect();

            values.sort_unstable();

            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Int32 => {
            let samples = samples.as_primitive::<Int32Type>();

            let mut values: Vec<i32> = (0..samples.len())
                .filter(|&i| !samples.is_null(i))
                .map(|i| samples.value(i))
                .collect();

            values.sort_unstable();

            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Int64 => {
            let samples = samples.as_primitive::<Int64Type>();

            let mut values: Vec<i64> = (0..samples.len())
                .filter(|&i| !samples.is_null(i))
                .map(|i| samples.value(i))
                .collect();

            values.sort_unstable();

            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let timestamps = samples.as_primitive::<TimestampSecondType>();

            let mut values: Vec<i64> = (0..timestamps.len())
                .filter(|&i| !timestamps.is_null(i))
                .map(|i| timestamps.value(i))
                .collect();

            values.sort_unstable();
            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let timestamps = samples.as_primitive::<TimestampMillisecondType>();

            let mut values: Vec<i64> = (0..timestamps.len())
                .filter(|&i| !timestamps.is_null(i))
                .map(|i| timestamps.value(i))
                .collect();

            values.sort_unstable();
            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let timestamps = samples.as_primitive::<TimestampMicrosecondType>();
            let mut values: Vec<i64> = (0..timestamps.len())
                .filter(|&i| !timestamps.is_null(i))
                .map(|i| timestamps.value(i))
                .collect();

            values.sort_unstable();
            values.windows(2).all(|w| w[1] > w[0])
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
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

fn compute_cardinality(samples: &dyn Array) -> Result<f32, NisabaError> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    if samples.len() < 2 {
        return Ok(1.0);
    }

    macro_rules! count_unique_vals {
        ($arr:expr, $array_type:ty) => {{
            let arr = $arr
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or(ArrowError::CastError(format!(
                    "Failed to cast to {}",
                    stringify!($array_type)
                )))?;

            let mut values: HashSet<_> = HashSet::new();

            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.insert(arr.value(i));
                }
            }

            values.len()
        }};
    }

    let unique_count = match samples.data_type() {
        DataType::Null => 0,
        DataType::Binary => {
            let values: HashSet<&[u8]> = samples.as_binary::<i32>().iter().flatten().collect();
            values.len()
        }
        DataType::BinaryView => {
            let values: HashSet<&[u8]> = samples.as_binary_view().iter().flatten().collect();

            values.len()
        }
        DataType::LargeBinary => count_unique_vals!(samples, LargeBinaryArray),
        DataType::Boolean => 2,
        DataType::Utf8 | DataType::Utf8View => {
            let values: HashSet<_> = samples.as_string::<i32>().iter().flatten().collect();

            values.len()
        }
        DataType::LargeUtf8 => count_unique_vals!(samples, LargeStringArray),
        DataType::Int8 => count_unique_vals!(samples, Int8Array),
        DataType::Int16 => count_unique_vals!(samples, Int16Array),
        DataType::Int32 => count_unique_vals!(samples, Int32Array),
        DataType::Int64 => count_unique_vals!(samples, Int64Array),
        DataType::UInt8 => count_unique_vals!(samples, UInt8Array),
        DataType::UInt16 => count_unique_vals!(samples, UInt16Array),
        DataType::UInt32 => count_unique_vals!(samples, UInt32Array),
        DataType::UInt64 => count_unique_vals!(samples, UInt64Array),
        DataType::Date32 => count_unique_vals!(samples, Date32Array),
        DataType::Date64 => count_unique_vals!(samples, Date64Array),
        DataType::FixedSizeBinary(_) => count_unique_vals!(samples, FixedSizeBinaryArray),

        // Safe assumption that cardinality in these types is near perfect if not perfect
        DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Dictionary(_, _)
        | DataType::FixedSizeList(_, _)
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::RunEndEncoded(_, _)
        | DataType::Struct(_)
        | DataType::Union(_, _)
        | DataType::Duration(_)
        | DataType::Map(_, _)
        | DataType::List(_)
        | DataType::ListView(_)
        | DataType::Timestamp(_, _)
        | DataType::Interval(_)
        | DataType::LargeListView(_)
        | DataType::LargeList(_)
        | DataType::Time32(_)
        | DataType::Time64(_) => samples.len(),
    };

    let cc = unique_count as f32 / samples.len() as f32;

    Ok(cc)
}

fn compute_avg_byte_length(samples: &dyn Array) -> Result<Option<f32>, NisabaError> {
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
