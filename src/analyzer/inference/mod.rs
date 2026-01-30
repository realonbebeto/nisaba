//! Necessary Inference implementations for common stores
//!
//!
//! Overview
//! - [`CastSafety`]: A rule based type to ensure type promotion is within allowable constraints.
//! - [`FileInferenceEngine`]: A type that is responsible for CSV, Excel and Parquet store inference.
//! - [`NoSQLInferenceEngine`]: A type that is repsonsible for MongoDB store inference.
//! - [`SQLInferenceEngine`]: A type responsible for MySQL, PostgreSQL, Sqlite store inference.
//!

use arrow::{
    array::{
        Array, AsArray, BooleanBuilder, Date32Array, Date64Array, FixedSizeBinaryArray,
        Float64Array, GenericStringArray, Int8Array, Int16Array, Int32Array, Int64Array,
        Int64Builder, LargeBinaryArray, LargeStringArray, RecordBatch, StringArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
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

pub use file::{CsvInferenceEngine, ExcelInferenceEngine, ParquetInferenceEngine};
pub use nosql::NoSQLInferenceEngine;
pub use sql::{MySQLInferenceEngine, PostgreSQLInferenceEngine, SqliteInferenceEngine};

use crate::{
    analyzer::inference::promote::{ColumnStats, TypeLatticeResolver, cast_utf8_column},
    error::NisabaError,
    types::{FieldDef, TableDef},
};

/// Trait for schema inference engines
pub trait SchemaInferenceEngine: std::fmt::Debug + Send + Sync {
    /// Get the name of the inference engine
    fn engine_name(&self) -> &str;

    fn enrich_table_def(
        &self,
        table_def: &mut TableDef,
        batch: &mut RecordBatch,
    ) -> Result<(), NisabaError> {
        // Enriching
        // 1. Type promotion and type confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature
        // 6. char_max_length
        // 7. numeric_precision
        // 8. numeric_scale
        // 9. datetime_precision

        let schema = batch.schema();

        let resolver = TypeLatticeResolver::new();

        for (index, field) in schema.fields().iter().enumerate() {
            let mut column = batch.column(index).clone();

            if self.engine_name() == "sqlite" {
                // Handling Boolean/Int64 from f64 for SQLite
                if column.data_type() == &DataType::Float64 {
                    let arr = column.as_any().downcast_ref::<Float64Array>().ok_or(
                        ArrowError::CastError("Failed to cast to Float64Array".into()),
                    )?;

                    if arr
                        .iter()
                        .flatten()
                        .collect::<Vec<f64>>()
                        .iter()
                        .all(|val| val.is_finite() && (*val == 0.0 || *val == 1.0))
                    {
                        let mut builder = BooleanBuilder::new();
                        for index in 0..arr.len() {
                            if arr.is_null(index) {
                                builder.append_null();
                            } else {
                                builder.append_value(arr.value(index) != 0.0);
                            }
                        }
                        column = Arc::new(builder.finish());
                    } else if arr
                        .iter()
                        .flatten()
                        .collect::<Vec<f64>>()
                        .iter()
                        .all(|val| {
                            val.is_finite()
                                && val.fract().abs() < f64::MIN_POSITIVE
                                && val.abs() < i64::MAX as f64
                        })
                    {
                        let mut builder = Int64Builder::new();

                        for index in 0..arr.len() {
                            if arr.is_null(index) {
                                builder.append_null();
                            } else {
                                builder.append_value(arr.value(index) as i64);
                            }
                        }

                        column = Arc::new(builder.finish());
                    }

                    if let Some(ff) = table_def
                        .fields
                        .iter_mut()
                        .find(|f| f.name == *field.name())
                    {
                        ff.canonical_type = column.data_type().clone();
                    }
                }
            }

            match column.data_type() {
                DataType::LargeUtf8 | DataType::Utf8 | DataType::Int32 | DataType::Int64 => {
                    let stats = ColumnStats::new(&column);
                    let resolved_result = resolver.promote(column.data_type(), &stats)?;

                    if let Some(ff) = table_def
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
                                ff.char_max_length = resolved_result.character_maximum_length;
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
                            cast_utf8_column(batch, &ff.name, &ff.canonical_type)?;
                        }
                    }
                }
                _ => {}
            }

            let metrics = compute_field_metrics(batch)?;

            for field in &mut table_def.fields {
                if let Some(m) = metrics.get(&field.name) {
                    field.char_class_signature = Some(m.char_class_signature);
                    field.is_monotonic = m.monotonicity;
                    field.cardinality = Some(m.cardinality);
                    field.avg_byte_length = m.avg_byte_length;
                }
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
/// The `SourceField` represents a field in a data source with various properties such as
/// column name, data type, and constraints.
pub struct SourceField {
    /// identifier of the silo to which the field belongs
    pub silo_id: String,
    /// schema or namespace to which the table belongs in a database.
    pub table_schema: String,
    /// name of the table to which the field belongs
    pub table_name: String,
    /// name of a column in a database table
    pub column_name: String,
    /// default value of the column
    pub column_default: Option<String>,
    /// flag if the column is nullable or not
    pub is_nullable: String,
    /// data type of the column
    pub data_type: String,
    /// precision of a numeric data type
    pub numeric_precision: Option<i32>,
    /// scale of a numeric data type
    pub numeric_scale: Option<i32>,
    /// precision of a datetime data type
    pub datetime_precision: Option<i32>,
    /// maximum length of character of the field
    pub character_maximum_length: Option<i32>,
    /// user-defined data type
    pub udt_name: String,
}

/// The function `convert_into_table_defs` converts a vector of source fields into table definitions and
/// organizes them into a vector of `TableDef` structs.
///
/// Arguments:
///
/// * `schemas`: The `schemas` parameter is a vector of `SourceField` structs. Each `SourceField` struct
///   contains information about a field in a data source, such as column name, data type, nullability,
///   etc.
///
/// Returns:
///
/// The function `convert_into_table_defs` returns a `Result` containing either a vector of `TableDef`
/// structs or a `NisabaError`.
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
                table_schema: v.table_schema,
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

/// The function `sql_to_arrow_type` converts SQL data types to Arrow data types with optional
/// precision and scale considerations.
///
/// Arguments:
///
/// * `data_type`: The `data_type` parameter represents the SQL data type that you want to convert to an
///   Arrow data type. It could be any valid SQL data type like `int`, `varchar`, `boolean`, etc.
///
/// * `numeric_precision`: The `numeric_precision` parameter represents the precision of a numeric data
///   type in SQL. It indicates the total number of digits that can be stored, including both the digits
///   before and after the decimal point.
///
/// * `numeric_scale`: The `numeric_scale` parameter represents the scale or the number of digits to the
///   right of the decimal point in a numeric data type. It is used to determine the precision and scale of
///   the `DataType::Decimal128` in the Arrow data type conversion.
///
/// * `datetime_precision`: The `datetime_precision` parameter refers to the precision of datetime data
///   types like `time`, `timestamp`, or `timestamptz`. It specifies the number of fractional digits in
///   the seconds part of the time value.
///
/// Returns:
///
/// The function `sql_to_arrow_type` returns a `Result` containing a `DataType` or a `NisabaError`.
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
        "datetime" => Ok(DataType::Timestamp(TimeUnit::Second, None)),
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
/// The `FieldMetrics represents metrics related to a field, including character class
/// signature, monotonicity, average byte length, and cardinality.
///
/// Properties:
///
/// * `char_class_signature`: The `char_class_signature` property is an array of 4 floating-point numbers (`f32`)
///   and represents some kind of signature or pattern related to character classes in the field data.
///
/// * `monotonicity`: the `monotonicity` field is a boolean value that indicates whether the values in the field exhibit a
///   monotonic pattern or not.
///
/// * `avg_byte_length`: The `avg_byte_length` property represents the average length of the bytes in the
///   field. It is an optional `f32` value, which means it can either hold a floating-point number or be `None`.
///
/// * `cardinality`: The `cardinality` property represents the number of unique values in a field. It is
///   a measure of the distinctiveness or diversity of values in the field.
pub struct FieldMetrics {
    pub char_class_signature: [f32; 4],
    pub monotonicity: bool,
    pub avg_byte_length: Option<f32>,
    pub cardinality: f32,
}

/// The function `compute_field_metrics` processes a RecordBatch to compute various metrics for each
/// field and returns a HashMap of field names mapped to their corresponding metrics.
///
/// Arguments:
///
/// * `batch`: The `compute_field_metrics` function takes a reference to a `RecordBatch` which represents
///   a collection of rows with a consistent schema.
///
/// Returns:
///
/// The `compute_field_metrics` function returns a `Result` containing a `HashMap` mapping field names
/// to `FieldMetrics`, or an error of type `NisabaError`.
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

/// The function `compute_char_class_signature` calculates the distribution of character classes in a
/// string array.
///
/// Arguments:
///
/// * `samples`: The function `compute_char_class_signature` takes a reference to a trait object
///   `samples` that implements the `Array` trait. The goal of this function is to compute a signature
///   based on the character classes present in the data contained in the `samples` array.
///
/// Returns:
///
/// The function `compute_char_class_signature` returns an array of 4 floating-point numbers `[f32; 4]`.
/// The array contains the ratios of different character classes (digits, alphabetic characters,
/// whitespace characters, and other characters) found in the input samples.
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

/// The function `detect_monotonicity` checks if the given array of samples is monotonically increasing.
///
/// Arguments:
///
/// * `samples`: The `detect_monotonicity` function takes a reference to a trait object `Array` as
///   input, which represents an array of samples with different data types. The function checks if the
///   samples are monotonically increasing based on their data type.
///
/// Returns:
///
/// The `detect_monotonicity` function returns a boolean value indicating whether the samples provided
/// in the input array are monotonic (i.e., always increasing) or not.
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

/// The function `compute_cardinality` calculates the cardinality ratio of unique values in a given
/// array of data samples.
///
/// Arguments:
///
/// * `samples`: The function `compute_cardinality` takes a reference to a dynamic array `samples` as
///   input and calculates the cardinality of the data in the array. The function uses different logic
///   based on the data type of the array elements to determine the unique count of values in the array.
///
/// Returns:
///
/// The function `compute_cardinality` returns a `Result<f32, NisabaError>`, where the `Ok` variant
/// contains the calculated cardinality as a floating-point number (`f32`).
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

/// The function `compute_avg_byte_length` calculates the average byte length of strings or binary data
/// in an array.
///
/// Arguments:
///
/// * `samples`: The `compute_avg_byte_length` function takes a reference to a trait object `samples`
///   that implements the `Array` trait. The function calculates the average byte length of the elements
///   in the array based on the data type of the array.
///
/// Returns:
///
/// The function `compute_avg_byte_length` returns a `Result` containing an `Option<f32>` or a
/// `NisabaError`.
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

/// The function `table_def_to_arrow_schema` converts a table definition into an Arrow schema.
///
/// Arguments:
///
/// * `table_def`: The `table_def` parameter is a reference to a `TableDef`, which
///   contains information about the fields and properties of a table in a database.
///
/// Returns:
///
/// An `Arc` containing the Arrow schema representation of the table definition provided as input.
fn table_def_to_arrow_schema(table_def: &TableDef) -> Arc<Schema> {
    let fields: Vec<Field> = table_def
        .fields
        .iter()
        .map(|v| Field::new(v.name.clone(), v.canonical_type.clone(), v.is_nullable))
        .collect();

    Arc::new(Schema::new(fields))
}
