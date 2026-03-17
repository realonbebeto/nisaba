//! Necessary Inference implementations for common stores
//!
//!
//! Overview
//! - [`CastSafety`]: A rule based type to ensure type promotion is within allowable constraints.
//! - [`CsvInferenceEngine`]: A type that is responsible for CSV store inference.
//! - [`ExcelInferenceEngine`]: A type that is responsible for  Excel store inference.
//! - [`ParquetInferenceEngine`]: A type that is responsible for Parquet store inference.
//! - [`NoSQLInferenceEngine`]: A type that is repsonsible for MongoDB store inference.
//! - [`MySqlInferenceEngine`]: A type responsible for MySQL store inference.
//! - [`PostgreSQLInferenceEngine`]: A type responsible PostgreSQL store inference.
//! - [`SqliteInferenceEngine`]: A type responsible for Sqlite store inference.
//!

use arrow::{
    array::{Array, RecordBatch},
    datatypes::{DataType, Field, Schema, TimeUnit},
};
use sqlx::prelude::FromRow;
use std::{collections::HashMap, str::FromStr, sync::Arc};
use uuid::Uuid;

mod file;
mod nosql;
mod sql;

pub use file::{CsvInferenceEngine, ExcelInferenceEngine, ParquetInferenceEngine};
pub use nosql::NoSQLInferenceEngine;
pub use sql::{MySQLInferenceEngine, PostgreSQLInferenceEngine, SqliteInferenceEngine};

use crate::{
    error::NisabaError,
    reconciler::{
        metrics::FieldStats,
        promote::{TypeLatticeResolver, cast_column},
    },
    types::{FieldDef, FieldProfile, TableDef},
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
        // 10. global null detection

        let schema = batch.schema();

        let resolver = TypeLatticeResolver::new();

        for (index, field) in schema.fields().iter().enumerate() {
            let field_name = field.name();

            // Re-fetch from batch so promotion sees any mutations above
            let column = batch.column(index).clone();

            // Calculate metrics for all fields
            let mut stats = FieldStats::calculate(&column, &table_def.name, field_name)?;

            let Some(ff) = table_def
                .fields
                .iter_mut()
                .find(|f| f.field_def.name == *field.name())
            else {
                continue;
            };

            ff.field_stats = Some(stats.clone());
            ff.field_def.is_nullable = stats.is_null;

            if matches!(
                column.data_type(),
                DataType::LargeUtf8
                    | DataType::Utf8
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Timestamp(_, _)
                    | DataType::Float64
            ) {
                let resolved_result = resolver.promote(&stats)?;

                // Update char_max_length
                match (&ff.field_def.canonical_type, &resolved_result.dest_type) {
                    (DataType::Utf8, DataType::Utf8)
                    | (DataType::LargeUtf8, DataType::LargeUtf8)
                    | (DataType::Utf8View, DataType::Utf8View) => {
                        ff.field_def.char_max_length = resolved_result.character_maximum_length;
                    }

                    (_, _) => {}
                }

                // Update type related signals when there is a mismatch on types
                if ff.field_def.canonical_type != resolved_result.dest_type {
                    ff.field_def.canonical_type = resolved_result.dest_type;
                    ff.field_def.type_confidence = Some(resolved_result.confidence);
                    ff.field_def.is_nullable = resolved_result.nullable;
                    ff.field_def.char_max_length = resolved_result.character_maximum_length;
                    ff.field_def.numeric_precision = resolved_result.numeric_precision;
                    ff.field_def.numeric_scale = resolved_result.numeric_scale;
                    ff.field_def.datetime_precision = resolved_result.datetime_precision;

                    // Update values in batch to match new type
                    cast_column(batch, &ff.field_def.name, &ff.field_def.canonical_type)?;

                    // Re-fetch again so stats calculations see type-promotion mutations
                    let column = batch.column(index).clone();

                    // Recalculate and update FieldStats
                    let ustats = stats.re_calculate(&column)?;

                    ff.field_stats = Some(ustats);
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

#[derive(Debug, sqlx::FromRow)]
/// The `MySQLField` represents a field in a MySQL inference_schema.columns with various properties such as
/// column name, data type, and constraints.
pub struct MySQLField {
    /// schema or namespace to which the table belongs in a database.
    pub table_schema: Vec<u8>,
    /// name of the table to which the field belongs
    pub table_name: Vec<u8>,
    /// name of a column in a database table
    pub column_name: Vec<u8>,
    /// default value of the column
    pub column_default: Option<Vec<u8>>,
    /// flag if the column is nullable or not
    pub is_nullable: Vec<u8>,
    /// data type of the column
    pub data_type: Vec<u8>,
    /// precision of a numeric data type
    pub numeric_precision: Option<u32>,
    /// scale of a numeric data type
    pub numeric_scale: Option<u32>,
    /// precision of a datetime data type
    pub datetime_precision: Option<u32>,
    /// maximum length of character of the field
    pub character_maximum_length: Option<i64>,
    /// user-defined data type
    pub udt_name: Vec<u8>,
}

impl MySQLField {
    pub fn with_silo_id(self, silo_id: &str) -> Result<SourceField, NisabaError> {
        let column_default = match self.column_default {
            Some(v) => Some(String::from_utf8(v)?),
            None => None,
        };
        Ok(SourceField {
            silo_id: silo_id.to_owned(),
            table_schema: String::from_utf8(self.table_schema)?,
            table_name: String::from_utf8(self.table_name)?,
            column_name: String::from_utf8(self.column_name)?,
            column_default,
            is_nullable: String::from_utf8(self.is_nullable)?,
            data_type: String::from_utf8(self.data_type)?,
            numeric_precision: self.numeric_precision.map(|v| v as i32),
            numeric_scale: self.numeric_scale.map(|v| v as i32),
            datetime_precision: self.datetime_precision.map(|v| v as i32),
            character_maximum_length: self.character_maximum_length.map(|v| v as i32),
            udt_name: String::from_utf8(self.udt_name)?,
        })
    }
}

#[derive(Debug, sqlx::FromRow)]
/// The `RawField` represents a field in a data source with various properties such as
/// column name, data type, and constraints.
pub struct PostgresField {
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

impl PostgresField {
    pub fn with_silo_id(self, silo_id: &str) -> SourceField {
        SourceField {
            silo_id: silo_id.to_owned(),
            table_schema: self.table_schema,
            table_name: self.table_name,
            column_name: self.column_name,
            column_default: self.column_default,
            is_nullable: self.is_nullable,
            data_type: self.data_type,
            numeric_precision: self.numeric_precision,
            numeric_scale: self.numeric_scale,
            datetime_precision: self.datetime_precision,
            character_maximum_length: self.character_maximum_length,
            udt_name: self.udt_name,
        }
    }
}

pub fn to_source_fields(silo_id: &str, rows: Vec<PostgresField>) -> Vec<SourceField> {
    rows.into_iter()
        .map(|rf| rf.with_silo_id(silo_id))
        .collect()
}

#[derive(Debug, FromRow)]
pub struct PragmaField {
    name: String,
    #[sqlx(rename = "type")]
    data_type: String,
    notnull: bool,
    dflt_value: Option<String>,
}

impl PragmaField {
    pub fn into_source_field(self, silo_id: &str, table_name: String) -> SourceField {
        SourceField {
            silo_id: silo_id.to_string(),
            table_schema: String::from("default"),
            table_name,
            column_name: self.name,
            column_default: self.dflt_value,
            is_nullable: if self.notnull {
                String::from("YES")
            } else {
                String::from("NO")
            },
            data_type: self.data_type.to_lowercase(),
            numeric_precision: None,
            numeric_scale: None,
            datetime_precision: None,
            character_maximum_length: None,
            udt_name: self.data_type.to_lowercase(),
        }
    }
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
pub fn convert_into_table_defs(
    source_fields: Vec<SourceField>,
) -> Result<Vec<TableDef>, NisabaError> {
    if source_fields.is_empty() {
        return Ok(Vec::new());
    }

    let mut map: HashMap<String, TableDef> = HashMap::new();

    for src_field in source_fields {
        let is_nullable = src_field.is_nullable.trim().eq_ignore_ascii_case("yes");
        let canonical_type = sql_to_arrow_type(
            &src_field.udt_name,
            src_field.numeric_precision,
            src_field.numeric_scale,
            src_field.datetime_precision,
        )?;

        let table_def = map
            .entry(src_field.table_name.clone())
            .or_insert_with(|| TableDef {
                id: Uuid::now_v7(),
                silo_id: src_field.silo_id,
                table_schema: src_field.table_schema,
                name: src_field.table_name.clone(),
                fields: Vec::new(),
            });

        let table_id = table_def.id;

        table_def.fields.push(FieldProfile {
            field_def: FieldDef {
                id: Uuid::now_v7(),
                table_id,
                name: src_field.column_name,
                canonical_type,
                type_confidence: Some(0.98),
                column_default: src_field.column_default,
                is_nullable,
                char_max_length: src_field.character_maximum_length,
                numeric_precision: src_field.numeric_precision,
                numeric_scale: src_field.numeric_scale,
                datetime_precision: src_field.datetime_precision,
            },
            field_stats: None,
        });

        // TODO: nested and max-depth will have to be updated later
    }

    Ok(map.into_values().collect())
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
        "date64" => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
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
        "datetime" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
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
        "utf8view" => Ok(DataType::Utf8View),
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
        .map(|v| {
            Field::new(
                v.field_def.name.clone(),
                v.field_def.canonical_type.clone(),
                v.field_def.is_nullable,
            )
        })
        .collect();

    Arc::new(Schema::new(fields))
}
