use std::sync::Arc;

use arrow::{
    array::{
        Array, ArrayBuilder, ArrayRef, ArrowPrimitiveType, BinaryBuilder, BooleanBuilder,
        Date32Builder, Date64Builder, Decimal128Builder, FixedSizeBinaryBuilder,
        FixedSizeListBuilder, Float16Builder, Float32Builder, Float64Array, Float64Builder,
        Int8Builder, Int16Builder, Int32Builder, Int64Builder, PrimitiveBuilder, RecordBatch,
        StringBuilder, Time32MillisecondBuilder, Time64MicrosecondBuilder,
        TimestampMicrosecondBuilder, TimestampMillisecondBuilder, TimestampNanosecondBuilder,
        TimestampSecondBuilder,
    },
    datatypes::{
        DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type, Time64MicrosecondType,
        Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType,
    },
    error::ArrowError,
};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use futures::executor::block_on;
use sqlx::{MySqlPool, PgPool, Row, SqlitePool};

use crate::{
    analyzer::{
        catalog::{StorageBackend, StorageConfig},
        inference::{
            SchemaInferenceEngine, SourceField, compute_field_metrics, convert_into_table_defs,
            promote::{ColumnStats, TypeLatticeResolver, cast_utf8_column},
            table_def_to_arrow_schema,
        },
    },
    error::NisabaError,
    types::TableDef,
};

// =================================================
// SQL-Like Inference Engine
// =================================================

/// SQL inference engine for common OLTPs
#[derive(Debug)]
pub struct SQLInferenceEngine {
    sample_size: usize,
}

impl Default for SQLInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl SQLInferenceEngine {
    pub fn new() -> Self {
        SQLInferenceEngine::default()
    }

    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    async fn infer_from_mysql(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        let conn_str = config.connection_string()?;

        let pool = MySqlPool::connect(&conn_str).await?;

        let source_fields = read_mysql_fields(&pool, config).await?;

        let mut table_defs = convert_into_table_defs(source_fields)?;

        // Enriching
        // 1. type_confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature

        for table_def in &mut table_defs {
            let data = read_mysql_table(&pool, table_def, self.sample_size).await?;

            if let Some(batch) = data {
                let metrics = compute_field_metrics(&batch)?;

                let _ = table_def.fields.iter_mut().map(|f| {
                    let fmetrics = metrics.get(&f.name);
                    f.enrich_from_arrow(fmetrics);
                    f.type_confidence = Some(1.0);
                });
            }
        }

        Ok(table_defs)
    }

    async fn infer_from_postgres(
        &self,
        config: &StorageConfig,
    ) -> Result<Vec<TableDef>, NisabaError> {
        let conn_str = config.connection_string()?;

        let pool = PgPool::connect(&conn_str).await?;

        let source_fields = read_postgres_fields(&pool, config).await?;

        let mut table_defs = convert_into_table_defs(source_fields)?;

        // Enriching
        // 1. type_confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature

        for table_def in &mut table_defs {
            let data = read_postgres_table(&pool, table_def, self.sample_size).await?;

            if let Some(batch) = data {
                let metrics = compute_field_metrics(&batch)?;

                let _ = table_def.fields.iter_mut().map(|f| {
                    let fmetrics = metrics.get(&f.name);
                    f.enrich_from_arrow(fmetrics);
                    f.type_confidence = Some(1.0);
                });
            }
        }

        Ok(table_defs)
    }

    async fn infer_from_sqlite(
        &self,
        config: &StorageConfig,
    ) -> Result<Vec<TableDef>, NisabaError> {
        let conn_str = config.connection_string()?;

        let pool = SqlitePool::connect(&conn_str).await?;

        // Execute query and create result set
        let source_fields = read_sqlite_fields(&pool, config).await?;

        let mut table_defs = convert_into_table_defs(source_fields)?;

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

        for table_def in &mut table_defs {
            let data = read_sqlite_table(&pool, table_def, self.sample_size).await?;

            if let Some(mut batch) = data {
                // Promotion
                let schema = batch.schema();

                for (index, field) in schema.fields().iter().enumerate() {
                    let mut column = batch.column(index).clone();
                    // Handling Boolean/Int64 from f64
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

                    match column.data_type() {
                        DataType::LargeUtf8
                        | DataType::Utf8
                        | DataType::Int32
                        | DataType::Int64 => {
                            let stats = ColumnStats::new(&column);
                            let resolver = TypeLatticeResolver::new();
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
                                        ff.char_max_length =
                                            resolved_result.character_maximum_length;
                                    }

                                    (_, _) => {}
                                }
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

                let _ = table_def.fields.iter_mut().map(|f| {
                    let fmetrics = metrics.get(&f.name);

                    if let Some(m) = fmetrics {
                        f.char_class_signature = Some(m.char_class_signature);
                        f.is_monotonic = m.monotonicity;
                        f.cardinality = Some(m.cardinality);
                        f.avg_byte_length = m.avg_byte_length
                    }
                });
            }
        }

        Ok(table_defs)
    }
}

impl SchemaInferenceEngine for SQLInferenceEngine {
    fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        match config.backend {
            StorageBackend::MySQL => block_on(self.infer_from_mysql(config)),
            StorageBackend::PostgreSQL => block_on(self.infer_from_postgres(config)),
            StorageBackend::SQLite => block_on(self.infer_from_sqlite(config)),
            _ => Err(NisabaError::Unsupported(format!(
                "{:?} SQL store provided unsupported by SQL engine",
                config.backend
            ))),
        }
    }

    fn can_handle(&self, backend: &StorageBackend) -> bool {
        matches!(
            backend,
            StorageBackend::MySQL | StorageBackend::PostgreSQL | StorageBackend::SQLite
        )
    }

    fn engine_name(&self) -> &str {
        "SQL"
    }
}

// ====================
// Postgres Inference
// ====================

async fn read_postgres_table(
    pool: &PgPool,
    table_def: &TableDef,
    sample_size: usize,
) -> Result<Option<RecordBatch>, NisabaError> {
    let query = format!("SELECT * FROM {} LIMIT $1", table_def.name);
    let rows = sqlx::query(&query)
        .bind(sample_size as i32)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_postgres_fields(
    pool: &PgPool,
    config: &StorageConfig,
) -> Result<Vec<SourceField>, NisabaError> {
    let silo_id = format!("{}-{}", config.backend, config.host.clone().unwrap());

    let query = "SELECT table_schema, 
                                table_name, 
                                column_name,
                                column_default,
                                is_nullable, 
                                data_type, 
                                numeric_precision, 
                                numeric_scale, 
                                datetime_precision,
                                character_maximum_length,
                                udt_name 
                        FROM information_schema.columns 
                        WHERE table_schema = $1";

    let rows = sqlx::query(query)
        .bind(config.namespace.clone().unwrap_or("public".into()))
        .fetch_all(pool)
        .await?;

    let mut source_fields = Vec::new();

    for row in rows {
        let table_schema: String = row.get("table_schema");
        let table_name: String = row.get("table_name");
        let column_name: String = row.get("column_name");
        let column_default: Option<String> = row.get("column_default");
        let is_nullable: String = row.get("is_nullable");
        let data_type: String = row.get("data_type");
        let numeric_precision: Option<i32> = row.get("numeric_precision");
        let numeric_scale: Option<i32> = row.get("numeric_scale");
        let datetime_precision: Option<i32> = row.get("datetime_precision");
        let character_maximum_length: Option<i32> = row.get("character_maximum_length");
        let udt_name: String = row.get("udt_name");

        source_fields.push(SourceField {
            silo_id: silo_id.clone(),
            table_schema,
            table_name,
            column_name,
            column_default,
            is_nullable,
            data_type,
            numeric_precision,
            numeric_scale,
            datetime_precision,
            character_maximum_length,
            udt_name,
        });
    }

    Ok(source_fields)
}

fn build_record_batches<R>(
    rows: Vec<R>,
    table_def: &TableDef,
) -> Result<Option<RecordBatch>, NisabaError>
where
    R: Row,
    for<'a> &'a str: sqlx::ColumnIndex<R>,
    bool: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i8: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i16: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Encode<'a, R::Database>,
    Vec<u8>: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveDate: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveDateTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    if rows.is_empty() {
        return Ok(None);
    }
    let schema = table_def_to_arrow_schema(table_def);

    // Build array builders for each column
    let mut builders: Vec<Box<dyn ArrayBuilder>> = schema
        .fields()
        .iter()
        .map(|field| {
            let field_details = table_def
                .fields
                .iter()
                .find(|f| f.name == *field.name())
                .unwrap();

            create_array_builder(
                field.data_type(),
                rows.len(),
                field_details.char_max_length.map(|v| v as usize),
            )
        })
        .collect();

    for row in &rows {
        for (index, field) in schema.fields().iter().enumerate() {
            append_value(&mut builders[index], row, field.name().as_str(), field)?;
        }
    }

    let columns: Vec<ArrayRef> = builders.into_iter().map(|mut b| b.finish()).collect();

    let batch = RecordBatch::try_new(schema, columns)?;

    Ok(Some(batch))
}

// ====================
// MySQL Inference
// ====================
async fn read_mysql_table(
    pool: &MySqlPool,
    table_def: &TableDef,
    sample_size: usize,
) -> Result<Option<RecordBatch>, NisabaError> {
    let query = format!("SELECT * FROM {} LIMIT ?", table_def.name);
    let rows = sqlx::query(&query)
        .bind(sample_size as i32)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_mysql_fields(
    pool: &MySqlPool,
    config: &StorageConfig,
) -> Result<Vec<SourceField>, NisabaError> {
    let silo_id = format!("{}-{}", config.backend, config.host.clone().unwrap());

    let query = "SELECT 
                            table_schema AS table_schema, 
                            table_name AS table_name, 
                            column_name AS column_name, 
                            column_default AS column_default,
                            is_nullable AS is_nullable,
                            data_type AS data_type,
                            numeric_precision AS numeric_precision, 
                            numeric_scale AS numeric_scale, 
                            datetime_precision AS datetime_precision,
                            character_maximum_length AS character_maximum_length,
                            column_type AS udt_name 
                        FROM information_schema.COLUMNS
                        WHERE table_schema = DATABASE();";

    let rows = sqlx::query(query).fetch_all(pool).await?;

    let mut source_fields = Vec::new();

    for row in rows {
        let table_schema: String = row.get("table_schema");
        let table_name: String = row.get("table_name");
        let column_name: String = row.get("column_name");
        let column_default: Option<String> = row.get("column_default");
        let is_nullable: String = row.get("is_nullable");
        let data_type: Vec<u8> = row.get("data_type");
        let data_type = String::from_utf8(data_type)?;
        let numeric_precision: Option<u32> = row.get("numeric_precision");
        let numeric_scale: Option<u32> = row.get("numeric_scale");
        let datetime_precision: Option<u32> = row.get("datetime_precision");
        let character_maximum_length: Option<i64> = row.get("character_maximum_length");
        let udt_name: Vec<u8> = row.get("udt_name");
        let udt_name = String::from_utf8(udt_name)?;

        source_fields.push(SourceField {
            silo_id: silo_id.clone(),
            table_schema,
            table_name,
            column_name,
            column_default,
            is_nullable,
            data_type,
            numeric_precision: numeric_precision.map(|v| v as i32),
            numeric_scale: numeric_scale.map(|v| v as i32),
            datetime_precision: datetime_precision.map(|v| v as i32),
            character_maximum_length: character_maximum_length.map(|v| v as i32),
            udt_name,
        });
    }

    Ok(source_fields)
}

// ====================
// SQLite Inference
// ====================
async fn read_sqlite_table(
    pool: &SqlitePool,
    table_def: &TableDef,
    sample_size: usize,
) -> Result<Option<RecordBatch>, NisabaError> {
    let query = format!("SELECT * FROM {} LIMIT $1", table_def.name);
    let rows = sqlx::query(&query)
        .bind(sample_size as i32)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_sqlite_fields(
    pool: &SqlitePool,
    config: &StorageConfig,
) -> Result<Vec<SourceField>, NisabaError> {
    let silo_id = format!("{}-{}", config.backend, config.dir_path.clone().unwrap());

    let mut source_fields = Vec::new();

    let query = "SELECT tbl_name FROM sqlite_master WHERE type = 'table';";

    let table_names = sqlx::query(query).fetch_all(pool).await?;

    let table_names = table_names
        .into_iter()
        .map(|r| r.get("tbl_name"))
        .collect::<Vec<String>>();

    for table_name in table_names {
        let query = format!("PRAGMA table_info({})", table_name);

        let rows = sqlx::query(&query).fetch_all(pool).await?;

        let schemas: Result<Vec<SourceField>, std::string::FromUtf8Error> = rows
            .into_iter()
            .map(|r| {
                let table_schema = String::from("default");

                let column_name: String = r.get("name");

                let column_default: Option<String> = r.get("dflt_value");

                let is_nullable: bool = r.get("notnull");

                let data_type: String = r.get("type");

                Ok(SourceField {
                    silo_id: silo_id.clone(),
                    table_schema,
                    table_name: table_name.clone(),
                    column_name,
                    column_default,
                    is_nullable: if is_nullable {
                        String::from("NO")
                    } else {
                        String::from("YES")
                    },
                    data_type: data_type.to_lowercase(),
                    numeric_precision: None,
                    numeric_scale: None,
                    datetime_precision: None,
                    character_maximum_length: None,
                    udt_name: data_type.to_lowercase(),
                })
            })
            .collect();

        source_fields.extend(schemas?);
    }

    Ok(source_fields)
}

fn create_array_builder(
    data_type: &DataType,
    capacity: usize,
    byte_size: Option<usize>,
) -> Box<dyn ArrayBuilder> {
    match data_type {
        DataType::Binary => {
            let data_capacity = byte_size
                .map(|size| capacity * size)
                .unwrap_or(capacity * 256);

            Box::new(BinaryBuilder::with_capacity(capacity, data_capacity))
        }
        DataType::Boolean => Box::new(BooleanBuilder::with_capacity(capacity)),
        DataType::Date32 => Box::new(Date32Builder::with_capacity(capacity)),
        DataType::Date64 => Box::new(Date64Builder::with_capacity(capacity)),
        DataType::Decimal128(_, _) => Box::new(Decimal128Builder::with_capacity(capacity)),
        DataType::FixedSizeBinary(p) => {
            Box::new(FixedSizeBinaryBuilder::with_capacity(capacity, *p))
        }
        DataType::FixedSizeList(a, b) => {
            let values_builder = create_array_builder(a.data_type(), (*b) as usize, None);

            Box::new(FixedSizeListBuilder::with_capacity(
                values_builder,
                *b,
                capacity,
            ))
        }

        DataType::Float16 => Box::new(Float16Builder::with_capacity(capacity)),
        DataType::Float32 => Box::new(Float32Builder::with_capacity(capacity)),
        DataType::Float64 => Box::new(Float64Builder::with_capacity(capacity)),

        DataType::Int8 => Box::new(Int8Builder::with_capacity(capacity)),
        DataType::Int16 => Box::new(Int16Builder::with_capacity(capacity)),
        DataType::Int32 => Box::new(Int32Builder::with_capacity(capacity)),
        DataType::Int64 => Box::new(Int64Builder::with_capacity(capacity)),

        DataType::Time32(_) => Box::new(Time32MillisecondBuilder::with_capacity(capacity)),
        DataType::Time64(_) => Box::new(Time64MicrosecondBuilder::with_capacity(capacity)),
        DataType::Timestamp(TimeUnit::Second, _) => {
            Box::new(TimestampSecondBuilder::with_capacity(capacity))
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            Box::new(TimestampMillisecondBuilder::with_capacity(capacity))
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            Box::new(TimestampMicrosecondBuilder::with_capacity(capacity))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            Box::new(TimestampNanosecondBuilder::with_capacity(capacity))
        }

        DataType::Utf8 => {
            let data_capacity = byte_size
                .map(|size| capacity * size)
                .unwrap_or(capacity * 256);

            Box::new(StringBuilder::with_capacity(capacity, data_capacity))
        }

        _ => {
            let data_capacity = byte_size
                .map(|size| capacity * size)
                .unwrap_or(capacity * 256);

            Box::new(StringBuilder::with_capacity(capacity, data_capacity))
        }
    }
}

// =======================
// Value Appender
// =======================
fn append_value<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
    field: &Field,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    bool: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i8: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i16: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    Vec<u8>: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveDate: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    NaiveDateTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    match field.data_type() {
        DataType::Binary => append_binary(builder, row, index),
        DataType::Boolean => append_bool(builder, row, index),
        DataType::Date32 => append_date(builder, row, index),
        DataType::Decimal128(_, scale) => append_decimal(builder, row, index, *scale),
        DataType::FixedSizeBinary(_) => append_fixed_size_binary(builder, row, index),
        DataType::Int8 => append_int::<_, Int8Type>(builder, row, index),
        DataType::Int16 => append_int::<_, Int16Type>(builder, row, index),
        DataType::Int32 => append_int::<_, Int32Type>(builder, row, index),
        DataType::Int64 => append_int::<_, Int64Type>(builder, row, index),
        DataType::Float32 => append_float32(builder, row, index),
        DataType::Float64 => append_float64(builder, row, index),
        DataType::Time64(TimeUnit::Microsecond) => {
            append_time::<_, Time64MicrosecondType>(builder, row, index)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            append_time::<_, Time64NanosecondType>(builder, row, index)
        }

        DataType::Timestamp(TimeUnit::Second, _) => {
            append_timestamp::<_, TimestampSecondType>(builder, row, index)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            append_timestamp::<_, TimestampMillisecondType>(builder, row, index)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            append_timestamp::<_, TimestampMicrosecondType>(builder, row, index)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            append_timestamp::<_, TimestampNanosecondType>(builder, row, index)
        }

        _ => append_string(builder, row, index),
    }
}

fn append_binary<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    Vec<u8>: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<BinaryBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to mutable BinaryBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<Vec<u8>, _>(index) {
        b.append_value(val);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_bool<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    bool: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<BooleanBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to mutable BooleanBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<bool, _>(index) {
        b.append_value(val);
    } else if let Ok(val) = row.try_get::<i64, _>(index) {
        b.append_value(val != 0);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_date<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    NaiveDate: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<Date32Builder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to Date32Builder".into(),
        ))?;

    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();

    if let Ok(val) = row.try_get::<NaiveDate, _>(index) {
        b.append_value((val - epoch).num_days() as i32);
    } else if let Ok(val) = row.try_get::<String, _>(index) {
        let formats = [
            "%d%b%Y%p%I%M%S%.f",
            "%y/%m/%d %H",
            "%y/%m/%d %H:%M",
            // ISO-8601 (T separator)
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y-%m-%dT%H:%M:%S%:z",
            "%Y-%m-%dT%H:%M:%S%.f%:z",
            "%Y-%m-%dT%H:%M:%S%.f",
            // Space-separated
            "%Y-%m-%d %H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S%:z",
            "%Y-%m-%d %H:%M:%S%.f%:z",
            // RFC / logs
            "%a, %d %b %Y %H:%M:%S GMT",
            "%d/%b/%Y:%H:%M:%S %z",
            // ISO / SQL standard
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%y/%m/%d %H:%M:%S",
            "%Y%m%d%H%M%S",
            "%a, %d %b %Y",
        ];
        let parsed = formats
            .iter()
            .find_map(|fmt| NaiveDate::parse_from_str(&val, fmt).ok());
        if let Some(val) = parsed {
            b.append_value((val - epoch).num_days() as i32);
        } else {
            b.append_null();
        }
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_decimal<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
    scale: i8,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<Decimal128Builder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to Decimal128Builder".into(),
        ))?;
    let scale_factor = 10_i128.pow(scale as u32);

    if let Ok(val) = row.try_get::<f64, _>(index) {
        b.append_value((val * scale_factor as f64) as i128);
    } else if let Ok(val) = row.try_get::<String, _>(index) {
        if let Ok(val) = val.parse::<f64>() {
            b.append_value((val * scale_factor as f64) as i128);
        } else {
            b.append_null();
        }
    } else {
        b.append_null();
    }

    Ok(())
}

fn append_float32<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    f32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<Float32Builder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to PrimitiveBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<f32, _>(index) {
        b.append_value(val);
    } else if let Ok(val) = row.try_get::<f64, _>(index) {
        b.append_value(val as f32);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_float64<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    f32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<Float64Builder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to PrimitiveBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<f64, _>(index) {
        b.append_value(val);
    } else if let Ok(val) = row.try_get::<f32, _>(index) {
        b.append_value(val as f64);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_int<'r, R, T>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    T: ArrowPrimitiveType,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    T::Native: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database> + TryFrom<i64>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to PrimitiveBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<T::Native, _>(index) {
        b.append_value(val);
    } else if let Ok(val) = row.try_get::<i64, _>(index) {
        if let Ok(converted) = T::Native::try_from(val) {
            b.append_value(converted);
        } else {
            b.append_null();
        }
    } else {
        b.append_null();
    }

    Ok(())
}

fn append_string<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<StringBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to StringBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<String, _>(index) {
        b.append_value(val);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_time<'r, R, T>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    T: ArrowPrimitiveType<Native = i64>,
    &'r str: sqlx::ColumnIndex<R>,
    NaiveTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to PrimitiveBuilder".into(),
        ))?;

    if let Ok(time) = row.try_get::<NaiveTime, _>(index) {
        let micros =
            time.num_seconds_from_midnight() as i64 * 1_000_000 + time.nanosecond() as i64 / 1_0000;

        b.append_value(micros);
    } else {
        b.append_null();
    }

    Ok(())
}

fn append_timestamp<'r, R, T>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    T: ArrowPrimitiveType<Native = i64>,
    &'r str: sqlx::ColumnIndex<R>,
    NaiveDateTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<PrimitiveBuilder<T>>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to PrimitiveBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<NaiveDateTime, _>(index) {
        b.append_value(val.and_utc().timestamp_micros());
    } else if let Ok(val) = row.try_get::<String, _>(index) {
        let formats = [
            "%d%b%Y%p%I%M%S%.f",
            "%y/%m/%d %H",
            "%y/%m/%d %H:%M",
            // ISO-8601 (T separator)
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%.fZ",
            "%Y-%m-%dT%H:%M:%S%:z",
            "%Y-%m-%dT%H:%M:%S%.f%:z",
            "%Y-%m-%dT%H:%M:%S%.f",
            // Space-separated
            "%Y-%m-%d %H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S%:z",
            "%Y-%m-%d %H:%M:%S%.f%:z",
            // RFC / logs
            "%a, %d %b %Y %H:%M:%S GMT",
            "%d/%b/%Y:%H:%M:%S %z",
            // ISO / SQL standard
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S%.f",
            "%y/%m/%d %H:%M:%S",
            "%Y%m%d%H%M%S",
        ];

        let parsed = formats
            .iter()
            .find_map(|fmt| NaiveDateTime::parse_from_str(&val, fmt).ok());

        if let Some(ts) = parsed {
            b.append_value(ts.and_utc().timestamp_micros());
        } else {
            b.append_null();
        }
    } else if let Ok(val) = row.try_get::<i64, _>(index) {
        b.append_value(val * 1_000_000);
    } else {
        b.append_null();
    }
    Ok(())
}

fn append_fixed_size_binary<'r, R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: &'r str,
) -> Result<(), NisabaError>
where
    R: Row,
    &'r str: sqlx::ColumnIndex<R>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    Vec<u8>: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<FixedSizeBinaryBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to FixedSizeBinaryBuilder".into(),
        ))?;

    if let Ok(val) = row.try_get::<Vec<u8>, _>(index) {
        let _ = b.append_value(val);
    } else if let Ok(val) = row.try_get::<String, _>(index) {
        let _ = b.append_value(val.as_bytes());
    } else {
        b.append_null();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mysql_inference() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::MySQL,
            "localhost",
            3306,
            "mysql_store",
            "mysql",
            "mysql",
            None::<String>,
        )
        .unwrap();

        let sql_inference = SQLInferenceEngine::default();

        let result = block_on(sql_inference.infer_from_mysql(&config)).unwrap();

        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_postgresql_inference() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::PostgreSQL,
            "localhost",
            5432,
            "postgres",
            "postgres",
            "postgres",
            Some("public"),
        )
        .unwrap();

        let sql_inference = SQLInferenceEngine::default();

        let result = block_on(sql_inference.infer_from_postgres(&config)).unwrap();

        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_sqlite_inference() {
        let config = StorageConfig::new_file_backend(
            StorageBackend::SQLite,
            "./assets/sqlite/nisaba.sqlite",
        )
        .unwrap();

        let sql_inference = SQLInferenceEngine::default();

        let result = block_on(sql_inference.infer_from_sqlite(&config)).unwrap();

        assert_eq!(result.len(), 2);
    }
}
