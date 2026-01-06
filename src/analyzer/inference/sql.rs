use arrow::{
    array::{
        ArrayBuilder, ArrayRef, ArrowPrimitiveType, BinaryBuilder, BooleanBuilder, Date32Builder,
        Date64Builder, Decimal128Builder, FixedSizeBinaryBuilder, FixedSizeListBuilder,
        Float16Builder, Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder,
        Int64Builder, PrimitiveBuilder, RecordBatch, StringBuilder, Time32MillisecondBuilder,
        Time64MicrosecondBuilder, TimestampMicrosecondBuilder,
    },
    datatypes::{DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type, TimeUnit},
    error::ArrowError,
};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use futures::executor::block_on;
use sqlx::{MySqlPool, PgPool, Row, SqlitePool};

use crate::{
    analyzer::{
        catalog::{DataLocation, DataStoreType},
        inference::{
            SchemaInferenceEngine, SourceField, compute_field_metrics, convert_into_table_defs,
            promote::{ColumnStats, TypeLatticeResolver, cast_utf8_column},
            table_def_to_arrow_schema,
        },
    },
    error::NError,
    types::TableDef,
};

// =================================================
// SQL-Like Inference Engine
// =================================================

/// SQL inference engine for common OLTPs
#[derive(Debug)]
pub struct SQLInferenceEngine {
    profile_data: bool,
}

impl Default for SQLInferenceEngine {
    fn default() -> Self {
        Self { profile_data: true }
    }
}

impl SQLInferenceEngine {
    pub fn new() -> Self {
        SQLInferenceEngine::default()
    }

    pub fn with_profiling(mut self, profile: bool) -> Self {
        self.profile_data = profile;
        self
    }

    async fn infer_from_mysql(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let conn_str = location.connection_string()?;

        let pool = MySqlPool::connect(&conn_str).await?;

        let source_fields = read_mysql_fields(&pool, location).await?;

        let mut table_defs = convert_into_table_defs(source_fields)?;

        // Enriching
        // 1. type_confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature

        for table_def in &mut table_defs {
            let data = read_mysql_table(&pool, table_def).await?;

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

    async fn infer_from_postgres(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let conn_str = location.connection_string()?;

        let pool = PgPool::connect(&conn_str).await?;

        let source_fields = read_postgres_fields(&pool, location).await?;

        let mut table_defs = convert_into_table_defs(source_fields)?;

        // Enriching
        // 1. type_confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature

        for table_def in &mut table_defs {
            let data = read_postgres_table(&pool, table_def).await?;

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

    async fn infer_from_sqlite(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let conn_str = location.connection_string()?;

        let pool = SqlitePool::connect(&conn_str).await?;

        // Execute query and create result set
        let source_fields = read_sqlite_fields(&pool, location).await?;

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
            let data = read_sqlite_table(&pool, table_def).await?;

            if let Some(mut batch) = data {
                // Promotion
                let schema = batch.schema();

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

                            if let Some(ff) = table_def
                                .fields
                                .iter_mut()
                                .find(|f| f.name == *field.name())
                                && ff.canonical_type != resolved_result.dest_type
                            {
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
    fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        match location.store_type {
            DataStoreType::MySQL => block_on(self.infer_from_mysql(location)),
            DataStoreType::PostgreSQL => block_on(self.infer_from_postgres(location)),
            DataStoreType::SQLite => block_on(self.infer_from_sqlite(location)),
            _ => Err(NError::Unsupported(format!(
                "{:?} SQL store provided unsupported by SQL engine",
                location.store_type
            ))),
        }
    }

    fn can_handle(&self, store_type: &DataStoreType) -> bool {
        matches!(
            store_type,
            DataStoreType::MySQL | DataStoreType::PostgreSQL | DataStoreType::SQLite
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
) -> Result<Option<RecordBatch>, NError> {
    let query = "SELECT * FROM $1 LIMIT 1000";
    let rows = sqlx::query(query)
        .bind(&table_def.name)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_postgres_fields(
    pool: &PgPool,
    location: &DataLocation,
) -> Result<Vec<SourceField>, NError> {
    let silo_id = format!("{}-{}", location.store_type, location.host.clone().unwrap());

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
                                udt_name, 
                        FROM information_schema.columns 
                        WHERE table_schema = $1";

    let rows = sqlx::query(query)
        .bind(location.namespace.clone().unwrap_or("public".into()))
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
) -> Result<Option<RecordBatch>, NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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
            append_value(&mut builders[index], row, index, field)?;
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
) -> Result<Option<RecordBatch>, NError> {
    let query = "SELECT * FROM $1 LIMIT 1000";
    let rows = sqlx::query(query)
        .bind(&table_def.name)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_mysql_fields(
    pool: &MySqlPool,
    location: &DataLocation,
) -> Result<Vec<SourceField>, NError> {
    let silo_id = format!("{}-{}", location.store_type, location.host.clone().unwrap());

    let query = "SELECT 
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
                            column_type AS udt_name, 
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE();";

    let rows = sqlx::query(query).fetch_all(pool).await?;

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

// ====================
// SQLite Inference
// ====================
async fn read_sqlite_table(
    pool: &SqlitePool,
    table_def: &TableDef,
) -> Result<Option<RecordBatch>, NError> {
    let query = "SELECT * FROM $1 LIMIT 1000";
    let rows = sqlx::query(query)
        .bind(&table_def.name)
        .fetch_all(pool)
        .await?;

    build_record_batches(rows, table_def)
}

async fn read_sqlite_fields(
    pool: &SqlitePool,
    location: &DataLocation,
) -> Result<Vec<SourceField>, NError> {
    let silo_id = format!(
        "{}-{}",
        location.store_type,
        location.dir_path.clone().unwrap()
    );

    let mut source_fields = Vec::new();

    let query = "SELECT table_name FROM sqlite_master WHERE type = 'table';";

    let table_names = sqlx::query(query).fetch_all(pool).await?;

    let table_names = table_names
        .into_iter()
        .map(|r| r.get("table_name"))
        .collect::<Vec<String>>();

    for table_name in table_names {
        let query = "PRAGMA table_info($1)";

        let rows = sqlx::query(query).bind(&table_name).fetch_all(pool).await?;

        let schemas: Vec<SourceField> = rows
            .into_iter()
            .map(|r| {
                let table_schema = String::from("default");
                let column_name: String = r.get("name");
                let column_default: Option<String> = r.get("dflt_value");
                let is_nullable: bool = r.get("notnull");
                let data_type: String = r.get("type");

                SourceField {
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
                }
            })
            .collect();

        source_fields.extend(schemas);
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
        DataType::Timestamp(_, _) => Box::new(TimestampMicrosecondBuilder::with_capacity(capacity)),

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
fn append_value<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
    field: &Field,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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
        DataType::Time64(TimeUnit::Microsecond) => append_time(builder, row, index),
        DataType::Timestamp(TimeUnit::Microsecond, _) => append_timestamp(builder, row, index),

        _ => append_string(builder, row, index),
    }
}

fn append_binary<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_bool<R>(builder: &mut Box<dyn ArrayBuilder>, row: &R, index: usize) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_date<R>(builder: &mut Box<dyn ArrayBuilder>, row: &R, index: usize) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_decimal<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
    scale: i8,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_float32<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_float64<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
    f32: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    f64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
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

fn append_int<R, T>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_string<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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

fn append_time<R>(builder: &mut Box<dyn ArrayBuilder>, row: &R, index: usize) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
    NaiveTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<Time64MicrosecondBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to Time64MicrosecondBuilder".into(),
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

fn append_timestamp<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
    NaiveDateTime: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    String: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
    i64: for<'a> sqlx::Decode<'a, R::Database> + sqlx::Type<R::Database>,
{
    let b = builder
        .as_any_mut()
        .downcast_mut::<TimestampMicrosecondBuilder>()
        .ok_or(ArrowError::CastError(
            "Failed to cast to TimestampMicrosecondBuilder".into(),
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

fn append_fixed_size_binary<R>(
    builder: &mut Box<dyn ArrayBuilder>,
    row: &R,
    index: usize,
) -> Result<(), NError>
where
    R: Row,
    usize: sqlx::ColumnIndex<R>,
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
