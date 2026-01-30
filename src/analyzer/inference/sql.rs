use arrow::{
    array::{
        ArrayBuilder, ArrayRef, ArrowPrimitiveType, BinaryBuilder, BooleanBuilder, Date32Builder,
        Date64Builder, Decimal128Builder, FixedSizeBinaryBuilder, FixedSizeListBuilder,
        Float16Builder, Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder,
        Int64Builder, PrimitiveBuilder, RecordBatch, StringBuilder, Time32MillisecondBuilder,
        Time64MicrosecondBuilder, TimestampMicrosecondBuilder, TimestampMillisecondBuilder,
        TimestampNanosecondBuilder, TimestampSecondBuilder,
    },
    datatypes::{
        DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type, Time64MicrosecondType,
        Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType,
    },
    error::ArrowError,
};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use futures::{StreamExt, stream};
use sqlx::{MySqlPool, PgPool, Row, SqlitePool};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    analyzer::{
        datastore::{Extra, Source},
        inference::{
            SchemaInferenceEngine, SourceField, convert_into_table_defs, table_def_to_arrow_schema,
        },
        probe::InferenceStats,
    },
    error::NisabaError,
    types::{TableDef, TableRep},
};
// =================================================
// SQL-Like Inference Engine
// =================================================
/// SQL inference engines for common OLTPs

// =================================================
// MySQL Inference Engine
// =================================================
#[derive(Debug)]
pub struct MySQLInferenceEngine {
    sample_size: usize,
}

impl Default for MySQLInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl MySQLInferenceEngine {
    pub fn new(sample_size: Option<usize>) -> Self {
        MySQLInferenceEngine {
            sample_size: sample_size.unwrap_or(1000),
        }
    }

    /// The function `mysql_store_infer` asynchronously reads table fields from a MySQL database,
    /// converts them into table definitions, enriches the definitions with various metrics, and returns
    /// the resulting table definitions.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter in the `mysql_store_infer` function is of type
    ///   `&StorageConfig`, which is a reference to a `StorageConfig` struct. This parameter is used to
    ///   provide configuration details for connecting to a MySQL database, such as the connection string.
    ///
    /// Returns:
    ///
    /// The function `mysql_store_infer` returns a `Result` containing a vector of `TableDef` structs or
    /// a `NisabaError` if an error occurs during the process.
    pub async fn mysql_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        on_table: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let pool = source
            .client
            .as_mysql()
            .ok_or(NisabaError::Missing("No MySQL pool provided".into()))?;

        let source_fields = read_mysql_fields(pool, &source.metadata.silo_id).await?;

        let table_defs = convert_into_table_defs(source_fields)?;

        let total_tables_p = table_defs.len();

        // Enriching
        // 1. type_confidence
        // 2. cardinality
        // 3. avg_byte_length
        // 4. is_monotonic
        // 5. char_class_signature

        let table_results = stream::iter(table_defs)
            .map(|mut table_def| {
                let shared_pool = pool.clone();

                async move {
                    let data = read_mysql_table(&shared_pool, &table_def, self.sample_size).await?;

                    if let Some(mut batch) = data {
                        self.enrich_table_def(&mut table_def, &mut batch)?;
                    }

                    let table_rep: TableRep = (&table_def).into();

                    Ok::<_, NisabaError>((table_rep, table_def))
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<_>>()
            .await;

        let mut errors = Vec::new();
        let mut table_defs = Vec::new();
        let mut table_reps = Vec::new();

        for result in table_results {
            match result {
                Ok((rep, def)) => {
                    table_reps.push(rep);
                    table_defs.push(def);
                }
                Err(e) => errors.push(e.to_string()),
            }
        }

        {
            let mut stats = infer_stats.lock().await;
            stats.errors.extend(errors);

            stats.tables_inferred += table_reps.len();

            stats.tables_found += total_tables_p;

            stats.fields_inferred += table_reps.iter().map(|t| t.fields.len()).sum::<usize>();
        }

        on_table(table_defs).await?;

        Ok(table_reps)
    }
}

impl SchemaInferenceEngine for MySQLInferenceEngine {
    fn engine_name(&self) -> &str {
        "mysql"
    }
}

// =================================================
// PostgreSQL Inference Engine
// =================================================
#[derive(Debug)]
pub struct PostgreSQLInferenceEngine {
    sample_size: usize,
}

impl Default for PostgreSQLInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl PostgreSQLInferenceEngine {
    pub fn new(sample_size: Option<usize>) -> Self {
        PostgreSQLInferenceEngine {
            sample_size: sample_size.unwrap_or(1000),
        }
    }
    /// The function `postgres_store_infer` reads table definitions from a PostgreSQL database, enriches
    /// them with additional metrics, and returns the resulting table definitions.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter is of type `&StorageConfig` and is used to
    ///   provide configuration details for connecting to a PostgreSQL database.
    ///
    /// Returns:
    ///
    /// The `postgres_store_infer` function returns a `Result` containing a vector of `TableRep` structs
    /// or a `NisabaError` if an error occurs during the process.
    pub async fn postgres_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        on_table: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let pool = source
            .client
            .as_postgres()
            .ok_or(NisabaError::Missing("No PgPool provided".into()))?;

        let source_fields = read_postgres_fields(pool, source).await?;

        let table_defs = convert_into_table_defs(source_fields)?;

        let total_tables_p = table_defs.len();

        let table_results = stream::iter(table_defs)
            .map(|mut table_def| {
                let shared_pool = pool.clone();

                async move {
                    let data =
                        read_postgres_table(&shared_pool, &table_def, self.sample_size).await?;

                    if let Some(mut batch) = data {
                        self.enrich_table_def(&mut table_def, &mut batch)?;
                    }

                    let table_rep: TableRep = (&table_def).into();

                    Ok::<_, NisabaError>((table_rep, table_def))
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<_>>()
            .await;

        let mut errors = Vec::new();
        let mut table_defs = Vec::new();
        let mut table_reps = Vec::new();

        for result in table_results {
            match result {
                Ok((rep, def)) => {
                    table_reps.push(rep);
                    table_defs.push(def);
                }
                Err(e) => errors.push(e.to_string()),
            }
        }

        {
            let mut stats = infer_stats.lock().await;
            stats.errors.extend(errors);

            stats.tables_inferred += table_reps.len();

            stats.tables_found += total_tables_p;

            stats.fields_inferred += table_reps.iter().map(|t| t.fields.len()).sum::<usize>();
        }

        on_table(table_defs).await?;

        Ok(table_reps)
    }
}

impl SchemaInferenceEngine for PostgreSQLInferenceEngine {
    fn engine_name(&self) -> &str {
        "postgresql"
    }
}

// =================================================
// PostgreSQL Inference Engine
// =================================================
#[derive(Debug)]
pub struct SqliteInferenceEngine {
    sample_size: usize,
}

impl Default for SqliteInferenceEngine {
    fn default() -> Self {
        Self { sample_size: 1000 }
    }
}

impl SqliteInferenceEngine {
    pub fn new(sample_size: Option<usize>) -> Self {
        Self {
            sample_size: sample_size.unwrap_or(1000),
        }
    }
    /// The function `infer_from_sqlite` asynchronously reads data from a SQLite database,
    /// processes and enriches the data by inferring types and other metrics for each table, and returns
    /// a vector of `TableDef` structs.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter  is of type `&StorageConfig`and used to
    ///   provide configuration settings for a SQLite instance.
    ///
    /// Returns:
    ///
    /// The function `infer_from_sqlite` returns a `Result` containing a vector of `TableDef` structs or
    /// a `NisabaError` if an error occurs during the execution.
    pub async fn sqlite_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        on_table: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let pool = source
            .client
            .as_sqlite()
            .ok_or(NisabaError::Missing("No SqlitePool provided".into()))?;

        // Execute query and create result set
        let source_fields = read_sqlite_fields(pool, &source.metadata.silo_id).await?;

        let table_defs = convert_into_table_defs(source_fields)?;

        let total_tables_p = table_defs.len();

        let table_results = stream::iter(table_defs)
            .map(|mut table_def| {
                let shared_pool = pool.clone();

                async move {
                    let data =
                        read_sqlite_table(&shared_pool, &table_def, self.sample_size).await?;

                    if let Some(mut batch) = data {
                        self.enrich_table_def(&mut table_def, &mut batch)?;
                    }

                    let table_rep: TableRep = (&table_def).into();

                    Ok::<_, NisabaError>((table_rep, table_def))
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<_>>()
            .await;

        let mut errors = Vec::new();
        let mut table_defs = Vec::new();
        let mut table_reps = Vec::new();

        for result in table_results {
            match result {
                Ok((rep, def)) => {
                    table_reps.push(rep);
                    table_defs.push(def);
                }
                Err(e) => errors.push(e.to_string()),
            }
        }

        {
            let mut stats = infer_stats.lock().await;
            stats.errors.extend(errors);

            stats.tables_inferred += table_reps.len();

            stats.tables_found += total_tables_p;

            stats.fields_inferred += table_reps.iter().map(|t| t.fields.len()).sum::<usize>();
        }

        on_table(table_defs).await?;

        Ok(table_reps)
    }
}

impl SchemaInferenceEngine for SqliteInferenceEngine {
    fn engine_name(&self) -> &str {
        "sqlite"
    }
}

// ====================
// Postgres Inference
// ====================

/// The `read_postgres_table` function infers the schema of a database table based on the specified
/// storage configuration.
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `PgPool`, providing pooled connections to Postgres.
/// * `table_def`: The `table_def` parameter is a reference of `TableDef`, providing table information.
/// * `sample_size`: The `sample_size` parameter is of type usize, providing size of which to use for
///   inference.
///
/// Returns:
///
/// The `read_postgres_table` function returns a `Result` containing an `Option` of `RecordBatch` or a
/// `NisabaError` if an error occurs during record batch creation.
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

/// The `read_postgres_fields` function reads the fields from the information_schema.columns view
/// of the database connection specified
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `PgPool`, providing pooled connections to Postgres.
/// * `config`: The `config` parameter is a reference of `StorageConfig`, providing database connection details.
///
/// Returns:
///
/// The `read_postgres_fields` function returns a `Result` containing a `Vec` of `SourceField` or a
/// `NisabaError` if an error occurs during reading the source view.
async fn read_postgres_fields(
    pool: &PgPool,
    source: &Source,
) -> Result<Vec<SourceField>, NisabaError> {
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
        .bind(
            source
                .metadata
                .extra
                .get(&Extra::PostgresSchema)
                .unwrap_or(&"public".to_string()),
        )
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
            silo_id: source.metadata.silo_id.clone(),
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

/// The `build_record_batches` function converts row values to columnar representation RecordBatch
///
/// Arguments:
///
/// * `rows`: The `rows` parameter is a `Vec` of objects implementing sqlx Row trait.
/// * `table_def`: The `table_def` parameter is a reference of `TableDef`, providing table information
///   for schema generation.
///
/// Returns:
///
/// The `build_record_batches` function returns a `Result` containing an `Option` of `recordBatch` or a
/// `NisabaError` if an error occurs during RecordBatch creation.
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
/// The `read_mysql_table` function infers the schema of a database table based on the specified
/// storage configuration.
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `MySqlPool`, providing pooled connections to MySQL.
/// * `table_def`: The `table_def` parameter is a reference of `TableDef`, providing table information.
/// * `sample_size`: The `sample_size` parameter is of type usize, providing size of which to use for
///   inference.
///
/// Returns:
///
/// The `read_mysql_table` function returns a `Result` containing an `Option` of `RecordBatch` or a
/// `NisabaError` if an error occurs during record batch creation.
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

/// The `read_mysql_fields` function reads the fields from the information_schema.columns view
/// of the database connection specified
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `MySqlPool`, providing pooled connections to MySQL.
/// * `config`: The `config` parameter is a reference of `StorageConfig`, providing database connection details.
///
/// Returns:
///
/// The `read_mysql_fields` function returns a `Result` containing a `Vec` of `SourceField` or a
/// `NisabaError` if an error occurs during reading the source view.
async fn read_mysql_fields(
    pool: &MySqlPool,
    silo_id: &str,
) -> Result<Vec<SourceField>, NisabaError> {
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
            silo_id: silo_id.to_string(),
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
/// The `read_sqlite_table` function infers the schema of a database table based on the specified
/// storage configuration.
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `SqlitePool`, providing pooled connections to Sqlite.
/// * `table_def`: The `table_def` parameter is a reference of `TableDef`, providing table information.
/// * `sample_size`: The `sample_size` parameter is of type usize, providing size of which to use for
///   inference.
///
/// Returns:
///
/// The `read_sqlite_table` function returns a `Result` containing an `Option` of `RecordBatch` or a
/// `NisabaError` if an error occurs during record batch creation.
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

/// The `read_sqlite_fields` function reads the fields from sqlite_master view and PRAGMA table commands
/// of the database connection specified
///
/// Arguments:
///
/// * `pool`: The `pool` parameter is a reference of `SqlitePool`, providing pooled connections to Sqlite.
/// * `config`: The `config` parameter is a reference of `StorageConfig`, providing database connection details.
///
/// Returns:
///
/// The `read_sqlite_fields` function returns a `Result` containing a `Vec` of `SourceField` or a
/// `NisabaError` if an error occurs during reading the source view.
async fn read_sqlite_fields(
    pool: &SqlitePool,
    silo_id: &str,
) -> Result<Vec<SourceField>, NisabaError> {
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
                    silo_id: silo_id.to_string(),
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

/// The `create_array_builder` function creates an array builder depending on the datatype
///
/// Arguments:
///
/// * `data_type`: The `data_type` parameter is a reference of `DataType`, to determine what type of
///   array builder to create.
/// * `capacity`: The `capacity` parameter is of type `usize`, providing a limit of memmory to reserve.
/// * `byte_size`: The `byte_size` parameter is of type `Option usize`, providing limit for fixed width data
///   like binary or utf8 to assist in memory reservation.
///
/// Returns:
///
/// The `create_array_builder` function returns a `Box` containing an object implementing `ArrayBuilder`.
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
/// The `append_value` function determines which builder to append value to
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
/// * `field`: The `field` parameter is reference to `Field` providing the DataType to determine which builder
///   to which value will be appended to.
///
/// Returns:
///
/// The `append_value` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a value.
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

/// The `append_binary` function appends binary to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_binary` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a binary.
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

/// The `append_bool` function appends bool to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_bool` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a bool.
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

/// The `append_date` function appends date to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_date` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a date.
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

/// The `append_decimal` function appends decimal to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_decimal` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a decimal.
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

/// The `append_float32` function appends float32 to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_float32` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a float32.
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

/// The `append_float64` function appends float64 to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_float64` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a float64.
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

/// The `append_int` function appends int to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_int` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a int.
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

/// The `append_string` function appends string to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_string` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a string.
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

/// The `append_time` function appends time to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_time` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a time.
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

/// The `append_timestamp` function appends timestamp to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_timestamp` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a timestamp.
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

/// The `append_fixed_size_binary` function appends fixed_size_binary to builder
///
/// Arguments:
///
/// * `builder`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `row`: The `row` parameter is reference to type implementing sqlx `Row`, providing the source data.
/// * `index`: The `index` parameter is reference to `str`, providing the location in the source data (row).
///
/// Returns:
///
/// The `append_fixed_size_binary` function returns a `Result` containing a unit value or a
/// `NisabaError` if an error occurs during appending a fixed_size_binary.
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

    use crate::{AnalyzerConfig, LatentStore, types::FieldDef};

    use super::*;

    #[tokio::test]
    async fn test_mysql_inference() {
        let source = Source::mysql()
            .auth("mysql", "mysql")
            .host("localhost")
            .port(3306)
            .database("mysql_store")
            .build()
            .await
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

        let sql_inference = MySQLInferenceEngine::default();

        let result = sql_inference
            .mysql_store_infer(&source, stats, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 9);
    }

    #[tokio::test]
    async fn test_postgresql_inference() {
        let source = Source::postgresql()
            .auth("postgres", "postgres")
            .host("localhost")
            .database("postgres")
            .port(5432)
            .namespace("public")
            .build()
            .await
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

        let sql_inference = PostgreSQLInferenceEngine::default();

        let result = sql_inference
            .postgres_store_infer(&source, stats, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 9);
    }

    #[tokio::test]
    async fn test_sqlite_inference() {
        let source = Source::sqlite()
            .path("./assets/sqlite/nisaba.sqlite")
            .build()
            .await
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

        let sql_inference = SqliteInferenceEngine::default();

        let result = sql_inference
            .sqlite_store_infer(&source, stats, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(result.len(), 9);
    }
}
