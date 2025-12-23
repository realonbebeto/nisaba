use arrow::{
    array::{AsArray, RecordBatch, RecordBatchReader},
    datatypes::{DataType, Field, Fields, TimeUnit},
};
use arrow_odbc::{
    OdbcReaderBuilder,
    odbc_api::{ConnectionOptions, Cursor, Environment},
};
use futures::{TryStreamExt, executor::block_on};
use mongodb::{
    Client,
    bson::{Bson, Document, doc},
    options::ClientOptions,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use std::{
    collections::HashMap,
    fs::{self, File},
    sync::Arc,
};
use uuid::Uuid;

use crate::{
    analyzer::catalog::{DataLocation, DataStoreType},
    analyzer::conversion::{InfoSchema, StdSchema},
    error::NError,
    types::TableDef,
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
    pub fn get_engine(&self, store_type: &DataStoreType) -> Option<&dyn SchemaInferenceEngine> {
        self.engines
            .values()
            .find(|eng| eng.can_handle(store_type))
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
    pub fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let engine = self.get_engine(&location.store_type).ok_or_else(|| {
            NError::Unsupported(format!(
                "No engine available for store type: {:?}",
                location.store_type
            ))
        })?;

        let table_defs = engine.infer_schema(location)?;

        Ok(table_defs)
    }

    pub fn discover_ecosystem(
        &self,
        locations: Vec<DataLocation>,
    ) -> Result<Vec<TableDef>, NError> {
        let mut table_defs = Vec::new();

        for location in locations {
            table_defs.extend(self.infer_schema(&location)?);
        }

        Ok(table_defs)
    }
}

/// Trait for schema inference engines
pub trait SchemaInferenceEngine: std::fmt::Debug + Send + Sync {
    /// Infer schema from a data source
    fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError>;

    /// Check if this engine can handle the given data source
    fn can_handle(&self, store_type: &DataStoreType) -> bool;

    /// Get the name of the inference engine
    fn engine_name(&self) -> &str;
}

// =================================================
// Flat-File Inference Engine
// =================================================

/// File inference engine for common data file formats
#[derive(Debug)]
pub struct FileInferenceEngine {
    sample_size: u32,
    profile_data: bool,
}

impl Default for FileInferenceEngine {
    fn default() -> Self {
        Self {
            sample_size: 1000,
            profile_data: true,
        }
    }
}

impl FileInferenceEngine {
    pub fn new() -> Self {
        FileInferenceEngine::default()
    }

    pub fn with_sample_size(mut self, size: u32) -> Self {
        self.sample_size = size;
        self
    }

    pub fn with_profiling(mut self, profile: bool) -> Self {
        self.profile_data = profile;
        self
    }

    fn infer_from_csv(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let conn_str = location.connection_string()?;
        let odbc_env = Environment::new()?;
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let connection =
            odbc_env.connect_with_connection_string(&conn_str, ConnectionOptions::default())?;

        let mut cursor = connection.tables("", "", "", "TABLE")?;

        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        while let Some(mut row) = cursor.next_row()? {
            let mut buf = Vec::new();
            let _ = row.get_text(3, &mut buf);
            let table_name = String::from_utf8(buf).unwrap();

            let query = format!("SELECT * FROM {} LIMIT 0", table_name);

            // Execute query and create result set
            let cursor = connection
                .execute(&query, (), None)?
                .expect("SELECT statement must produce a cursor");

            let reader = OdbcReaderBuilder::new().build(cursor)?;

            let schema = reader.schema();

            let result: Vec<InfoSchema> = schema
                .fields()
                .iter()
                .map(|field| InfoSchema {
                    silo_id: silo_id.clone(),
                    table_schema: "default".into(),
                    table_name: table_name.clone(),
                    column_name: field.name().into(),
                    udt_name: format!("{}", field.data_type()),
                    data_type: field.extension_type_name().unwrap_or_default().to_string(),
                    column_default: None,
                    character_maximum_length: "".into(),
                    is_nullable: format!("{}", field.is_nullable()),
                })
                .collect();

            info_schemas.extend(result);
        }

        Ok(info_schemas)
    }

    fn infer_from_excel(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let dir_str = &location.dir_path.clone().ok_or(NError::InvalidPath(
            "Directory with Excel workbooks not provided".into(),
        ))?;
        // Get all excel filenames
        let entries = fs::read_dir(dir_str).map_err(|e| NError::InvalidPath(e.to_string()))?;
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| NError::InvalidPath(e.to_string()))?;
            let path = entry.path();

            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| {
                    [".xls", ".xlsx", ".xlsm", ".xlsb"]
                        .iter()
                        .any(|sufx| s.ends_with(sufx))
                })
                .unwrap_or(false)
            {
                let file_path = path.to_str().unwrap();
                let conn_str = location.connection_string()?.replace(dir_str, file_path);

                let odbc_env = Environment::new()?;

                let connection = odbc_env
                    .connect_with_connection_string(&conn_str, ConnectionOptions::default())?;

                let mut cursor = connection.tables("", "", "", "TABLE")?;

                while let Some(mut row) = cursor.next_row()? {
                    let mut buf = Vec::new();
                    let _ = row.get_text(3, &mut buf);
                    let table_name = String::from_utf8(buf).unwrap();

                    let query = format!("SELECT * FROM {} LIMIT 0", table_name);

                    // Execute query and create result set
                    let cursor = connection
                        .execute(&query, (), None)?
                        .expect("SELECT statement must produce a cursor");

                    let reader = OdbcReaderBuilder::new().build(cursor)?;

                    let schema = reader.schema();

                    let result: Vec<InfoSchema> = schema
                        .fields()
                        .iter()
                        .map(|field| InfoSchema {
                            silo_id: silo_id.clone(),
                            table_schema: "default".into(),
                            table_name: table_name.clone(),
                            column_name: field.name().into(),
                            udt_name: format!("{}", field.data_type()),
                            data_type: field.extension_type_name().unwrap_or_default().to_string(),
                            column_default: None,
                            character_maximum_length: "".into(),
                            is_nullable: format!("{}", field.is_nullable()),
                        })
                        .collect();

                    info_schemas.extend(result);
                }
            }
        }

        Ok(info_schemas)
    }

    fn infer_from_parquet(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let dir_str = location.connection_string()?;
        // Get all parquet filenames
        let entries = fs::read_dir(dir_str).map_err(|e| NError::InvalidPath(e.to_string()))?;
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| NError::InvalidPath(e.to_string()))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                let file = File::open(path).map_err(|e| NError::FileError(e.to_string()))?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let schema = builder.schema();

                let result: Vec<InfoSchema> = schema
                    .fields()
                    .iter()
                    .map(|field| InfoSchema {
                        silo_id: silo_id.clone(),
                        table_schema: "default".into(),
                        table_name: "default".into(),
                        column_name: field.name().into(),
                        udt_name: format!("{}", field.data_type()),
                        data_type: field.extension_type_name().unwrap_or_default().to_string(),
                        column_default: None,
                        character_maximum_length: "".into(),
                        is_nullable: format!("{}", field.is_nullable()),
                    })
                    .collect();

                info_schemas.extend(result);
            }
        }

        Ok(info_schemas)
    }
}

impl SchemaInferenceEngine for FileInferenceEngine {
    fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let info_schemas = match location.store_type {
            DataStoreType::Csv => self.infer_from_csv(location),
            DataStoreType::Excel => self.infer_from_excel(location),
            DataStoreType::Parquet => self.infer_from_parquet(location),
            _ => Err(NError::Unsupported(format!(
                "{:?} file store unsupported by File engine",
                location.store_type
            ))),
        };

        info_schemas?.try_into_table_defs()
    }

    fn can_handle(&self, store_type: &DataStoreType) -> bool {
        matches!(
            store_type,
            DataStoreType::Avro
                | DataStoreType::Csv
                | DataStoreType::Excel
                | DataStoreType::Json
                | DataStoreType::JsonL
                | DataStoreType::Feather
                | DataStoreType::Orc
                | DataStoreType::Parquet
        )
    }

    fn engine_name(&self) -> &str {
        "File"
    }
}

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

    fn infer_from_mysql(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let conn_str = location.connection_string()?;
        let schema_name = location.database.clone().unwrap();
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let query = format!("SELECT TABLE_SCHEMA AS table_schema, TABLE_NAME AS table_name, COLUMN_NAME AS column_name, COLUMN_TYPE AS udt_name, IS_NULLABLE AS is_nullable  FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = {};", schema_name);

        let info_schemas = self.odbc_reader(&conn_str, &query, &silo_id)?;

        Ok(info_schemas)
    }

    fn infer_from_postgres(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let conn_str = location.connection_string()?;
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let query = format!(
            "SELECT table_schema, table_name, column_name, data_type, udt_name, is_nullable FROM information_schema.columns WHERE table_schema = {}",
            location.namespace.clone().unwrap_or("public".into())
        );

        let info_schemas = self.odbc_reader(&conn_str, &query, &silo_id)?;

        Ok(info_schemas)
    }

    fn infer_from_sqlite(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let conn_str = location.connection_string()?;
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());

        let query = "SELECT sql FROM sqlite_master WHERE type = 'table';";

        let batches = self.odbc_read_batches(&conn_str, query)?;

        // Execute query and create result set
        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        for batch in batches {
            let sql_str = batch.column(0).as_string::<i64>();

            for i in 0..batch.num_rows() {
                let v = sql_str.value(i);
                let batch_result = self.parse_sqlite_master(v, &silo_id)?;

                info_schemas.extend(batch_result);
            }
        }

        Ok(info_schemas)
    }

    fn odbc_reader(
        &self,
        conn_str: &str,
        query: &str,
        silo_id: &str,
    ) -> Result<Vec<InfoSchema>, NError> {
        let batches = self.odbc_read_batches(conn_str, query)?;
        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        for batch in batches {
            let table_schema = batch.column(0).as_string::<i64>();
            let table_name = batch.column(1).as_string::<i64>();
            let column_name = batch.column(2).as_string::<i64>();
            let data_type = batch.column(3).as_string::<i64>();
            let udt_name = batch.column(4).as_string::<i64>();
            let is_nullable = batch.column(5).as_string::<i64>();

            for i in 0..batch.num_rows() {
                let is = InfoSchema {
                    silo_id: silo_id.into(),
                    table_schema: table_schema.value(i).into(),
                    table_name: table_name.value(i).into(),
                    column_name: column_name.value(i).into(),
                    udt_name: udt_name.value(i).into(),
                    data_type: data_type.value(i).into(),
                    column_default: None,
                    character_maximum_length: "".into(),
                    is_nullable: is_nullable.value(i).to_string(),
                };

                info_schemas.push(is);
            }
        }

        Ok(info_schemas)
    }

    fn odbc_read_batches(&self, conn_str: &str, query: &str) -> Result<Vec<RecordBatch>, NError> {
        let odbc_env = Environment::new()?;

        let connection =
            odbc_env.connect_with_connection_string(conn_str, ConnectionOptions::default())?;

        // Execute query and create result set
        let cursor = connection
            .execute(query, (), None)?
            .expect("SELECT statement must produce a cursor");

        let reader = OdbcReaderBuilder::new().build(cursor)?;

        let batches: Result<Vec<_>, _> = reader.collect();

        batches.map_err(Into::into)
    }

    fn parse_sqlite_master(&self, sql_str: &str, silo_id: &str) -> Result<Vec<InfoSchema>, NError> {
        let sql_str = sql_str.trim().replace('\n', " ").to_lowercase();

        let re = Regex::new(r#"(?i)CREATE\s+TABLE\s+"?(?P<table>\w+)"?\s*\((?P<cols>.+)\)"#)?;

        let caps = re
            .captures(&sql_str)
            .ok_or(regex::Error::Syntax("No match found <Table DDL>".into()))?;

        let table_name = caps
            .name("table")
            .ok_or(regex::Error::Syntax("Table name not found".into()))?
            .as_str()
            .to_lowercase();

        let cols = caps
            .name("cols")
            .ok_or(regex::Error::Syntax("Columns not found".into()))?
            .as_str();

        let columns: Vec<&str> = cols
            .split(",")
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .collect();

        fn parse_col(table_name: &str, col: &str, silo_id: &str) -> Result<InfoSchema, NError> {
            let col_re = Regex::new(
                r#"(?i)"?(?P<name>\w+)"?\s+(?P<dtype>\w+)(?:\s+DEFAULT\s+(?P<default>.+))?"#,
            )?;

            let col_caps = col_re
                .captures(col)
                .ok_or(regex::Error::Syntax("No match found <Column>".into()))?;

            let data_type = col_caps
                .name("dtype")
                .ok_or(regex::Error::Syntax("Data type not found".into()))?
                .as_str()
                .to_lowercase();

            let column_name = col_caps
                .name("name")
                .ok_or(regex::Error::Syntax("Column name not found".into()))?
                .as_str()
                .to_string();

            Ok(InfoSchema {
                silo_id: silo_id.into(),
                table_schema: "main".to_string(),
                table_name: table_name.to_string(),
                column_name,
                column_default: col_caps.name("default").map(|d| d.as_str().to_string()),
                is_nullable: "YES".to_string(),
                data_type: data_type.clone(),
                character_maximum_length: "".to_string(),
                udt_name: data_type,
            })
        }

        let info_schemas: Result<Vec<InfoSchema>, _> = columns
            .into_iter()
            .map(|c| parse_col(&table_name, c, silo_id))
            .collect();

        match info_schemas {
            Ok(result) => Ok(result),
            Err(e) => Err(e),
        }
    }
}

impl SchemaInferenceEngine for SQLInferenceEngine {
    fn infer_schema(&self, location: &DataLocation) -> Result<Vec<TableDef>, NError> {
        let info_schemas = match location.store_type {
            DataStoreType::MySQL => self.infer_from_mysql(location),
            DataStoreType::PostgreSQL => self.infer_from_postgres(location),
            DataStoreType::SQLite => self.infer_from_sqlite(location),
            _ => Err(NError::Unsupported(format!(
                "{:?} SQL store provided unsupported by SQL engine",
                location.store_type
            ))),
        };

        info_schemas?.try_into_table_defs()
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

    async fn infer_from_mongodb(&self, location: &DataLocation) -> Result<Vec<InfoSchema>, NError> {
        let silo_id = format!("{}-{}", location.store_type, Uuid::now_v7());
        let conn_str = location.connection_string()?;
        let client_options = ClientOptions::parse(&conn_str).await?;
        let db_name = client_options.clone().default_database.unwrap();

        let client = Client::with_options(client_options)?;
        let db = client.database(&db_name);
        let collections = db.list_collection_names().await?;

        let mut info_schemas: Vec<InfoSchema> = Vec::new();

        for collection_name in collections {
            let collection = db.collection::<Document>(&collection_name);
            let mut cursor = collection
                .find(doc! {})
                .limit(i64::from(self.sample_size))
                .await?;

            let result = self
                .mongo_collection_infer(&db_name, &collection_name, &mut cursor, &silo_id)
                .await?;

            info_schemas.extend(result);
        }

        Ok(info_schemas)
    }

    async fn mongo_collection_infer(
        &self,
        db_name: &str,
        collection_name: &str,
        cursor: &mut mongodb::Cursor<Document>,
        silo_id: &str,
    ) -> Result<Vec<InfoSchema>, NError> {
        let mut field_types: HashMap<String, DataType> = HashMap::new();

        while let Some(doc) = cursor.try_next().await? {
            self.mongo_document_infer(&doc, &mut field_types);
        }

        let fields: Vec<InfoSchema> = field_types
            .into_iter()
            .map(|(name, dtype)| InfoSchema {
                silo_id: silo_id.into(),
                table_schema: db_name.into(),
                table_name: collection_name.into(),
                column_name: name,
                column_default: None,
                is_nullable: "yes".into(),
                data_type: format!("{}", dtype),
                character_maximum_length: "".into(),
                udt_name: format!("{}", dtype),
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
        let info_schemas = match location.store_type {
            DataStoreType::MongoDB => block_on(self.infer_from_mongodb(location)),
            _ => Err(NError::Unsupported(format!(
                "{:?} NoSQL store provided unsupported by NoSQL engine",
                location.store_type
            ))),
        };

        info_schemas?.try_into_table_defs()
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
