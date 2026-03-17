use std::{collections::HashMap, path::PathBuf, sync::Arc};

use mongodb::{Client, options::ClientOptions};

use sqlx::{MySqlPool, PgPool, SqlitePool};
use uuid::Uuid;

use crate::error::NisabaError;

#[derive(Clone, Debug)]
pub enum SourceClient {
    File(PathBuf),
    MongoDB(mongodb::Client),
    MySQL(sqlx::MySqlPool),
    Postgres(sqlx::PgPool),
    Sqlite(sqlx::SqlitePool),
}

impl SourceClient {
    pub fn as_path(&self) -> Option<&PathBuf> {
        match self {
            SourceClient::File(path) => Some(path),
            _ => None,
        }
    }

    pub fn as_mongo(&self) -> Option<&mongodb::Client> {
        match self {
            SourceClient::MongoDB(client) => Some(client),
            _ => None,
        }
    }

    pub fn as_mysql(&self) -> Option<&sqlx::MySqlPool> {
        match self {
            SourceClient::MySQL(pool) => Some(pool),
            _ => None,
        }
    }

    pub fn as_postgres(&self) -> Option<&sqlx::PgPool> {
        match self {
            SourceClient::Postgres(pool) => Some(pool),
            _ => None,
        }
    }

    pub fn as_sqlite(&self) -> Option<&sqlx::SqlitePool> {
        match self {
            SourceClient::Sqlite(pool) => Some(pool),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SourceType {
    FileStore(FileStoreType),
    Database(DatabaseType),
}

/// File-based Data Stores
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FileStoreType {
    Csv,
    Excel,
    Parquet,
}

impl std::fmt::Display for FileStoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv => write!(f, "csv"),
            Self::Excel => write!(f, "excel"),
            Self::Parquet => write!(f, "parquet"),
        }
    }
}

/// Network-based Data Sources
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DatabaseType {
    MongoDB,
    MySQL,
    PostgreSQL,
    SQLite,
}

impl std::fmt::Display for DatabaseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MongoDB => write!(f, "mongodb"),
            Self::MySQL => write!(f, "mysql"),
            Self::PostgreSQL => write!(f, "postgresql"),
            Self::SQLite => write!(f, "sqlite"),
        }
    }
}

/// Data Source with connection and metadata like identifier
pub struct Source {
    pub client: Arc<SourceClient>,
    pub metadata: SourceMetadata,
}

pub struct SourceMetadata {
    pub silo_id: String,
    pub source_type: SourceType,
    /// Should contain namespace for postgres, pattern for files
    /// Must contain database for mongodb
    pub extra: HashMap<Extra, String>,
    pub num_rows: usize,
    pub has_header: bool,
}

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum Extra {
    PostgresSchema,
    FilePattern,
    MongoDatabase,
    HasHeader,
}

impl Source {
    pub fn files(kind: FileStoreType) -> FileSourceBuilder {
        FileSourceBuilder::builder(kind)
    }

    pub fn mongodb() -> MongoSourceBuilder {
        MongoSourceBuilder::builder()
    }

    pub fn mysql() -> MySQLSourceBuilder {
        MySQLSourceBuilder::builder()
    }

    pub fn postgresql() -> PostgresSourceBuilder {
        PostgresSourceBuilder::builder()
    }

    pub fn sqlite() -> SQLiteSourceBuilder {
        SQLiteSourceBuilder::builder()
    }
}

#[derive(Clone, Debug)]
pub struct Credentials {
    pub username: String,
    pub password: String,
}

impl Credentials {
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            username: username.into(),
            password: password.into(),
        }
    }
}

pub struct FileSourceBuilder {
    filestore_type: FileStoreType,
    path: Option<PathBuf>,
    num_rows: Option<usize>,
    pattern: Option<String>,
    has_header: Option<bool>,
}

impl FileSourceBuilder {
    pub fn builder(kind: FileStoreType) -> FileSourceBuilder {
        FileSourceBuilder {
            filestore_type: kind,
            path: None,
            num_rows: None,
            pattern: None,
            has_header: None,
        }
    }

    pub fn path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn num_rows(mut self, num_rows: usize) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    pub fn has_header(mut self, has_header: bool) -> Self {
        self.has_header = Some(has_header);
        self
    }

    pub fn build(self) -> Result<Source, NisabaError> {
        let path = validate_path(self.path)?;

        let mut extra = HashMap::new();

        if let Some(pattern) = self.pattern
            && !pattern.is_empty()
        {
            extra.insert(Extra::FilePattern, pattern);
        }

        let has_header = if let Some(flag) = self.has_header {
            if flag && matches!(self.filestore_type, FileStoreType::Parquet) {
                return Err(NisabaError::Invalid(
                    "has_header flag is not required for Parquet".into(),
                ));
            }
            flag
        } else {
            true
        };

        let num_rows = validate_num_rows(self.num_rows)?;

        Ok(Source {
            client: Arc::new(SourceClient::File(path)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", self.filestore_type.clone(), Uuid::now_v7()),
                source_type: SourceType::FileStore(self.filestore_type),
                extra,
                num_rows: num_rows.unwrap_or(10),
                has_header,
            },
        })
    }
}

pub struct MongoSourceBuilder {
    host: Option<String>,
    port: Option<u16>,
    database: Option<String>,
    credentials: Option<Credentials>,
    use_srv: bool,
    pool_size: Option<usize>,
    num_rows: Option<usize>,
}

impl MongoSourceBuilder {
    pub fn builder() -> MongoSourceBuilder {
        MongoSourceBuilder {
            host: None,
            port: None,
            database: None,
            credentials: None,
            use_srv: false,
            pool_size: None,
            num_rows: None,
        }
    }

    pub fn uri(uri: impl Into<String>, num_rows: Option<usize>) -> Result<Source, NisabaError> {
        let uri = uri.into();

        if uri.is_empty() {
            return Err(NisabaError::Invalid(
                "Invalid MongoDB uri. Cannot be empty".into(),
            ));
        }

        let client_options = ClientOptions::parse(uri).run()?;

        let client = Client::with_options(client_options)?;

        let num_rows = validate_num_rows(num_rows)?;

        Ok(Source {
            client: Arc::new(SourceClient::MongoDB(client)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", DatabaseType::MongoDB, Uuid::now_v7()),
                source_type: SourceType::Database(DatabaseType::MongoDB),
                extra: HashMap::new(),
                num_rows: num_rows.unwrap_or(10),
                has_header: false,
            },
        })
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    pub fn auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.credentials = Some(Credentials::basic(username, password));
        self
    }

    pub fn use_srv(mut self, use_srv: bool) -> Self {
        self.use_srv = use_srv;
        self
    }

    pub fn pool_size(mut self, pool_size: usize) -> Self {
        self.pool_size = Some(pool_size);
        self
    }

    pub fn num_rows(mut self, num_rows: usize) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub fn build(self) -> Result<Source, NisabaError> {
        let host = validate_host(self.host)?;

        let creds = validate_credentials(self.credentials)?;

        let database = validate_database(self.database)?;

        let server_address = if !self.use_srv {
            let port = self
                .port
                .ok_or(NisabaError::Missing("Host cannot be empty".into()))?;

            validate_port(port)?;

            mongodb::options::ServerAddress::Tcp {
                host,
                port: Some(port),
            }
        } else {
            mongodb::options::ServerAddress::Tcp { host, port: None }
        };

        let credential = mongodb::options::Credential::builder()
            .username(Some(creds.username))
            .password(Some(creds.password))
            .build();

        let client_options = ClientOptions::builder()
            .app_name(Some("nisaba".into()))
            .credential(Some(credential))
            .min_pool_size(self.pool_size.map(|v| v as u32))
            .default_database(Some(database.clone()))
            .hosts(vec![server_address])
            .build();

        let client = Client::with_options(client_options)?;

        let mut extra = HashMap::new();
        extra.insert(Extra::MongoDatabase, database);

        let num_rows = validate_num_rows(self.num_rows)?;

        Ok(Source {
            client: Arc::new(SourceClient::MongoDB(client)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", DatabaseType::MongoDB, Uuid::now_v7()),
                source_type: SourceType::Database(DatabaseType::MongoDB),
                extra,
                num_rows: num_rows.unwrap_or(10),
                has_header: false,
            },
        })
    }
}

pub struct MySQLSourceBuilder {
    host: Option<String>,
    port: Option<u16>,
    database: Option<String>,
    credentials: Option<Credentials>,
    num_rows: Option<usize>,
}

impl MySQLSourceBuilder {
    pub fn builder() -> Self {
        Self {
            host: None,
            port: None,
            database: None,
            credentials: None,
            num_rows: None,
        }
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    pub fn auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.credentials = Some(Credentials::basic(username, password));
        self
    }

    pub fn num_rows(mut self, num_rows: usize) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub async fn build(self) -> Result<Source, NisabaError> {
        let database = validate_database(self.database)?;

        let host = validate_host(self.host)?;

        let port = self
            .port
            .ok_or(NisabaError::Missing("Host cannot be empty".into()))?;

        validate_port(port)?;

        let creds = validate_credentials(self.credentials)?;

        let conn_str = format!(
            "mysql://{}:{}@{}:{}/{}",
            url_encode(creds.username.as_str()),
            url_encode(creds.password.as_str()),
            host,
            port,
            database
        );

        let num_rows = validate_num_rows(self.num_rows)?;

        let pool = MySqlPool::connect(&conn_str).await?;

        Ok(Source {
            client: Arc::new(SourceClient::MySQL(pool)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", DatabaseType::MySQL, Uuid::now_v7()),
                source_type: SourceType::Database(DatabaseType::MySQL),
                extra: HashMap::new(),
                num_rows: num_rows.unwrap_or(10),
                has_header: false,
            },
        })
    }
}

// PostgreSQL Data Source Builder

pub struct PostgresSourceBuilder {
    host: Option<String>,
    port: Option<u16>,
    database: Option<String>,
    namespace: Option<String>,
    credentials: Option<Credentials>,
    num_rows: Option<usize>,
}

impl PostgresSourceBuilder {
    pub fn builder() -> PostgresSourceBuilder {
        PostgresSourceBuilder {
            host: None,
            port: Some(5432),
            database: None,
            namespace: None,
            credentials: None,
            num_rows: None,
        }
    }
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    pub fn auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.credentials = Some(Credentials::basic(username, password));
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    pub fn num_rows(mut self, num_rows: usize) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub async fn build(self) -> Result<Source, NisabaError> {
        let namespace = validate_namespace(self.namespace)?;

        let database = validate_database(self.database)?;

        let creds = validate_credentials(self.credentials)?;

        let host = validate_host(self.host)?;

        let port = self
            .port
            .ok_or(NisabaError::Missing("Host cannot be empty".into()))?;

        validate_port(port)?;

        let conn_str = format!(
            "postgresql://{}:{}@{}:{}/{}",
            url_encode(creds.username.as_str()),
            url_encode(creds.password.as_str()),
            host,
            port,
            database,
        );

        let mut extra = HashMap::new();

        if let Some(ns) = namespace {
            extra.insert(Extra::FilePattern, ns);
        }

        let num_rows = validate_num_rows(self.num_rows)?;

        let pool = PgPool::connect(&conn_str).await?;

        Ok(Source {
            client: Arc::new(SourceClient::Postgres(pool)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", DatabaseType::PostgreSQL, Uuid::now_v7()),
                source_type: SourceType::Database(DatabaseType::PostgreSQL),
                extra,
                num_rows: num_rows.unwrap_or(10),
                has_header: false,
            },
        })
    }
}

pub struct SQLiteSourceBuilder {
    path: Option<String>,
    num_rows: Option<usize>,
}

impl SQLiteSourceBuilder {
    pub fn builder() -> Self {
        SQLiteSourceBuilder {
            path: None,
            num_rows: None,
        }
    }

    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn num_rows(mut self, num_rows: usize) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub async fn build(self) -> Result<Source, NisabaError> {
        let path = self
            .path
            .ok_or(NisabaError::Missing("Path cannot be empty".into()))?;

        if path.is_empty() {
            return Err(NisabaError::Invalid("Path cannot be empty string".into()));
        }

        let num_rows = validate_num_rows(self.num_rows)?;

        let pool = SqlitePool::connect(&path).await?;

        Ok(Source {
            client: Arc::new(SourceClient::Sqlite(pool)),
            metadata: SourceMetadata {
                silo_id: format!("{}-{}", DatabaseType::SQLite, Uuid::now_v7()),
                source_type: SourceType::Database(DatabaseType::SQLite),
                extra: HashMap::new(),
                num_rows: num_rows.unwrap_or(10),
                has_header: false,
            },
        })
    }
}

/// The `validate_credentials` function runs specific validations on Credentials
///
/// Returns:
///
/// A `Credential` Result on success or NisabaError if validation fails.
fn validate_credentials(credentials: Option<Credentials>) -> Result<Credentials, NisabaError> {
    let creds = credentials.ok_or(NisabaError::Missing("No credentials provided/found".into()))?;

    if creds.username.trim().is_empty() {
        return Err(NisabaError::Missing("Username cannot be empty".into()));
    }

    if creds.password.trim().is_empty() {
        return Err(NisabaError::Missing("Password cannot be empty".into()));
    }

    Ok(creds)
}

/// The `validate_database_name` function runs specific validations on database_name
///
/// Returns:
///
/// A String Result on success or NisabaError if validation fails.
fn validate_database(database: Option<String>) -> Result<String, NisabaError> {
    let database = database.ok_or(NisabaError::Missing("Database cannot be empty".into()))?;

    if database.trim().is_empty() {
        return Err(NisabaError::Missing(
            "Database name cannot be empty string".into(),
        ));
    }

    if database.contains(|c: char| c.is_whitespace() || c == '/' || c == '\\') {
        return Err(NisabaError::Invalid(
            "Database name contains invalid characters".into(),
        ));
    }

    Ok(database)
}

/// The `validate_host` function runs specific validations on host
///
/// Returns:
///
/// A String Result on success or NisabaError if validation fails.
fn validate_host(host: Option<String>) -> Result<String, NisabaError> {
    let host = host.ok_or(NisabaError::Missing("Host cannot be empty".into()))?;

    if host.trim().is_empty() {
        return Err(NisabaError::Missing("Host cannot be empty string".into()));
    }

    if host.contains(|c: char| c.is_whitespace() || c == '@' || c == '/') {
        return Err(NisabaError::Invalid(
            "Host contains invalid characters".into(),
        ));
    }

    Ok(host)
}

/// The `validate_namespace` function runs specific validations on host for PostgreSQL
///
/// Returns:
///
/// A Option Result on success or NisabaError if validation fails.
fn validate_namespace(namespace: Option<String>) -> Result<Option<String>, NisabaError> {
    let namespace = match namespace {
        Some(nm) => {
            if nm.contains(|c: char| c.is_whitespace() || c == '@' || c == '/' || c == '\\') {
                return Err(NisabaError::Invalid(
                    "Namespace/schema contains invalid characters".into(),
                ));
            }

            Some(nm)
        }
        None => None,
    };
    Ok(namespace)
}

fn validate_num_rows(num_rows: Option<usize>) -> Result<Option<usize>, NisabaError> {
    match num_rows {
        Some(v) => {
            if v == 0 {
                Err(NisabaError::Invalid(
                    "Invalid number of rows (0) provided".into(),
                ))?
            }
            Ok(Some(v))
        }
        None => Ok(None),
    }
}

/// The `validate_path` function runs specific validations on the path provided for file-based backend
///
/// Returns:
///
/// A Result of unit on success or NisabaError if validation fails.
fn validate_path(path: Option<PathBuf>) -> Result<PathBuf, NisabaError> {
    let path = path.ok_or(NisabaError::Missing("No path provided".into()))?;

    if !path.exists() {
        return Err(NisabaError::Missing("Path not found".into()));
    }

    Ok(path)
}

/// The `validate_port` function runs specific validations on port
///
/// Returns:
///
/// A Result of unit on success or NisabaError if validation fails.
fn validate_port(port: u16) -> Result<(), NisabaError> {
    if port == 0 {
        return Err(NisabaError::Invalid("Port cannot be 0".into()));
    }
    Ok(())
}

/// The `url_encode` function encodes a connection string to url safe
///
/// Returns:
///
/// A String result of encoded connection string.
fn url_encode(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '@' => "%40".to_string(),
            ':' => "%3A".to_string(),
            '/' => "%2F".to_string(),
            ' ' => "%20".to_string(),
            c if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == '~' => {
                c.to_string()
            }
            c => format!("%{:02X}", c as u8),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_source() {
        let source = Source::files(FileStoreType::Csv)
            .has_header(true)
            .num_rows(10)
            .path("./assets/csv")
            .build()
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::FileStore(FileStoreType::Csv)
        ));

        assert!(source.client.as_path().is_some());

        assert!(matches!(*source.client, SourceClient::File(..)));
    }

    #[test]
    fn test_excel_source() {
        let source = Source::files(FileStoreType::Excel)
            .has_header(true)
            .num_rows(10)
            .path("./assets/xlsx")
            .build()
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::FileStore(FileStoreType::Excel)
        ));

        assert!(source.client.as_path().is_some());

        assert!(matches!(*source.client, SourceClient::File(..)));
    }

    #[test]
    fn test_parquet_source() {
        let source = Source::files(FileStoreType::Parquet)
            .num_rows(10)
            .path("./assets/parquet")
            .build()
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::FileStore(FileStoreType::Parquet)
        ));

        assert!(source.client.as_path().is_some());

        assert!(matches!(*source.client, SourceClient::File(..)));
    }

    #[test]
    fn test_standard_mongo_source() {
        let source = Source::mongodb()
            .auth("username", "password")
            .host("localhost")
            .database("mydb")
            .pool_size(5)
            .port(27017)
            .build()
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::MongoDB)
        ));

        assert!(source.client.as_mongo().is_some());

        assert!(matches!(*source.client, SourceClient::MongoDB(..)));
    }

    #[test]
    fn test_srv_mongo_source() {
        let source = Source::mongodb()
            .auth("username", "password")
            .host("localhost")
            .database("mydb")
            .pool_size(5)
            .port(27017)
            .use_srv(true)
            .build()
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::MongoDB)
        ));

        assert!(source.client.as_mongo().is_some());

        assert!(matches!(*source.client, SourceClient::MongoDB(..)));
    }

    #[tokio::test]
    async fn test_mysql_source() {
        let source = Source::mysql()
            .auth("mysql", "mysql")
            .host("localhost")
            .database("mysql_store")
            .port(3306)
            .build()
            .await
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::MySQL)
        ));

        assert!(source.client.as_mysql().is_some());

        assert!(matches!(*source.client, SourceClient::MySQL(..)));
    }

    #[tokio::test]
    async fn test_postgres_source() {
        let source = Source::postgresql()
            .auth("postgres", "postgres")
            .host("localhost")
            .port(5432)
            .database("postgres")
            .namespace("public")
            .build()
            .await
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::PostgreSQL)
        ));

        assert!(source.client.as_postgres().is_some());

        assert!(matches!(*source.client, SourceClient::Postgres(..)));
    }

    #[tokio::test]
    async fn test_postgres_source_without_namespace() {
        let source = Source::postgresql()
            .auth("postgres", "postgres")
            .host("localhost")
            .port(5432)
            .database("postgres")
            .build()
            .await
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::PostgreSQL)
        ));

        assert!(source.client.as_postgres().is_some());

        assert!(matches!(*source.client, SourceClient::Postgres(..)));
    }

    #[tokio::test]
    async fn test_sqlite_source() {
        let source = Source::sqlite()
            .path("./assets/sqlite/nisaba.sqlite")
            .build()
            .await
            .unwrap();

        assert!(matches!(
            source.metadata.source_type,
            SourceType::Database(DatabaseType::SQLite)
        ));

        assert!(source.client.as_sqlite().is_some());

        assert!(matches!(*source.client, SourceClient::Sqlite(..)));
    }
}
