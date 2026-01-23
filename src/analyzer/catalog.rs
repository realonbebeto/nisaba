use std::path::Path;

use crate::error::NisabaError;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum StorageBackend {
    Csv,
    Excel,
    MongoDB,
    MySQL,
    Parquet,
    PostgreSQL,
    SQLite,
}

impl std::fmt::Display for StorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv => write!(f, "csv"),
            Self::Excel => write!(f, "excel"),
            Self::MongoDB => write!(f, "mongodb"),
            Self::MySQL => write!(f, "mysql"),
            Self::Parquet => write!(f, "parquet"),
            Self::PostgreSQL => write!(f, "postgresql"),
            Self::SQLite => write!(f, "sqlite"),
        }
    }
}

impl StorageBackend {
    /// The function checks to confirm if a Backend is file-based
    pub fn is_file_based(&self) -> bool {
        matches!(self, Self::Csv | Self::Excel | Self::Parquet | Self::SQLite)
    }

    /// The function checks if a backend is network-based
    pub fn is_network_based(&self) -> bool {
        matches!(self, Self::MongoDB | Self::MySQL | Self::PostgreSQL)
    }

    /// The function returns fields important of file-based or network-based backend
    pub fn required_fields(&self) -> Vec<&'static str> {
        match self {
            Self::Csv | Self::Excel | Self::Parquet | Self::SQLite => vec!["dir_path"],
            Self::MongoDB | Self::MySQL | Self::PostgreSQL => {
                vec!["host", "port", "database", "username", "password"]
            }
        }
    }

    /// The function checks on that PostgreSQL only allows namespace use
    pub fn supports_namespace(&self) -> bool {
        matches!(self, Self::PostgreSQL)
    }
}

#[derive(Clone, Debug)]
/// The `StorageConfig` sets out connection details to a store
pub struct StorageConfig {
    pub backend: StorageBackend,
    pub dir_path: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub database: Option<String>,
    pub namespace: Option<String>,
    // For MongoDB SRV connection strings
    pub use_srv: bool,
}

impl StorageConfig {
    /// The `new_file_backed` creates a `StorageConfig` with necessary
    /// validations for a file-based backend
    ///
    /// Arguments:
    ///
    /// * `backend`: The `backend` parameter  takes a StorageBackend variant
    ///
    /// * `dir_path`: The `dir_path` takes value that can be converted to String
    ///   pointing to the location/directory of the files.
    ///
    /// Returns:
    ///
    /// A Result of StorageConfig on success or NisabaError if validation fails.
    pub fn new_file_backend(
        backend: StorageBackend,
        dir_path: impl Into<String>,
    ) -> Result<Self, NisabaError> {
        if !backend.is_file_based() {
            return Err(NisabaError::Unsupported(format!(
                "{} is not a file-based store",
                backend
            )));
        }

        let path: String = dir_path.into();
        Self::validate_path(&path)?;

        Ok(Self {
            backend,
            dir_path: Some(path),
            host: None,
            port: None,
            username: None,
            password: None,
            database: None,
            namespace: None,
            use_srv: false,
        })
    }

    /// The `new_network_backed` creates a `StorageConfig` with necessary
    /// validations for a network-based backend
    ///
    /// Arguments:
    ///
    /// * `backend`: The `backend` parameter  takes a StorageBackend variant
    ///
    /// * `host`: The `host` takes value pointint to the IP of the server
    ///   where the store is resident.
    ///
    /// * `port`: The `port` parameter takes a u16 to give specific access to
    ///   the desired store.
    ///
    /// * `database`: The `database` parameter takes a value that can be String to
    ///   indicate the identity/name of the physical/logical instance.
    ///
    /// * `username`: The `username` parameter takes a value that can be String as identity to
    ///   user allowed to access the storage backend.
    ///
    /// * `password`: The `password` parameter takes a value that can be String as secret
    ///   key to access the the storage backend
    ///
    /// * `namespace`: The `namespace` parameter takes and optional value that can be String
    ///   to point to specific Schema/Grouping tables
    ///
    /// Returns:
    ///
    /// A Result of StorageConfig on success or NisabaError if validation fails.
    pub fn new_network_backend(
        backend: StorageBackend,
        host: impl Into<String>,
        port: u16,
        database: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
        namespace: Option<impl Into<String>>,
    ) -> Result<Self, NisabaError> {
        if !backend.is_network_based() {
            return Err(NisabaError::Unsupported(format!(
                "{} is not a network-based store type",
                backend
            )));
        }

        if !backend.supports_namespace() && namespace.is_some() {
            return Err(NisabaError::Unsupported(format!(
                "{} does not require namespace/schema",
                backend
            )));
        }

        let host: String = host.into();
        let database: String = database.into();
        let username: String = username.into();
        let password: String = password.into();
        let namespace: Option<String> = namespace.map(|n| n.into());

        Self::validate_host(&host)?;
        Self::validate_database_name(&database)?;
        Self::validate_credentials(&username, &password)?;

        if let Some(v) = &namespace {
            Self::validate_namespace(v)?
        }

        Ok(Self {
            backend,
            dir_path: None,
            host: Some(host),
            port: Some(port),
            username: Some(username),
            password: Some(password),
            database: Some(database),
            namespace,
            use_srv: false,
        })
    }

    /// The `new_mongo_srv_backed` creates a `StorageConfig` with necessary
    /// validations for an SRV configured MongoDB
    ///
    /// Arguments:
    ///
    /// * `host`: The `host` takes value pointint to the IP of the server
    ///   where the store is resident.
    ///
    /// * `database`: The `database` parameter takes a value that can be String to
    ///   indicate the identity/name of the physical/logical instance.
    ///
    /// * `username`: The `username` parameter takes a value that can be String as identity to
    ///   user allowed to access the storage backend.
    ///
    /// * `password`: The `password` parameter takes a value that can be String as secret
    ///   key to access the the storage backend
    ///
    /// Returns:
    ///
    /// A Result of StorageConfig on success or NisabaError if validation fails.
    pub fn new_mongo_srv_backend(
        host: impl Into<String>,
        database: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Result<Self, NisabaError> {
        let host: String = host.into();
        let database: String = database.into();
        let username: String = username.into();
        let password: String = password.into();

        Self::validate_host(&host)?;
        Self::validate_database_name(&database)?;
        Self::validate_credentials(&username, &password)?;

        Ok(Self {
            backend: StorageBackend::MongoDB,
            dir_path: None,
            host: Some(host),
            port: None,
            username: Some(username),
            password: Some(password),
            database: Some(database),
            namespace: None,
            use_srv: true,
        })
    }

    /// The `connection_string` function creates a full string for access
    ///
    /// Returns:
    ///
    /// A Result of String on success or NisabaError if a required detail is missing.
    pub fn connection_string(&self) -> Result<String, NisabaError> {
        self.validate()?;

        match self.backend {
            StorageBackend::Csv
            | StorageBackend::Excel
            | StorageBackend::Parquet
            | StorageBackend::SQLite => match &self.dir_path {
                Some(path) => Ok(path.clone()),
                None => Err(NisabaError::Missing(format!(
                    "Directory with {} not provided",
                    self.backend
                ))),
            },

            StorageBackend::MySQL => {
                match (
                    &self.host,
                    &self.port,
                    &self.database,
                    &self.username,
                    &self.password,
                ) {
                    (Some(host), Some(port), Some(database), Some(username), Some(password)) => {
                        Ok(format!(
                            "mysql://{}:{}@{}:{}/{}",
                            Self::url_encode(username),
                            Self::url_encode(password),
                            host,
                            port,
                            database
                        ))
                    }

                    _ => Err(NisabaError::Missing(
                        "Proper MySQL connection details not provided".into(),
                    )),
                }
            }
            StorageBackend::MongoDB => {
                match (&self.username, &self.password, &self.host, &self.database) {
                    (Some(username), Some(password), Some(host), Some(database)) => {
                        if !self.use_srv {
                            let port = self.port.unwrap();

                            Ok(format!(
                                "mongodb://{}:{}@{}:{}/{}?authSource=admin",
                                Self::url_encode(username),
                                Self::url_encode(password),
                                host,
                                port,
                                database
                            ))
                        } else {
                            Ok(format!(
                                "mongodb+srv://{}:{}@{}/{}?authSource=admin",
                                Self::url_encode(username),
                                Self::url_encode(password),
                                host,
                                database
                            ))
                        }
                    }
                    _ => Err(NisabaError::Missing(
                        "Proper MongoDB connection details not provided".into(),
                    )),
                }
            }
            StorageBackend::PostgreSQL => match (
                &self.host,
                &self.port,
                &self.database,
                &self.username,
                &self.password,
            ) {
                (Some(host), Some(port), Some(database), Some(username), Some(password)) => {
                    Ok(format!(
                        "postgresql://{}:{}@{}:{}/{}",
                        Self::url_encode(username),
                        Self::url_encode(password),
                        host,
                        port,
                        database,
                    ))
                }
                _ => Err(NisabaError::Missing(
                    "Proper PostgreSQL connection details not provided".into(),
                )),
            },
        }
    }

    /// The `validate` function runs necessary validations
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    pub fn validate(&self) -> Result<(), NisabaError> {
        match self.backend {
            StorageBackend::Csv
            | StorageBackend::Excel
            | StorageBackend::Parquet
            | StorageBackend::SQLite => self.validate_file_based_config(),
            StorageBackend::MongoDB | StorageBackend::MySQL | StorageBackend::PostgreSQL => {
                self.validate_network_based_config()
            }
        }
    }

    /// The `validate_file_based_config` function runs specific validations on file-based configs
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_file_based_config(&self) -> Result<(), NisabaError> {
        match &self.dir_path {
            Some(path) => Self::validate_path(path),
            None => Err(NisabaError::Missing(format!(
                "Directory path required for {}",
                self.backend
            ))),
        }
    }

    /// The `validate_network_based_config` function runs specific validations on network-based configs
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_network_based_config(&self) -> Result<(), NisabaError> {
        let host = self.host.as_ref().ok_or(NisabaError::Missing(format!(
            "Host required for {}",
            self.backend
        )))?;

        // Port is not required for MongoDB SRV connection
        if !self.use_srv || self.backend != StorageBackend::MongoDB {
            let port = self.port.ok_or(NisabaError::Missing(format!(
                "Port required for {}",
                self.backend
            )))?;

            Self::validate_port(port)?;
        } else if self.use_srv && self.port.is_some() {
            return Err(NisabaError::Invalid(
                "Port should not be specified for MongoDB SRV connections".into(),
            ));
        }

        let database = self.database.as_ref().ok_or(NisabaError::Missing(format!(
            "Database required for {}",
            self.backend
        )))?;

        let username = self.username.as_ref().ok_or(NisabaError::Missing(format!(
            "Username required for {}",
            self.backend
        )))?;

        let password = self.password.as_ref().ok_or(NisabaError::Missing(format!(
            "Password required for {}",
            self.backend
        )))?;

        Self::validate_host(host)?;
        Self::validate_database_name(database)?;
        Self::validate_credentials(username, password)?;

        Ok(())
    }

    /// The `validate_path` function runs specific validations on the path provided for file-based backend
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_path(path: &str) -> Result<(), NisabaError> {
        if path.trim().is_empty() {
            return Err(NisabaError::Missing("Path cannot be empty".into()));
        }

        let path_obj = Path::new(path);
        if path_obj.as_os_str().is_empty() {
            return Err(NisabaError::Invalid("Invalid path format".into()));
        }

        Ok(())
    }

    /// The `validate_host` function runs specific validations on host for network-nased backend
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_host(host: &str) -> Result<(), NisabaError> {
        if host.trim().is_empty() {
            return Err(NisabaError::Missing("Host cannot be empty".into()));
        }

        if host.contains(|c: char| c.is_whitespace() || c == '@' || c == '/') {
            return Err(NisabaError::Invalid(
                "Host contains invalid characters".into(),
            ));
        }

        Ok(())
    }
    /// The `validate_port` function runs specific validations on port for network-nased backend
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

    /// The `validate_database_name` function runs specific validations on database_name for network-nased backend
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_database_name(database: &str) -> Result<(), NisabaError> {
        if database.trim().is_empty() {
            return Err(NisabaError::Missing("Database name cannot be empty".into()));
        }

        if database.contains(|c: char| c.is_whitespace() || c == '/' || c == '\\') {
            return Err(NisabaError::Invalid(
                "Database name contains invalid characters".into(),
            ));
        }

        Ok(())
    }

    /// The `validate_namespace` function runs specific validations on host for PostgreSQL
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_namespace(namespace: &str) -> Result<(), NisabaError> {
        if namespace.contains(|c: char| c.is_whitespace() || c == '@' || c == '/' || c == '\\') {
            return Err(NisabaError::Invalid(
                "Namespace/schema contains invalid characters".into(),
            ));
        }
        Ok(())
    }

    /// The `validate_credentials` function runs specific validations on credentials for network-nased backend
    ///
    /// Returns:
    ///
    /// A Result of unit on success or NisabaError if validation fails.
    fn validate_credentials(username: &str, password: &str) -> Result<(), NisabaError> {
        if username.trim().is_empty() {
            return Err(NisabaError::Missing("Username cannot be empty".into()));
        }

        if password.trim().is_empty() {
            return Err(NisabaError::Missing("Password cannot be empty".into()));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_backend_creation() {
        let config = StorageConfig::new_file_backend(StorageBackend::Csv, "/data/csvs").unwrap();

        assert_eq!(config.dir_path, Some("/data/csvs".to_string()));
        assert!(config.connection_string().is_ok());
    }

    #[test]
    fn test_validation_fails_empty_path() {
        let config = StorageConfig::new_file_backend(StorageBackend::Csv, "");

        assert!(config.is_err())
    }

    #[test]
    fn test_network_backend_creation_without_namespace() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::PostgreSQL,
            "localhost",
            5432,
            "mydb",
            "user",
            "pass",
            None::<String>,
        )
        .unwrap();

        assert_eq!(config.host, Some("localhost".into()));
        assert!(config.connection_string().is_ok());
        assert!(config.namespace.is_none());
    }

    #[test]
    fn test_network_backend_creation_with_namespace() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::PostgreSQL,
            "localhost",
            5432,
            "mydb",
            "user",
            "pass",
            Some("private"),
        )
        .unwrap();

        assert_eq!(config.host, Some("localhost".into()));
        assert!(config.connection_string().is_ok());
        assert!(config.namespace.is_some());
    }

    #[test]
    fn test_network_backend_creation_with_namespace_mysql() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::MySQL,
            "localhost",
            5432,
            "mydb",
            "user",
            "pass",
            Some("private"),
        );

        assert!(config.is_err());
    }

    #[test]
    fn test_url_encoding() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::PostgreSQL,
            "localhost",
            5432,
            "mydb",
            "user@domain",
            "pass:word",
            Some("private"),
        )
        .unwrap();

        let conn_str = config.connection_string().unwrap();
        assert!(conn_str.contains("%40"));
        assert!(conn_str.contains("%3A"));
    }

    #[test]
    fn test_mongodb_srv_connection() {
        let config = StorageConfig::new_mongo_srv_backend(
            "cluster0.mongodb.net",
            "mydb",
            "username",
            "password",
        )
        .unwrap();

        let conn_str = config.connection_string().unwrap();

        assert!(conn_str.starts_with("mongodb+srv://"));
        assert_eq!(config.backend, StorageBackend::MongoDB);
    }

    #[test]
    fn test_mongodb_standard_connection() {
        let config = StorageConfig::new_network_backend(
            StorageBackend::MongoDB,
            "localhost",
            27017,
            "mydb",
            "username",
            "password",
            None::<String>,
        )
        .unwrap();

        let conn_str = config.connection_string().unwrap();

        assert!(conn_str.starts_with("mongodb://"));
        assert!(conn_str.contains(":27017"));
        assert!(!conn_str.starts_with("mongodb+srv://"));
    }
}
