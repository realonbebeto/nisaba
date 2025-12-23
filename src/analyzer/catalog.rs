use crate::error::NError;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum DataStoreType {
    Avro,
    Csv,
    Excel,
    Feather,
    Json,
    JsonL,
    MongoDB,
    MySQL,
    Orc,
    Parquet,
    PostgreSQL,
    SQLite,
}

impl std::fmt::Display for DataStoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Avro => write!(f, "avro"),
            Self::Csv => write!(f, "csv"),
            Self::Excel => write!(f, "excel"),
            Self::Feather => write!(f, "feather"),
            Self::Json => write!(f, "json"),
            Self::JsonL => write!(f, "jsonl"),
            Self::MongoDB => write!(f, "mongodb"),
            Self::MySQL => write!(f, "mysql"),
            Self::Orc => write!(f, "orc"),
            Self::Parquet => write!(f, "parquet"),
            Self::PostgreSQL => write!(f, "postgresql"),
            Self::SQLite => write!(f, "sqlite"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataLocation {
    pub store_type: DataStoreType,
    pub dir_path: Option<String>,
    pub host: Option<String>,
    pub port: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub database: Option<String>,
    pub namespace: Option<String>,
}

impl DataLocation {
    pub fn connection_string(&self) -> Result<String, NError> {
        match self.store_type {
            DataStoreType::Csv => match &self.dir_path {
                Some(path) => Ok(format!(
                    "Driver={{Microsoft Text Driver (*.txt; *.csv)}}; DBQ={};Extensions=asc,csv,tab,txt;",
                    path
                )),
                None => Err(NError::InvalidPath(
                    "Directory with CSVs not provided".into(),
                )),
            },
            DataStoreType::Excel => match &self.dir_path {
                Some(path) => Ok(format!(
                    "Driver={{Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)}};DBQ={};",
                    path
                )),
                None => Err(NError::InvalidPath(
                    "Directory with Excel workbooks not provided".into(),
                )),
            },

            DataStoreType::MySQL => {
                match (
                    &self.host,
                    &self.port,
                    &self.database,
                    &self.username,
                    &self.password,
                ) {
                    (Some(host), Some(port), Some(database), Some(username), Some(password)) => {
                        Ok(format!(
                            "Driver={{MySQL ODBC 5.2 UNICODE Driver}};Server={};Port={};Database={};User={};Password={};Option=3;",
                            host, port, database, username, password
                        ))
                    }

                    _ => Err(NError::InvalidPath("Proper Postgres".into())),
                }
            }
            DataStoreType::MongoDB => {
                match (
                    &self.username,
                    &self.password,
                    &self.host,
                    &self.port,
                    &self.database,
                ) {
                    (Some(username), Some(password), Some(host), Some(port), Some(database)) => {
                        Ok(format!(
                            "mongodb://{}:{}@{}:{}/{}?authSource=admin",
                            username, password, host, port, database
                        ))
                    }
                    _ => Err(NError::InvalidPath(
                        "Proper MongoDB connection details not provided".into(),
                    )),
                }
            }
            DataStoreType::Parquet => match &self.dir_path {
                Some(path) => Ok(path.clone()),
                None => Err(NError::InvalidPath(
                    "Directory with Parquet files not provided".into(),
                )),
            },
            DataStoreType::PostgreSQL => match (
                &self.host,
                &self.port,
                &self.database,
                &self.username,
                &self.password,
            ) {
                (Some(host), Some(port), Some(database), Some(username), Some(password)) => {
                    Ok(format!(
                        "Driver={{PostgreSQL}};Server={};Port={};Database={};Uid={};Pwd={};",
                        host, port, database, username, password,
                    ))
                }
                _ => Err(NError::InvalidPath(
                    "Proper PostgreSQL connection details not provided".into(),
                )),
            },

            DataStoreType::SQLite => match &self.dir_path {
                Some(path) => Ok(format!(
                    "DRIVER=SQLite3 ODBC Driver;Database={};LongNames=0;Timeout=1000;NoTXN=0;SyncPragma=NORMAL;StepAPI=0;",
                    path
                )),
                None => Err(NError::InvalidPath(
                    "SQLite database path not provided".into(),
                )),
            },
            _ => Ok("".into()),
        }
    }
}
