#[derive(Debug, thiserror::Error)]
pub enum NisabaError {
    // Schema & Type
    #[error("Unsupported: {0}")]
    Unsupported(String),
    #[error("No TableDef Generated")]
    NoTableDefGenerated,
    #[error("No RecordBatch")]
    NoRecordBatch,

    // Resources
    #[error("Invalid: {0}")]
    Invalid(String),
    #[error("Missing: {0}")]
    Missing(String),

    // External errors
    #[error(transparent)]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error(transparent)]
    MongoDB(#[from] mongodb::error::Error),
    #[error(transparent)]
    LanceDB(#[from] lancedb::error::Error),
    #[error(transparent)]
    Sqlx(#[from] sqlx::Error),
    #[error(transparent)]
    Uuid(#[from] uuid::Error),
    #[error(transparent)]
    Ods(#[from] calamine::OdsError),
    #[error(transparent)]
    Xls(#[from] calamine::XlsError),
    #[error(transparent)]
    Xlsb(#[from] calamine::XlsbError),
    #[error(transparent)]
    Xlsx(#[from] calamine::XlsxError),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("error: {0}")]
    Graph(graphrs::Error),
    #[error(transparent)]
    ThreadPoolBuilder(#[from] rayon::ThreadPoolBuildError),
}
