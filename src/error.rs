#[derive(Debug, thiserror::Error)]
pub enum NError {
    #[error("Unknown Type")]
    UnknownType,
    #[error("Unsupported: {0}")]
    Unsupported(String),
    #[error("{0}")]
    SchemaError(String),
    #[error(transparent)]
    UnexpectedError(#[from] anyhow::Error),
    #[error(transparent)]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error(transparent)]
    RegexError(#[from] regex::Error),
    #[error("{0}")]
    FileError(String),
    #[error(transparent)]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[error("Unable to access dir/path provided: {0}")]
    InvalidDetail(String),
    #[error("Path/Dir provided misses files required: {0}")]
    EmptyResource(String),
    #[error("{0}")]
    MissingDetail(String),
    #[error(transparent)]
    MongoDBError(#[from] mongodb::error::Error),
    #[error(transparent)]
    LanceDBError(#[from] lancedb::error::Error),
    #[error(transparent)]
    UuidError(#[from] uuid::Error),
    #[error(transparent)]
    ArrowGraphError(#[from] GraphError),
    #[error(transparent)]
    OdsError(#[from] calamine::OdsError),
    #[error(transparent)]
    XlsError(#[from] calamine::XlsError),
    #[error(transparent)]
    XlsbError(#[from] calamine::XlsbError),
    #[error(transparent)]
    XlsxError(#[from] calamine::XlsxError),
    #[error(transparent)]
    SqlxError(#[from] sqlx::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Graph construction error: {0}")]
    GraphConstruction(String),

    #[error("Algorithm error: {0}")]
    Algorithm(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found: source={0}, target={1}")]
    EdgeNotFound(String, String),

    #[error("Graph is empty")]
    EmptyGraph,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Computation error: {0}")]
    Computation(String),
}
