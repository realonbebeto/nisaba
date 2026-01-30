#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

mod analyzer;
mod error;
mod types;

pub use analyzer::{
    AnalyzerConfig, SchemaAnalyzer,
    datastore::Source,
    inference::{
        CsvInferenceEngine, ExcelInferenceEngine, MySQLInferenceEngine, NoSQLInferenceEngine,
        ParquetInferenceEngine, PostgreSQLInferenceEngine, SchemaInferenceEngine,
        SqliteInferenceEngine,
    },
    probe::{ScoringConfig, SimilarityConfig},
    retriever::LatentStore,
};

pub use fastembed::EmbeddingModel;
pub use lancedb::DistanceType;
