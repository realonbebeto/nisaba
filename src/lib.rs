#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

mod analyzer;
mod error;
mod types;

pub use analyzer::{
    AnalyzerConfig, SchemaAnalyzerBuilder,
    catalog::{StorageBackend, StorageConfig},
    inference::{
        CsvInferenceEngine, ExcelInferenceEngine, MySQLInferenceEngine, NoSQLInferenceEngine,
        ParquetInferenceEngine, PostgreSQLInferenceEngine, SchemaInferenceEngine,
        SqliteInferenceEngine,
    },
    retriever::LatentStore,
};
