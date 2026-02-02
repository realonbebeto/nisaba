#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc = include_str!("../README.md")]

mod analyzer;
mod error;

mod types;

pub use analyzer::{
    AnalyzerConfig, SchemaAnalyzer,
    datastore::{FileStoreType, Source},
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

#[cfg(test)]
pub mod test {
    use std::sync::Arc;

    use tokio::sync::OnceCell;

    use super::*;

    static LATENT_STORE: OnceCell<Arc<LatentStore>> = OnceCell::const_new();

    pub async fn get_test_latent_store() -> Arc<LatentStore> {
        LATENT_STORE
            .get_or_init(|| async {
                Arc::new(
                    LatentStore::builder()
                        .analyzer_config(Arc::new(AnalyzerConfig::default()))
                        .build()
                        .await
                        .expect("Failed to initialize shared test LatentStore"),
                )
            })
            .await
            .clone()
    }

    #[tokio::test]
    async fn test_two_silo_probing() {
        let config = AnalyzerConfig::builder()
            .sample_size(10)
            .scoring(ScoringConfig::default())
            .similarity(SimilarityConfig::default())
            .build();

        let latent_store = get_test_latent_store().await;

        // analyzer
        let analyzer = SchemaAnalyzer::builder()
            .config(config)
            .name("nisaba1")
            .sources(vec![
                Source::files(FileStoreType::Csv)
                    .has_header(true)
                    .num_rows(10)
                    .path("./assets/csv")
                    .build()
                    .unwrap(),
                Source::files(FileStoreType::Parquet)
                    .num_rows(10)
                    .path("./assets/parquet")
                    .build()
                    .unwrap(),
            ])
            .build()
            .await
            .unwrap()
            .latent_store(latent_store);

        let result = analyzer.analyze().await.unwrap();

        assert!(result.is_some());
    }
}
