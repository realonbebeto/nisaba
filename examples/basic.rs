use nisaba::{
    AnalyzerConfig, DistanceType, EmbeddingModel, FileStoreType, SchemaAnalyzer, ScoringConfig,
    SimilarityConfig, Source,
};

#[tokio::main]
async fn main() {
    let config = AnalyzerConfig::builder()
        .sample_size(1000)
        .scoring(ScoringConfig {
            type_weight: 0.65,
            structure_weight: 0.35,
        })
        .similarity(SimilarityConfig {
            threshold: 0.59,
            top_k: Some(7),
            algorithm: DistanceType::Cosine,
        })
        .build();

    // analyzer
    let analyzer = SchemaAnalyzer::builder()
        .name("nisaba")
        .config(config)
        .embedding_model(EmbeddingModel::MultilingualE5Small)
        .source(
            Source::files(FileStoreType::Csv)
                .path("./assets/csv")
                .num_rows(1000)
                .has_header(true)
                .build()
                .unwrap(),
        )
        .sources(vec![
            Source::files(FileStoreType::Parquet)
                .path("./assets/parquet")
                .num_rows(1000)
                .build()
                .unwrap(),
        ])
        .build()
        .await
        .unwrap();

    let _result = analyzer.analyze().await.unwrap();
}
