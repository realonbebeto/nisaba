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
                .build()
                .unwrap(),
        )
        .sources(vec![
            Source::files(FileStoreType::Parquet)
                .path("./assets/parquet")
                .build()
                .unwrap(),
            Source::files(FileStoreType::Excel)
                .path("./assets/xlsx")
                .build()
                .unwrap(),
            Source::mongodb()
                .auth("mongodb", "mongodb")
                .host("localhost")
                .database("mongo_store")
                .pool_size(5)
                .port(27017)
                .build()
                .unwrap(),
            Source::mysql()
                .auth("mysql", "mysql")
                .host("localhost")
                .port(3306)
                .database("mysql_store")
                .build()
                .await
                .unwrap(),
            Source::postgresql()
                .auth("postgres", "postgres")
                .host("localhost")
                .database("postgres")
                .port(5432)
                .namespace("public")
                .build()
                .await
                .unwrap(),
            Source::sqlite()
                .path("./assets/sqlite/nisaba.sqlite")
                .build()
                .await
                .unwrap(),
        ])
        .build()
        .await
        .unwrap();

    let _result = analyzer.analyze().await.unwrap();
}
