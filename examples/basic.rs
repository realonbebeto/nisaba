use nisaba::{SchemaAnalyzerBuilder, StorageBackend, StorageConfig};

#[tokio::main]
async fn main() {
    let csv_config = StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap();

    let parquet_config =
        StorageConfig::new_file_backend(StorageBackend::Parquet, "./assets/parquet").unwrap();
    // analyzer
    let analyzer = SchemaAnalyzerBuilder::new()
        .with_storage_configs(vec![csv_config, parquet_config])
        .build();

    let _result = analyzer.analyze().await.unwrap();
}
