use nisaba::{SchemaAnalyzerBuilder, StorageBackend, StorageConfig};

#[tokio::main]
async fn main() {
    // analyzer
    let analyzer = SchemaAnalyzerBuilder::default().build();

    let csv_config = StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap();

    let parquet_config =
        StorageConfig::new_file_backend(StorageBackend::Parquet, "./assets/parquet").unwrap();

    let _result = analyzer
        .analyze(vec![csv_config, parquet_config])
        .await
        .unwrap();
}
