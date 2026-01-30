use fastembed::EmbeddingModel;
use lancedb::DistanceType;
use std::{collections::HashSet, sync::Arc};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    CsvInferenceEngine, ExcelInferenceEngine, MySQLInferenceEngine, NoSQLInferenceEngine,
    ParquetInferenceEngine, PostgreSQLInferenceEngine, SqliteInferenceEngine,
    analyzer::{
        calculation::GraphClusterer,
        datastore::{DatabaseType, FileStoreType, Source, SourceType},
        report::{FieldCluster, FieldMatch, FieldResult, TableCluster, TableMatch, TableResult},
        retriever::{LatentStore, Storable},
    },
    error::NisabaError,
    types::{FieldDef, TableRep},
};

#[derive(Debug, Clone)]
/// The code defines a struct `AnalyzerConfig` with various fields representing weights and thresholds
/// for an analyzer configuration.
///
/// Properties:
///
/// * `type_weight`: The `type_weight` property represents the weight assigned to the type of an
///   element when analyzing its importance or relevance.
///
/// * `sample_weight`: The `sample_weight` property represents the weight assigned to the samples when
///   analyzing data. This weight is a floating-point value (f32) that influences the importance of
///   samples in the analysis process.
///
/// * `structure_weight`: The `structure_weight` property represents the weight assigned to the
///   structure of the data when performing analysis. This weight determines how important the structure
///   of the data is compared to other factors in the analysis process.
///
/// * `similarity_threshold`: The `similarity_threshold` property represents the threshold value used
///   to determine the similarity between items during analysis.
///
/// * `top_k`: The `top_k` property in the `AnalyzerConfig` struct is an optional field that specifies
///   the maximum number of results to return. It is of type `Option<usize>`, which means it can either
///   contain a `usize` value or be `None`. This allows for flexibility in the configuration,
pub struct AnalyzerConfig {
    pub scoring: ScoringConfig,
    pub sample_size: Option<usize>,
    pub similarity: SimilarityConfig,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        AnalyzerConfig {
            scoring: ScoringConfig::default(),
            sample_size: Some(1000),
            similarity: SimilarityConfig::default(),
        }
    }
}

impl AnalyzerConfig {
    pub fn builder() -> AnalyzerConfigBuilder {
        AnalyzerConfigBuilder {
            scoring: None,
            sample_size: None,
            similarity: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScoringConfig {
    pub type_weight: f32,
    pub structure_weight: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            type_weight: 0.65,
            structure_weight: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimilarityConfig {
    pub threshold: f32,
    pub top_k: Option<usize>,
    pub algorithm: DistanceType,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            threshold: 0.55,
            top_k: None,
            algorithm: DistanceType::Cosine,
        }
    }
}

pub struct AnalyzerConfigBuilder {
    scoring: Option<ScoringConfig>,
    sample_size: Option<usize>,
    similarity: Option<SimilarityConfig>,
}

impl AnalyzerConfigBuilder {
    pub fn scoring(mut self, scoring: ScoringConfig) -> Self {
        self.scoring = Some(scoring);
        self
    }

    pub fn sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = Some(sample_size);
        self
    }

    pub fn similarity(mut self, similarity: SimilarityConfig) -> Self {
        self.similarity = Some(similarity);
        self
    }

    pub fn build(self) -> AnalyzerConfig {
        AnalyzerConfig {
            scoring: self.scoring.unwrap_or_default(),
            sample_size: self.sample_size,
            similarity: self.similarity.unwrap_or_default(),
        }
    }
}

pub struct InferenceContext {
    pub config: Arc<AnalyzerConfig>,
    pub persistence: Arc<LatentStore>,
    pub stats: Arc<Mutex<InferenceStats>>,
}

/// The `SchemaAnalyzer` provides an interface for store reconciliation. It contains fields for name,
/// configuration, inference engine registry, and latent store.
///
/// Properties:
///
/// * `name`: The `name` property is a `String` that represents the name of the schema analyzer.
///
/// * `config`: The `config` property holds a reference-counted smart pointer to an `AnalyzerConfig`
///   instance, allowing shared ownership of the `AnalyzerConfig` data across multiple parts of the program.
///
/// * `inference_engine`: The `inference_engine` property in the `SchemaAnalyzer` struct is of type
///   `InferenceEngineRegistry`. It is used to store and manage inference engines that are responsible for
///   analyzing and inferring information from the schema data.
///
/// * `latent_store`: The `latent_store` property in the `SchemaAnalyzer` struct is of type
///   `Arc<LatentStore>`. It is an atomic reference-counted smart pointer that allows shared ownership of
///   the `LatentStore` instance. This means that multiple parts of the code can have access to the embedding store`
pub struct SchemaAnalyzer {
    name: String,
    sources: Vec<Source>,
    context: InferenceContext,
}
impl std::fmt::Debug for SchemaAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analyzer")
            .field("name", &self.name)
            .field("config", &self.context.config)
            .field("stats", &self.context.stats)
            .finish()
    }
}

impl SchemaAnalyzer {
    pub fn builder() -> SchemaAnalyzerBuilder {
        SchemaAnalyzerBuilder::default()
    }
    /// The `analyze` function runs agains the beforehand provided storage location as StorageConfigs, infers their
    /// schemas, clusters them based on similarities.
    ///
    /// Arguments:
    ///
    /// Returns:
    ///
    /// The `analyze` function returns a `Result` containing either an `Option` of a vector of
    /// `TableCluster` objects or a `NisabaError`. If the clustering process is successful and at least
    /// one cluster is formed, it returns `Ok(Some(clusters))` with the clustered tables. If no clusters
    /// are formed, it returns `Ok(None)`.
    pub async fn analyze(&self) -> Result<Option<Vec<TableCluster>>, NisabaError> {
        let table_reps = self.discover_ecosystem().await?;

        let total_tables = table_reps.len();

        // Match Collection/Tables
        let mut clusterer = GraphClusterer::new();

        for tbl_rep in &table_reps {
            let candidates = self.find_candidates_table::<TableRep>(tbl_rep.id).await?;

            let ids: Vec<(Uuid, f32)> = candidates
                .iter()
                .map(|v| (v.schema.id, v.confidence))
                .collect();

            clusterer.add_ann_edges(self.context.config.clone(), tbl_rep.id, &ids)?;
        }

        let clusters = clusterer.clusters()?;

        let mut clusters = self.cluster_tables(clusters, table_reps.clone())?;

        // Successful clustering
        if clusters.len() < total_tables {
            // Match Field Names Inside matched Collection/Tables
            self.cluster_fields(&mut clusters, &table_reps).await?;

            return Ok(Some(clusters));
        }

        // TODO: Optional CleanUp - Delete the lancedb

        Ok(None)
    }

    fn cluster_tables(
        &self,
        clusters: Vec<HashSet<Uuid>>,
        table_reps: Vec<TableRep>,
    ) -> Result<Vec<TableCluster>, NisabaError> {
        let communities: Vec<TableCluster> = clusters
            .into_iter()
            .enumerate()
            .map(|(cluster_id, vals)| {
                let items = table_reps
                    .iter()
                    .filter(|d| vals.contains(&d.id))
                    .cloned()
                    .collect::<Vec<TableRep>>();

                let tables = items
                    .into_iter()
                    .map(|it| TableResult {
                        id: it.id,
                        silo_id: it.silo_id,
                        table_name: it.name,
                    })
                    .collect::<Vec<TableResult>>();

                TableCluster {
                    cluster_id: cluster_id as u32,
                    tables,
                    field_clusters: Vec::new(),
                }
            })
            .collect();

        Ok(communities)
    }

    /// The `cluster_fields` function asynchronously clusters fields based on table definitions
    /// and stores the results in a vector.
    ///
    /// Arguments:
    ///
    /// * `clusters`: The `clusters` parameter is a mutable slice of `TableCluster` structs. It
    ///   represents a collection of clusters, each containing information about tables and their field
    ///   clusters.
    ///
    /// * `table_defs`: The `table_defs` parameter is a slice of `TableDef` structs. Each `TableDef` struct
    ///   likely contains information about a table, including its ID and a vector of field definitions
    ///   (`FieldDef`).
    ///
    /// Returns:
    ///
    /// The `cluster_fields` function is returning a `Result<(), NisabaError>`.
    async fn cluster_fields(
        &self,
        clusters: &mut [TableCluster],
        table_reps: &[TableRep],
    ) -> Result<(), NisabaError> {
        for tbl_cluster in clusters {
            let mut cached_candidates = HashSet::new();
            let table_ids: Vec<Uuid> = tbl_cluster.tables.iter().map(|v| v.id).collect();

            let field_defs = table_reps
                .iter()
                .filter(|v| table_ids.contains(&v.id))
                .map(|v| v.fields.clone())
                .collect::<Vec<Vec<Uuid>>>()
                .concat();

            // Build graph
            let mut clusterer = GraphClusterer::new();

            for field_def in &field_defs {
                let candidates = self.find_candidates_field::<FieldDef>(*field_def).await?;

                let ids: Vec<(Uuid, f32)> = candidates
                    .iter()
                    .map(|v| (v.schema.id, v.confidence))
                    .collect();

                clusterer.add_ann_edges(self.context.config.clone(), *field_def, &ids)?;

                cached_candidates.extend(candidates.into_iter().map(|c| c.schema));
            }

            let clusters = clusterer.clusters()?;

            let communities: Vec<FieldCluster> = clusters
                .into_iter()
                .enumerate()
                .map(|(cluster_id, vals)| {
                    let items = cached_candidates
                        .iter()
                        .filter(|d| vals.contains(&d.id))
                        .collect::<Vec<&FieldDef>>();

                    let fields = items
                        .into_iter()
                        .map(|it| FieldResult {
                            id: it.id,
                            silo_id: it.silo_id.clone(),
                            table_name: it.table_name.clone(),
                            field_name: it.name.clone(),
                        })
                        .collect::<Vec<FieldResult>>();

                    FieldCluster {
                        cluster_id: cluster_id as u32,
                        fields,
                    }
                })
                .collect();

            tbl_cluster.field_clusters = communities;

            // table_handler.clear_table().await?;
        }

        Ok(())
    }

    async fn find_candidates_table<T: Storable>(
        &self,
        query_schema: Uuid,
    ) -> Result<Vec<TableMatch>, NisabaError> {
        let table_handler = self.context.persistence.table_handler::<T>();

        table_handler
            .search_table_rep(query_schema, T::result_columns())
            .await
    }

    async fn find_candidates_field<T: Storable>(
        &self,
        query_schema: Uuid,
    ) -> Result<Vec<FieldMatch>, NisabaError> {
        let table_handler = self.context.persistence.table_handler::<T>();

        table_handler
            .search_field_def(query_schema, T::result_columns())
            .await
    }

    async fn discover_ecosystem(&self) -> Result<Vec<TableRep>, NisabaError> {
        let mut table_reps = Vec::new();

        let table_handler = self.context.persistence.table_handler::<TableRep>();

        table_handler.initialize().await?;

        let field_handler = self.context.persistence.table_handler::<FieldDef>();

        field_handler.initialize().await?;

        for source in &self.sources {
            let reps = match source.metadata.source_type {
                SourceType::FileStore(FileStoreType::Csv) => {
                    let csv_inferer = CsvInferenceEngine::new(None, None);
                    csv_inferer.csv_store_infer(
                        source,
                        self.context.stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                SourceType::FileStore(FileStoreType::Excel) => {
                    let excel_inferer = ExcelInferenceEngine::new(None, None);
                    excel_inferer.excel_store_infer(
                        source,
                        self.context.stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                SourceType::Database(DatabaseType::MongoDB) => {
                    let mongo_inferer = NoSQLInferenceEngine::new(None);
                    mongo_inferer
                        .mongodb_store_infer(
                            source,
                            self.context.stats.clone(),
                            Arc::new(|table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            }),
                        )
                        .await
                }

                SourceType::Database(DatabaseType::MySQL) => {
                    let mysql_inferer = MySQLInferenceEngine::new(None);

                    mysql_inferer
                        .mysql_store_infer(source, self.context.stats.clone(), |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        })
                        .await
                }

                SourceType::FileStore(FileStoreType::Parquet) => {
                    let parquet_inferer = ParquetInferenceEngine::new(None);

                    parquet_inferer.parquet_store_infer(
                        source,
                        self.context.stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                SourceType::Database(DatabaseType::PostgreSQL) => {
                    let postgres_inferer = PostgreSQLInferenceEngine::new(None);

                    postgres_inferer
                        .postgres_store_infer(
                            source,
                            self.context.stats.clone(),
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                SourceType::Database(DatabaseType::SQLite) => {
                    let sqlite_inferer = SqliteInferenceEngine::new(None);

                    sqlite_inferer
                        .sqlite_store_infer(
                            source,
                            self.context.stats.clone(),
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }
            };

            table_reps.extend(reps?);
        }

        table_handler.create_index().await?;

        field_handler.create_index().await?;

        Ok(table_reps)
    }
}

#[derive(Debug, Default)]
pub struct InferenceStats {
    pub sources_analyzed: usize,
    pub tables_found: usize,
    pub tables_inferred: usize,
    pub fields_inferred: usize,
    pub errors: Vec<String>,
}

/// The `SchemaAnalyzerBuilder` helps build the schema analyzer. It contains fields for configuration related to schema
/// analysis.
///
/// Properties:
///
/// * `name`: The `name` property is used to store the name of the schema analyzer being built.
///   It is a `String` type and is marked as `pub(crate)` which means it is accessible within the same crate.
///
/// * `config`: The `config` property holds configuration settings or parameters for the schema analyzer.
///   These settings could include things like thresholds, rules, or options that affect how the schema analysis
///   is performed.
///
/// * `latent_store`: The `latent_store` property is of type `Arc<LatentStore>`. It is used to store and
///   manage latent data within the schema analyzer.

#[derive(Default)]
pub struct SchemaAnalyzerBuilder {
    // fields for configuration
    name: Option<String>,
    sources: Vec<Source>,
    config: Option<AnalyzerConfig>,
    persist_path: Option<String>,
    embedding_model: Option<EmbeddingModel>,
}

impl SchemaAnalyzerBuilder {
    pub fn new() -> Self {
        SchemaAnalyzerBuilder::default()
    }
    /// The `with_name` function takes a mutable reference to a struct and a string, sets the
    /// struct's name field to the string value, and returns the modified struct.
    ///
    /// Arguments:
    ///
    /// * `name`: The `name` parameter is a reference to a string (`&str`) that represents the name you
    ///   want to assign to the object.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after setting the `name` field to the provided `name`
    /// string.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn source(mut self, source: Source) -> Self {
        self.sources.push(source);
        self
    }

    pub fn sources(mut self, sources: impl IntoIterator<Item = Source>) -> Self {
        self.sources.extend(sources);
        self
    }

    pub fn persist_path(mut self, path: impl Into<String>) -> Self {
        self.persist_path = Some(path.into());
        self
    }

    pub fn embedding_model(mut self, model: EmbeddingModel) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// The `with_config` function sets the configuration for an analyzer and returns the
    /// modified object.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter in the `with_config` function is of type `AnalyzerConfig`. It
    ///   is used to set the configuration for the analyzer by passing an instance of `AnalyzerConfig` to
    ///   the function.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after updating the configuration with the provided
    /// `AnalyzerConfig` and converting it to an `Arc`.
    pub fn config(mut self, config: AnalyzerConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// The function `build` constructs a `SchemaAnalyzer` instance with specified weights and
    /// properties. The weights provided MUST add to 1.0.
    ///
    /// Returns:
    ///
    /// A `SchemaAnalyzer` instance is being returned from the `build` function.
    pub async fn build(self) -> Result<SchemaAnalyzer, NisabaError> {
        if self.sources.len() < 2 {
            return Err(NisabaError::Missing(
                "Not Enough Sources: At least 2 sources should be provided".into(),
            ));
        }

        let mut config = self.config.unwrap_or_default();

        assert!(
            config.scoring.type_weight + config.scoring.structure_weight == 1.0,
            "The summation of type_weight, sample weight, name_weight and structure_weight should be 1"
        );

        match config.similarity.top_k {
            Some(v) => {
                config.similarity.top_k = Some(v);
            }
            None => config.similarity.top_k = Some(self.sources.len()),
        }

        let config = Arc::new(config);

        let persistence = LatentStore::builder()
            .analyzer_config(config.clone())
            .connection_path(self.persist_path)
            .embedding_model(self.embedding_model)
            .build()
            .await?;

        let context = InferenceContext {
            config: config.clone(),
            persistence: Arc::new(persistence),
            stats: Arc::new(Mutex::new(InferenceStats::default())),
        };

        Ok(SchemaAnalyzer {
            name: self.name.unwrap_or("default".to_string()),
            context,
            sources: self.sources,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_two_silo_probing() {
        let config = AnalyzerConfig::builder()
            .sample_size(1000)
            .scoring(ScoringConfig::default())
            .similarity(SimilarityConfig::default())
            .build();

        // analyzer
        let analyzer = SchemaAnalyzer::builder()
            .config(config)
            .name("nisaba1")
            .sources(vec![
                Source::csv("./assets/csv", None).unwrap(),
                Source::parquet("./assets/parquet", None).unwrap(),
            ])
            .build()
            .await
            .unwrap();

        let result = analyzer.analyze().await.unwrap();

        assert!(result.is_some());
    }
}
