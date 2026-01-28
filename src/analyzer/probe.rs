use std::{collections::HashSet, sync::Arc};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    CsvInferenceEngine, ExcelInferenceEngine, MySQLInferenceEngine, NoSQLInferenceEngine,
    ParquetInferenceEngine, PostgreSQLInferenceEngine, SqliteInferenceEngine, StorageBackend,
    analyzer::{
        calculation::GraphClusterer,
        catalog::StorageConfig,
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
    pub type_weight: f32,
    pub sample_weight: f32,
    pub name_weight: f32,
    pub structure_weight: f32,
    pub similarity_threshold: f32,
    pub top_k: Option<usize>,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        AnalyzerConfig {
            type_weight: 0.50,
            sample_weight: 0.0,
            name_weight: 0.05,
            structure_weight: 0.45,
            similarity_threshold: 0.50,
            top_k: None,
        }
    }
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
    pub(crate) name: String,
    pub(crate) config: Arc<AnalyzerConfig>,
    pub(crate) latent_store: Arc<LatentStore>,
    pub(crate) storage_configs: Vec<StorageConfig>,
    pub(crate) infer_stats: Arc<Mutex<InferenceStats>>,
}
impl std::fmt::Debug for SchemaAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Analyzer")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish()
    }
}

impl SchemaAnalyzer {
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

            clusterer.add_ann_edges(self.config.clone(), tbl_rep.id, &ids)?;
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

                clusterer.add_ann_edges(self.config.clone(), *field_def, &ids)?;

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
        let table_handler = self.latent_store.table_handler::<T>(self.config.clone());

        table_handler
            .search_table_rep(query_schema, T::result_columns())
            .await
    }

    async fn find_candidates_field<T: Storable>(
        &self,
        query_schema: Uuid,
    ) -> Result<Vec<FieldMatch>, NisabaError> {
        let table_handler = self.latent_store.table_handler::<T>(self.config.clone());

        table_handler
            .search_field_def(query_schema, T::result_columns())
            .await
    }

    async fn discover_ecosystem(&self) -> Result<Vec<TableRep>, NisabaError> {
        let mut table_reps = Vec::new();

        let table_handler = self
            .latent_store
            .table_handler::<TableRep>(self.config.clone());

        table_handler.initialize().await?;

        let field_handler = self
            .latent_store
            .table_handler::<FieldDef>(self.config.clone());

        field_handler.initialize().await?;

        for config in &self.storage_configs {
            let reps = match config.backend {
                StorageBackend::Csv => {
                    let csv_inferer = CsvInferenceEngine::new(None, None);
                    csv_inferer.csv_store_infer(
                        config,
                        self.infer_stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                StorageBackend::Excel => {
                    let excel_inferer = ExcelInferenceEngine::new(None, None);
                    excel_inferer.excel_store_infer(
                        config,
                        self.infer_stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                StorageBackend::MongoDB => {
                    let mongo_inferer = NoSQLInferenceEngine::new(None);
                    mongo_inferer
                        .mongodb_store_infer(
                            config,
                            self.infer_stats.clone(),
                            Arc::new(|table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            }),
                        )
                        .await
                }

                StorageBackend::MySQL => {
                    let mysql_inferer = MySQLInferenceEngine::new(None);

                    mysql_inferer
                        .mysql_store_infer(config, self.infer_stats.clone(), |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        })
                        .await
                }

                StorageBackend::Parquet => {
                    let parquet_inferer = ParquetInferenceEngine::new(None);

                    parquet_inferer.parquet_store_infer(
                        config,
                        self.infer_stats.clone(),
                        4,
                        |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        },
                    )
                }

                StorageBackend::PostgreSQL => {
                    let postgres_inferer = PostgreSQLInferenceEngine::new(None);

                    postgres_inferer
                        .postgres_store_infer(
                            config,
                            self.infer_stats.clone(),
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                StorageBackend::SQLite => {
                    let sqlite_inferer = SqliteInferenceEngine::new(None);

                    sqlite_inferer
                        .sqlite_store_infer(config, self.infer_stats.clone(), |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        })
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
    pub tables_processed: usize,
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
    pub(crate) name: Option<String>,
    pub(crate) config: Option<AnalyzerConfig>,
    pub(crate) latent_store: Option<LatentStore>,
    pub(crate) storage_configs: Option<Vec<StorageConfig>>,
}

impl std::fmt::Debug for SchemaAnalyzerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyzerBuilder")
            .field("name", &self.name)
            .field("config", &self.config)
            .field("storage_configs", &self.storage_configs)
            .finish()
    }
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
    pub fn with_name(mut self, name: Option<&str>) -> Self {
        self.name = name.map(|n| n.to_string());
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
    pub fn with_config(mut self, config: Option<AnalyzerConfig>) -> Self {
        self.config = config;
        self
    }

    /// The `with_latent_store` function sets the latent store and returns the
    /// modified object.
    ///
    /// Arguments:
    ///
    /// * `latent_store`: The `latent_store` parameter is of type `LatentStore`. It
    ///   is used to set the vector store by passing an instance of `LatentStore` to
    ///   the function.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after updating the vector store with the provided instance
    pub fn with_latent_store(mut self, latent_store: LatentStore) -> Self {
        self.latent_store = Some(latent_store);
        self
    }

    /// The `with_latent_store` function sets the latent store and returns the
    /// modified object.
    ///
    /// Arguments:
    ///
    /// * `storage_config`: The `storage_configs` parameter is of type Vec of `StorageConfig`. It
    ///   is used to set the storage configurations by passing a vector of instances of `StorageConfig` to
    ///   the function.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after updating the vector store with the provided instance
    pub fn with_storage_configs(mut self, storage_configs: Vec<StorageConfig>) -> Self {
        self.storage_configs = Some(storage_configs);
        self
    }

    /// The function `build` constructs a `SchemaAnalyzer` instance with specified weights and
    /// properties. The weights provided MUST add to 1.0.
    ///
    /// Returns:
    ///
    /// A `SchemaAnalyzer` instance is being returned from the `build` function.
    pub fn build(self) -> SchemaAnalyzer {
        let mut config = self.config.unwrap_or_default();
        assert!(
            config.type_weight
                + config.sample_weight
                + config.name_weight
                + config.structure_weight
                == 1.0,
            "The summation of type_weight, sample weight, name_weight and structure_weight should be 1"
        );

        match config.top_k {
            Some(v) => {
                config.top_k = Some(v);
            }
            None => config.top_k = Some(self.storage_configs.as_ref().unwrap().len()),
        }

        SchemaAnalyzer {
            name: self.name.unwrap_or("default".to_string()),
            config: Arc::new(config),
            infer_stats: Arc::new(Mutex::new(InferenceStats::default())),
            latent_store: Arc::new(self.latent_store.unwrap()),
            storage_configs: self.storage_configs.unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::StorageBackend;

    use super::*;

    #[tokio::test]
    async fn test_two_silo_probing() {
        let ls = LatentStore::new(None, None).await.unwrap();
        let storage_configs = vec![
            StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap(),
            StorageConfig::new_file_backend(StorageBackend::Parquet, "./assets/parquet").unwrap(),
        ];

        // analyzer
        let builder = SchemaAnalyzerBuilder::new();
        let analyzer = builder
            .with_latent_store(ls)
            .with_storage_configs(storage_configs)
            .build();

        let result = analyzer.analyze().await.unwrap();

        assert!(result.is_some());
    }
}
