use futures::executor::block_on;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    analyzer::{
        calculation::GraphClusterer,
        catalog::StorageConfig,
        inference::InferenceEngineRegistry,
        report::{FieldCluster, FieldMatch, TableCluster, TableMatch},
        retriever::{LatentStore, Storable},
    },
    error::NisabaError,
    types::{FieldDef, TableDef},
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
    pub(crate) inference_engine: InferenceEngineRegistry,
    pub(crate) latent_store: Arc<LatentStore>,
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
    /// The `analyze` function takes multiple collections/tables as inputs, infers their
    /// schemas, clusters them based on similarities.
    ///
    /// Arguments:
    ///
    /// * `configs`: The `configs` parameter in the `analyze` function is a vector of `StorageConfig`
    ///   structs. These `StorageConfig` structs contains configuration information related to
    ///   storage settings, such as database connection details.
    ///
    /// Returns:
    ///
    /// The `analyze` function returns a `Result` containing either an `Option` of a vector of
    /// `TableCluster` objects or a `NisabaError`. If the clustering process is successful and at least
    /// one cluster is formed, it returns `Ok(Some(clusters))` with the clustered tables. If no clusters
    /// are formed, it returns `Ok(None)`.
    pub async fn analyze(
        &self,
        configs: Vec<StorageConfig>,
    ) -> Result<Option<Vec<TableCluster>>, NisabaError> {
        // Take inputs (multiple collections/tables)
        // E.g. Vec[Collection/Table], Vec[Collection/Table], Vec[Collection/Table] ...

        let tbl_defs = self.inference_engine.discover_ecosystem(configs)?;
        self.index_schemas(&tbl_defs).await?;

        // Match Collection/Tables
        let mut clusterer = GraphClusterer::new();

        for tbl_def in &tbl_defs {
            let candidates = self.find_candidates(tbl_def).await?;
            let candidates = candidates
                .into_iter()
                .filter(|c| c.schema.id != tbl_def.id)
                .collect::<Vec<TableMatch>>();

            clusterer.add_ann_edges(self.config.clone(), tbl_def, &candidates)?;
        }

        let mut clusters = clusterer.clusters(&tbl_defs, |cluster_id, tr| TableCluster {
            cluster_id,
            tables: tr,
            field_clusters: Vec::new(),
        })?;
        // Successful clustering
        if clusters.len() < tbl_defs.len() {
            // Match Field Names Inside matched Collection/Tables
            self.cluster_fields(&mut clusters, &tbl_defs).await?;

            return Ok(Some(clusters));
        }

        // TODO: Optional CleanUp - Delete the lancedb

        Ok(None)
    }

    /// The `index_schemas` function asynchronously initializes a table handler, stores schema
    /// items, and returns a result indicating success or an error.
    ///
    /// Arguments:
    ///
    /// * `schema_items`: `schema_items` is a slice of `TableDef` items that are passed as a parameter
    ///   to the `index_schemas` function.
    ///
    /// Returns:
    ///
    /// The `index_schemas` function is returning a `Result<(), NisabaError>`.
    async fn index_schemas(&self, schema_items: &[TableDef]) -> Result<(), NisabaError> {
        let table_handler = self
            .latent_store
            .table_handler::<TableDef>(self.config.clone());

        table_handler.initialize().await?;

        table_handler.store(schema_items).await?;

        Ok(())
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
        table_defs: &[TableDef],
    ) -> Result<(), NisabaError> {
        for tbl_cluster in clusters {
            let table_ids: Vec<Uuid> = tbl_cluster.tables.iter().map(|v| v.id).collect();

            let field_defs = table_defs
                .iter()
                .filter(|v| table_ids.contains(&v.id))
                .map(|v| v.fields.clone())
                .collect::<Vec<Vec<FieldDef>>>()
                .concat();

            // Initialize vector store
            let table_handler = self
                .latent_store
                .table_handler::<FieldDef>(self.config.clone());

            table_handler.initialize().await?;

            table_handler.store(&field_defs).await?;

            // Build graph
            let mut clusterer = GraphClusterer::new();

            for field_def in &field_defs {
                let candidates = self.find_candidates(field_def).await?;
                let candidates = candidates
                    .into_iter()
                    .filter(|c| {
                        (c.schema.id != field_def.id)
                            && (c.schema.table_name != field_def.table_name)
                    })
                    .collect::<Vec<FieldMatch>>();

                clusterer.add_ann_edges(self.config.clone(), field_def, &candidates)?;
            }

            let clusters = clusterer.clusters(&field_defs, |cluster_id, fr| FieldCluster {
                cluster_id,
                fields: fr,
            })?;

            tbl_cluster.field_clusters = clusters;

            // table_handler.clear_table().await?;
        }

        Ok(())
    }

    async fn find_candidates<T: Storable>(
        &self,
        query_schema: &T,
    ) -> Result<Vec<T::SearchResult>, NisabaError> {
        let table_handler = self.latent_store.table_handler::<T>(self.config.clone());

        let candidates = table_handler
            .search(query_schema, self.config.clone(), T::result_columns())
            .await?;

        Ok(candidates)
    }
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
/// * `inference_registry`: The `inference_registry` property is of type `InferenceEngineRegistry`. It stores and
///   manages inference engines that are responsible for inferring schema information.
///
///
/// * `latent_store`: The `latent_store` property is of type `Arc<LatentStore>`. It is used to store and
///   manage latent data within the schema analyzer.
pub struct SchemaAnalyzerBuilder {
    // fields for configuration
    pub(crate) name: String,
    pub(crate) config: AnalyzerConfig,
    pub(crate) inference_registry: InferenceEngineRegistry,
    pub(crate) latent_store: LatentStore,
}

impl std::fmt::Debug for SchemaAnalyzerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyzerBuilder")
            .field("name", &self.name)
            .field("config", &self.config)
            .field("inference_registry", &self.inference_registry)
            .finish()
    }
}

impl Default for SchemaAnalyzerBuilder {
    fn default() -> Self {
        let ls = block_on(LatentStore::new(None));
        SchemaAnalyzerBuilder {
            name: "default".to_string(),
            config: AnalyzerConfig::default(),
            inference_registry: InferenceEngineRegistry::new(),
            latent_store: ls,
        }
    }
}

impl SchemaAnalyzerBuilder {
    pub fn new() -> Self {
        Self::default()
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
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
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
    pub fn with_config(mut self, config: AnalyzerConfig) -> Self {
        self.config = config;
        self
    }

    /// The `with_inference_registry` function sets the inference registry store and returns the
    /// modified object.
    ///
    /// Arguments:
    ///
    /// * `registry`: The `registry` parameter is of type `InferenceEngineRegistry`. It
    ///   is used to set the store for the inference engines by passing an instance of `InferenceEngineRegistry` to
    ///   the function.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after updating the store with the provided instance
    pub fn with_inference_registry(mut self, registry: InferenceEngineRegistry) -> Self {
        self.inference_registry = registry;
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
        self.latent_store = latent_store;
        self
    }

    /// The function `build` constructs a `SchemaAnalyzer` instance with specified weights and
    /// properties. The weights provided MUST add to 1.0.
    ///
    /// Returns:
    ///
    /// A `SchemaAnalyzer` instance is being returned from the `build` function.
    pub fn build(self) -> SchemaAnalyzer {
        assert!(
            self.config.type_weight
                + self.config.sample_weight
                + self.config.name_weight
                + self.config.structure_weight
                == 1.0,
            "The summation of type_weight, sample weight, name_weight and structure_weight should be 1"
        );

        let mut config = self.config;

        match config.top_k {
            Some(v) => {
                config.top_k = Some(v);
            }
            None => config.top_k = Some(self.inference_registry.size()),
        }

        SchemaAnalyzer {
            name: self.name,
            config: Arc::new(config),
            inference_engine: self.inference_registry,
            latent_store: Arc::new(self.latent_store),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::StorageBackend;

    use super::*;

    #[tokio::test]
    async fn test_two_silo_probing() {
        // analyzer
        let analyzer = SchemaAnalyzerBuilder::default().build();

        let csv_config =
            StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap();

        let parquet_config =
            StorageConfig::new_file_backend(StorageBackend::Parquet, "./assets/parquet").unwrap();

        let result = analyzer
            .analyze(vec![csv_config, parquet_config])
            .await
            .unwrap();

        assert!(result.is_some());

        assert_eq!(result.unwrap().len(), 1);
    }
}
