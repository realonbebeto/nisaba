use arrow::{
    array::{FixedSizeBinaryArray, Float32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use futures::executor::block_on;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        calculation::{ArrowGraph, leiden_communities},
        catalog::StorageConfig,
        inference::InferenceEngineRegistry,
        report::{
            ClusterDef, ClusterItem, FieldCluster, FieldMatch, FieldResult, MatchExplanation,
            TableCluster, TableMatch, TableResult,
        },
        retriever::{LatentStore, Storable},
    },
    error::NisabaError,
    types::{FieldDef, MatchCandidate, TableDef},
};

#[derive(Debug, Clone)]
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
            similarity_threshold: 0.77,
            top_k: None,
        }
    }
}

pub struct SchemaAnalyzer {
    pub(crate) name: String,
    pub(crate) config: Arc<AnalyzerConfig>,
    pub(crate) inference_engine: InferenceEngineRegistry,
    pub(crate) latent_store: Arc<LatentStore>,
    pub(crate) conflict_scorer: ConflictScorer,
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
    pub async fn analyze(
        &self,
        configs: Vec<StorageConfig>,
    ) -> Result<Vec<TableCluster>, NisabaError> {
        // Take inputs (multiple collections/tables)
        // E.g. Vec[Collection/Table], Vec[Collection/Table], Vec[Collection/Table] ...

        let tbl_defs = self.inference_engine.discover_ecosystem(configs)?;

        self.index_schemas(&tbl_defs).await?;

        // Match Collection/Tables
        let (table_graph, tbl_explanations) = self.table_def_arrow_graph(&tbl_defs).await?;

        let mut clusters =
            self.cluster_leiden(&table_graph, &tbl_defs, None, None, |cluster_id, tr| {
                TableCluster {
                    cluster_id,
                    tables: tr,
                    field_clusters: Vec::new(),
                    explanations: Vec::new(),
                }
            })?;

        // Match Field Names Inside matched Collection/Tables
        self.cluster_fields(&mut clusters, &tbl_defs).await?;

        for cluster in &mut clusters {
            for table in &cluster.tables {
                cluster.explanations.extend(
                    tbl_explanations
                        .iter()
                        .filter(|e| e.source == table.id || e.target == table.id)
                        .cloned(),
                );
            }

            // Deduplicate explanations
            cluster
                .explanations
                .sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.source)));
            cluster
                .explanations
                .dedup_by(|a, b| (a.source == b.source) && (a.target == b.target));
        }

        // TODO: Optional CleanUp - Delete the lancedb

        Ok(clusters)
    }

    async fn index_schemas(&self, schema_items: &[TableDef]) -> Result<(), NisabaError> {
        let table_handler = self
            .latent_store
            .table_handler::<TableDef>(self.config.clone());

        table_handler.initialize().await?;

        table_handler.store(schema_items).await?;

        Ok(())
    }

    async fn table_def_arrow_graph(
        &self,
        table_defs: &[TableDef],
    ) -> Result<(ArrowGraph, Vec<MatchExplanation>), NisabaError> {
        let columns = vec![
            "id".to_string(),
            "silo_id".to_string(),
            "name".to_string(),
            "fields".to_string(),
            "_distance".to_string(),
        ];
        self.build_arrow_graph(
            table_defs,
            columns,
            |table, candidate: &TableMatch, threshold| {
                (1.0 - candidate.confidence >= threshold) && (candidate.schema.id != table.id)
            },
        )
        .await
    }

    async fn field_def_arrow_graph(
        &self,
        field_defs: &[FieldDef],
    ) -> Result<(ArrowGraph, Vec<MatchExplanation>), NisabaError> {
        let columns = vec![
            "id".to_string(),
            "silo_id".to_string(),
            "table_name".to_string(),
            "name".to_string(),
            "canonical_type".to_string(),
            "metadata".to_string(),
            "sample_values".to_string(),
            "cardinality".to_string(),
            "_distance".to_string(),
        ];

        self.build_arrow_graph(
            field_defs,
            columns,
            |field, candidate: &FieldMatch, threshold| {
                (1.0 - candidate.confidence >= threshold) && (candidate.schema.id != field.id)
            },
        )
        .await
    }

    async fn build_arrow_graph<T>(
        &self,
        defs: &[T],
        columns: Vec<String>,
        filter_fn: impl Fn(&T, &T::SearchResult, f32) -> bool,
    ) -> Result<(ArrowGraph, Vec<MatchExplanation>), NisabaError>
    where
        T: Storable,
        T::SearchResult: MatchCandidate<Body = T>,
    {
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        let mut explanations = Vec::new();

        for def in defs.iter() {
            let candidates = self.find_store_matches(def, columns.clone()).await?;

            // Filtering candidates out for matching schema ids and  similarity threshold set
            let candidates = candidates
                .into_iter()
                .filter(|c| filter_fn(def, c, self.config.similarity_threshold))
                .collect::<Vec<T::SearchResult>>();

            for candidate in &candidates {
                let penalty = self.conflict_scorer.compute_penalty(def, candidate.body());

                let adj_score = candidate.confidence() * (1.0 - penalty);

                if adj_score >= self.config.similarity_threshold {
                    sources.push(def.get_id());
                    targets.push(candidate.body().get_id());
                    weights.push(adj_score);

                    // Generate explanation
                    let mut reasons = vec![format!(
                        "Embedding cosine similarity: {:.3}",
                        candidate.confidence()
                    )];

                    if penalty > 0.0 {
                        reasons.push(format!("Conflict penalty: {:.3}", penalty));

                        if def.silo_id() == candidate.body().silo_id() {
                            reasons.push("Same silo - confidence reduced".to_string());
                        }
                    }

                    explanations.push(MatchExplanation {
                        source: def.get_id(),
                        target: candidate.body().get_id(),
                        similarity_score: candidate.confidence(),
                        conflict_penalty: penalty,
                        final_score: adj_score,
                        reasons,
                    });
                }
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::FixedSizeBinary(16), false),
            Field::new("target", DataType::FixedSizeBinary(16), false),
            Field::new("weight", DataType::Float32, false),
        ]));

        let edges = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(FixedSizeBinaryArray::try_from_iter(
                    sources.iter().map(|v| v.into_bytes()),
                )?),
                Arc::new(FixedSizeBinaryArray::try_from_iter(
                    targets.iter().map(|v| v.into_bytes()),
                )?),
                Arc::new(Float32Array::from(weights)),
            ],
        )?;

        let graph = ArrowGraph::from_edges(edges)?;

        Ok((graph, explanations))
    }

    pub fn table_cluster_leiden(
        &self,
        graph: &ArrowGraph,
        table_defs: &[TableDef],
        resolution: Option<f32>,
        max_iterations: Option<u32>,
    ) -> Result<Vec<TableCluster>, NisabaError> {
        let leiden_result = leiden_communities(graph, resolution, max_iterations)?;

        let mut communities: HashMap<u32, Vec<TableResult>> = HashMap::new();

        for (table_id, cluster_id) in leiden_result {
            let tbl_schema = table_defs.iter().find(|t| t.id == table_id);
            if let Some(v) = tbl_schema {
                let tbl = TableResult {
                    id: table_id,
                    silo_id: v.silo_id.clone(),
                    table_name: v.name.clone(),
                };
                communities.entry(cluster_id).or_default().push(tbl);
            }
        }

        let mut clusters = Vec::new();

        for (cluster_id, tables) in communities {
            clusters.push(TableCluster {
                cluster_id,
                tables,
                field_clusters: Vec::new(),
                explanations: Vec::new(),
            });
        }

        clusters.sort_by(|a, b| b.tables.len().cmp(&a.tables.len()));

        Ok(clusters)
    }

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
            let (field_graph, field_explanations) = self.field_def_arrow_graph(&field_defs).await?;

            // Cluster fields
            let mut clusters =
                self.cluster_leiden(&field_graph, &field_defs, None, None, |cluster_id, fr| {
                    FieldCluster {
                        cluster_id,
                        fields: fr,
                        explanations: Vec::new(),
                    }
                })?;

            //Attach /Update explanations
            for cluster in &mut clusters {
                for table in &cluster.fields {
                    cluster.explanations.extend(
                        field_explanations
                            .iter()
                            .filter(|e| e.source == table.id || e.target == table.id)
                            .cloned(),
                    );
                }

                // Deduplicate explanations
                cluster
                    .explanations
                    .sort_by(|a, b| a.source.cmp(&b.source).then(a.target.cmp(&b.source)));
                cluster
                    .explanations
                    .dedup_by(|a, b| (a.source == b.source) && (a.target == b.target));
            }

            tbl_cluster.field_clusters = clusters;

            table_handler.clear_table().await?;
        }

        Ok(())
    }

    fn cluster_leiden<D, R, C>(
        &self,
        graph: &ArrowGraph,
        defs: &[D],
        resolution: Option<f32>,
        max_iterations: Option<u32>,
        build_cluster: impl Fn(u32, Vec<R>) -> C,
    ) -> Result<Vec<C>, NisabaError>
    where
        D: ClusterDef<Id = Uuid>,
        R: ClusterItem<Def = D>,
    {
        let leiden_result = leiden_communities(graph, resolution, max_iterations)?;
        let mut communities: HashMap<u32, Vec<R>> = HashMap::new();

        for (item_id, cluster_id) in leiden_result {
            if let Some(def) = defs.iter().find(|d| d.id() == item_id) {
                communities
                    .entry(cluster_id)
                    .or_default()
                    .push(R::from_def(item_id, def));
            }
        }

        let clusters: Vec<C> = communities
            .into_iter()
            .map(|(cluster_id, items)| build_cluster(cluster_id, items))
            .collect();

        Ok(clusters)
    }

    pub fn field_cluster_leiden(
        &self,
        graph: &ArrowGraph,
        field_defs: &[FieldDef],
        resolution: Option<f32>,
        max_iterations: Option<u32>,
    ) -> Result<Vec<FieldCluster>, NisabaError> {
        let leiden_result = leiden_communities(graph, resolution, max_iterations)?;

        let mut communities: HashMap<u32, Vec<FieldResult>> = HashMap::new();

        for (field_id, cluster_id) in leiden_result {
            let tbl_schema = field_defs.iter().find(|t| t.id == field_id);
            if let Some(v) = tbl_schema {
                let tbl = FieldResult {
                    id: field_id,
                    silo_id: v.silo_id.clone(),
                    table_name: v.name.clone(),
                    field_name: v.name.clone(),
                };
                communities.entry(cluster_id).or_default().push(tbl);
            }
        }

        let mut clusters = Vec::new();

        for (cluster_id, fields) in communities {
            clusters.push(FieldCluster {
                cluster_id,
                fields,
                explanations: Vec::new(),
            });
        }

        clusters.sort_by(|a, b| b.fields.len().cmp(&a.fields.len()));

        Ok(clusters)
    }

    async fn find_store_matches<T: Storable>(
        &self,
        query_schema: &T,
        columns: Vec<String>,
    ) -> Result<Vec<T::SearchResult>, NisabaError> {
        let table_handler = self.latent_store.table_handler::<T>(self.config.clone());

        let candidates = table_handler
            .search(query_schema, self.config.clone(), columns)
            .await?;

        Ok(candidates)
    }
}

pub struct SchemaAnalyzerBuilder {
    // fields for configuration
    pub(crate) name: String,
    pub(crate) config: AnalyzerConfig,
    pub(crate) inference_registry: InferenceEngineRegistry,
    pub(crate) latent_store: Arc<LatentStore>,
    pub(crate) conflict_scorer: ConflictScorer,
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
            latent_store: Arc::new(ls),
            conflict_scorer: ConflictScorer::new(),
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
    /// * `name`: The `name` parameter is a reference to a string (`&str`) that represents the name you want to assign to the object. The intuition is to having multiple analyzers whose outputs can be averaged.
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
            latent_store: self.latent_store,
            conflict_scorer: self.conflict_scorer,
        }
    }
}

// ==================================
// Conflict Scoring System
// ==================================
pub struct ConflictScorer {
    same_silo_penalty: f32,
}

impl Default for ConflictScorer {
    fn default() -> Self {
        ConflictScorer {
            same_silo_penalty: 0.25,
        }
    }
}

impl ConflictScorer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The function `with_penalties` initializes a struct with a penalty value for the same
    /// silo.
    ///
    /// Arguments:
    ///
    /// * `same_silo`: The parameter `same_silo` in the `with_penalties` function represents the penalty
    ///   value associated with items being in the same silo.
    ///
    /// Returns:
    ///
    /// A struct instance with the field `same_silo_penalty` initialized with the value of the
    /// `same_silo` parameter is being returned.
    #[allow(unused)]
    pub fn with_penalties(same_silo: f32) -> Self {
        Self {
            same_silo_penalty: same_silo,
        }
    }

    pub fn compute_penalty<T: Storable>(&self, def_a: &T, def_b: &T) -> f32 {
        let mut penalty = 0.0;

        if def_a.silo_id() == def_b.silo_id() {
            penalty += self.same_silo_penalty;
        }

        penalty
    }
}
