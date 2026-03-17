//! Core implementations to handle reconciliation of data stores
//!
//! The core traits and types for data reconciliation, responsible for reading data sources, analyzing the data sources
//! and storage while providing extensibility on the part of inference as custom inference implementations can be developed.
//!
//! Overview
//! - [`FieldCluster`] - A TableCluster member type that is responsible to  hold field results.
//! - [`InferenceEngineRegistry`]: A type for managing inference engines.
//! - [`GraphCluster']: An important utility type for clustering store elements (Tables/Fields).
//! - [`LatentStore`]: A type responsible for giving access to Lancedb vector store.
//! - [`SchemaAnalyzer`]: A public type that is responsible to commence actual reconciliation.
//! - [`SchemaAnalyzerBuilder`]: A builder type for creating the SchemaAnalyzer.
//! - [`SchemaInferenceEngine`]: A primary trait representing built-in functionality for store reads and inference.
//! - [`Storable`]: A vital trait to give access to attributes of store elements (TableDef/FieldDef)
//! - [`Source`]: An type providing access ro various data sources
//! - [`TableCluster`]: A type responsible to hold table results from a reconciliation run.
//! - [`TypeLatticeResolver`]: A promotion type that is responsible to promoted field types that come from file based stores.
//!

pub mod calculation;
pub mod datastore;
pub mod inference;
pub mod metrics;
pub mod promote;
pub mod reconcile;
pub mod report;
pub mod retriever;

use std::{
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use fastembed::EmbeddingModel;
use lancedb::DistanceType;

use crate::{LatentStore, Source, error::NisabaError};

#[derive(Debug, Clone)]
/// The code defines a struct `AnalyzerConfig` with various fields representing weights and thresholds
/// for an analyzer configuration.
pub struct AnalyzerConfig {
    /// configs related to how search candidates are weighted using their attributes
    pub scoring: ScoringConfig,
    /// Number of rows/records used to make type inference and promotions
    pub sample_size: Option<usize>,
    /// configs related to how search candidates are compared for closeness/resemblance
    pub similarity: SimilarityConfig,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        AnalyzerConfig {
            scoring: ScoringConfig::default(),
            sample_size: Some(10),
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
    /// Impact of types on computing aggregated table embedding
    pub type_weight: f32,
    /// Impact of generalized structure on computing aggregated table embedding
    pub structure_weight: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        ScoringConfig {
            type_weight: 1.,
            structure_weight: 0.,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimilarityConfig {
    /// Measure for providing cutoff of similar items
    pub threshold: f32,
    /// Maximum number of retrieving similar  items
    pub top_k: Option<usize>,
    /// Type of measurement algorithm for disparity
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
    pub threads: usize,
}

/// The `SchemaAnalyzer` provides an interface for store reconciliation. It contains fields for name,
/// configuration, sources and runtime state.
pub struct SchemaAnalyzer {
    /// name of the schema analyzer.
    name: String,
    /// Array of sources to read and infer data from
    sources: Vec<Source>,
    /// Runtime state for persistence, stats, and compute
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

#[derive(Clone, Debug, Default)]
pub struct InferenceStats {
    pub sources_analyzed: usize,
    pub tables_found: usize,
    pub tables_inferred: usize,
    pub fields_inferred: usize,
    pub errors: Vec<String>,
}

/// The `SchemaAnalyzerBuilder` helps build the schema analyzer.
/// It contains fields for configuration related to schemaanalysis.
#[derive(Default)]
pub struct SchemaAnalyzerBuilder {
    // fields for configuration
    name: Option<String>,
    sources: Vec<Source>,
    config: Option<AnalyzerConfig>,
    persist_path: Option<String>,
    embedding_model: Option<EmbeddingModel>,
    threads: Option<usize>,
}

impl SchemaAnalyzerBuilder {
    pub fn new() -> Self {
        SchemaAnalyzerBuilder::default()
    }
    /// The `name` function takes a mutable reference to a struct and a string, sets the
    /// struct's name field to the string value, and returns the modified struct.
    ///
    /// Arguments:
    ///
    /// * `name`: The `name` parameter is value that can be String that represents the name you
    ///   want to assign to the analyzer.
    ///
    /// Returns:
    ///
    /// The `self` object is being returned after setting the `name` field to the provided `name`
    /// string.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The `source` function adds/appends a data source.
    ///
    /// Arguments:
    ///
    /// * `source`: This parameter is data source on which to run an analysis.
    ///
    pub fn source(mut self, source: Source) -> Self {
        self.sources.push(source);
        self
    }

    /// The `sources` function adds/appends multiple data sources
    ///
    /// Arguments:
    ///
    /// * `sources`: This parameter is data source on which to run an analysis.
    ///
    pub fn sources(mut self, sources: impl IntoIterator<Item = Source>) -> Self {
        self.sources.extend(sources);
        self
    }

    /// The `persist_path` function connects the ablyzer to an existing Lancedb store
    ///
    /// Arguments:
    ///
    /// * `path`: This parameter is the path to an existing Lancedb store.
    ///
    pub fn persist_path(mut self, path: impl Into<String>) -> Self {
        self.persist_path = Some(path.into());
        self
    }

    /// The `embedding_model` function sets the embedding model for vector generation
    ///
    /// Arguments:
    ///
    /// * `model`: This parameter is a fastembed embedding model.
    ///
    pub fn embedding_model(mut self, model: EmbeddingModel) -> Self {
        self.embedding_model = Some(model);
        self
    }

    /// The `threads` function sets the number of threads to use in reading files
    ///
    /// Arguments:
    ///
    /// * `threads`: This parameter is number of threads for file reads.
    ///
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// The `config` function sets the configuration for an analyzer
    ///
    /// Arguments:
    ///
    /// * `config`: This parameter is a configuration for the analyzer by passing an instance of `AnalyzerConfig`
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
            None => {
                config.similarity.top_k = Some(self.sources.len());
            }
        }

        let config = Arc::new(config);

        let persistence = LatentStore::builder()
            .analyzer_config(config.clone())
            .connection_path(self.persist_path)
            .embedding_model(self.embedding_model)
            .build()
            .await?;

        let threads = if let Some(threads) = self.threads {
            if threads == 0 {
                return Err(NisabaError::Invalid(
                    "Invalid thread size: Threads must be greater than 0".into(),
                ));
            }
            threads
        } else {
            std::thread::available_parallelism().map_or(2, NonZeroUsize::get)
        };

        let context = InferenceContext {
            config: config.clone(),
            persistence: Arc::new(persistence),
            stats: Arc::new(Mutex::new(InferenceStats::default())),
            threads,
        };

        Ok(SchemaAnalyzer {
            name: self.name.unwrap_or("default".to_string()),
            context,
            sources: self.sources,
        })
    }
}
