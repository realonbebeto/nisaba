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
//! - [`StorageConfig`]: An type responsible for giving access to storage backends.
//! - [`TableCluster`]: A type responsible to hold table results from a reconciliation run.
//! - [`TypeLatticeResolver`]: A promotion type that is responsible to promoted field types that come from file based stores.
//!

pub mod calculation;
pub mod datastore;
pub mod inference;
pub mod probe;
pub mod report;
pub mod retriever;

pub use probe::{AnalyzerConfig, SchemaAnalyzer};
