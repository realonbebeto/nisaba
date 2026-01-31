use uuid::Uuid;

use crate::types::{FieldDef, TableRep};

#[derive(Debug, Clone)]
/// The `TableCluster` represents the result of a clustering process
pub struct TableCluster {
    /// u32 identifier
    pub cluster_id: u32,
    /// member table results.
    pub tables: Vec<TableResult>,
    /// member field results.
    pub field_clusters: Vec<FieldCluster>,
}

#[derive(Debug, Clone)]
/// The `FieldCluster` represents the result of a field clustering process
pub struct FieldCluster {
    /// u32 identifier
    pub cluster_id: u32,
    /// member field results
    pub fields: Vec<FieldResult>,
}

#[derive(Debug, Clone)]
/// The `TableResult` represents the identity information of a table
pub struct TableResult {
    /// Uuid identifier
    pub id: Uuid,
    /// String Id of the silo in which the table is a membe
    pub silo_id: String,
    /// Utf8/String name of the table.
    pub table_name: String,
}

#[derive(Debug, Clone)]
/// The `FieldResult` represents the identity information of a field
pub struct FieldResult {
    /// Uuid identifier
    pub id: Uuid,
    /// String Id of the silo in which the table is a member
    pub silo_id: String,
    /// Utf8/String name of the table in which the field is a member
    pub table_name: String,
    /// Utf8/String name of the field.
    pub field_name: String,
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct MatchExplanation {
    pub source: Uuid,
    pub target: Uuid,
    pub similarity_score: f32,
    pub conflict_penalty: f32,
    pub final_score: f32,
    pub reasons: Vec<String>,
}

/// The `TableMatch` represents the table data as read from the latent store.
pub struct TableMatch {
    /// TableRep values as read from the latent store.
    pub schema: TableRep,
    /// floating-point value cosine distance from the latent store.
    pub confidence: f32,
}

/// The `FieldMatch` represents the field data as read from the latent store.
#[derive(Debug, PartialEq)]
pub struct FieldMatch {
    /// FieldDef values as read from the latent store.
    pub schema: FieldDef,
    /// floating-point value cosine distance from the latent store.
    pub confidence: f32,
}
