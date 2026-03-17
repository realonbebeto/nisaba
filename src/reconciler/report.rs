use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use nalgebra::DVector;
use uuid::Uuid;

use crate::{
    reconciler::InferenceStats,
    types::{FieldDef, TableRep},
};

#[derive(Debug, Clone)]
pub struct ReconcileReport {
    pub stats: InferenceStats,
    pub tables: Option<Vec<TableCluster>>,
}

#[derive(Debug, Clone)]
/// The `TableCluster` represents the result of a clustering process
pub struct TableCluster {
    /// u32 identifier
    pub cluster_id: u32,
    /// member table results.
    pub tables: Vec<TableResult>,
    /// member field results.
    pub field_clusters: Vec<FieldCluster>,
    /// similarity confidence of the cluste
    pub confidence: f32,
    /// Coverage score
    pub coverage_score: f32,
}

impl Hash for TableCluster {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut t_ids: Vec<Uuid> = self.tables.iter().map(|t| t.id).collect();
        t_ids.sort_unstable();
        t_ids.hash(state);

        let mut f_ids: Vec<Uuid> = self
            .field_clusters
            .iter()
            .flat_map(|fc| fc.fields.iter().map(|f| f.id))
            .collect();
        f_ids.sort_unstable();
        f_ids.hash(state);
    }
}

impl PartialEq for TableCluster {
    fn eq(&self, other: &Self) -> bool {
        let t_ids: HashSet<Uuid> = self.tables.iter().map(|t| t.id).collect();
        let ot_ids: HashSet<Uuid> = other.tables.iter().map(|t| t.id).collect();

        if t_ids != ot_ids {
            return false;
        }

        let f_ids: HashSet<Uuid> = self
            .field_clusters
            .iter()
            .flat_map(|fc| fc.fields.iter().map(|f| f.id))
            .collect();

        let of_ids: HashSet<Uuid> = self
            .field_clusters
            .iter()
            .flat_map(|fc| fc.fields.iter().map(|f| f.id))
            .collect();

        f_ids == of_ids
    }
}

impl Eq for TableCluster {}

#[derive(Debug, Clone)]
/// The `FieldCluster` represents the result of a field clustering process
pub struct FieldCluster {
    /// u32 identifier
    pub cluster_id: u32,
    /// member field results
    pub fields: Vec<FieldResult>,
    /// similarity confidence of the cluster
    pub confidence: f32,
}

impl PartialEq for FieldCluster {
    fn eq(&self, other: &Self) -> bool {
        let f_ids: HashSet<Uuid> = self.fields.iter().map(|fr| fr.id).collect();
        let of_ids: HashSet<Uuid> = other.fields.iter().map(|fr| fr.id).collect();

        f_ids == of_ids
    }
}

// impl Eq for FieldCluster {}

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
    /// Uuid of the table in which the field is a member
    pub table_id: Uuid,
    /// Utf8/String name of the field.
    pub field_name: String,
    /// Measure of resemblace in the group/cluster it exists in
    pub similarity: f32,
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

#[derive(Clone, Debug)]
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
    /// Embedding representation for the field
    pub embedding: DVector<f32>,
}

#[derive(Debug)]
pub struct LoadedFields {
    pub query_fields: Vec<FieldMatch>,
    pub candidate_fields: HashMap<Uuid, Vec<FieldMatch>>,
}
