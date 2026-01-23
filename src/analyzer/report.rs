use uuid::Uuid;

use crate::types::{FieldDef, MatchCandidate, Matchable, TableDef};

/// Trait used to gererate FieldResult or TableResult from FieldDef or TableDef
/// and Id of FieldMatch/TableMatch
pub trait ClusterItem: Sized {
    type Def: ClusterDef;
    fn from_def(id: <Self::Def as Matchable>::Id, def: &Self::Def) -> Self;
}

/// Trait used to provide access to name and table_name in FieldDef/TableDef
pub trait ClusterDef: Matchable + Clone {
    fn name(&self) -> &str;
    fn table_name(&self) -> &str;
}

#[derive(Debug, Clone)]
/// The `TableCluster` represents the result of a clustering process
///
/// Properties:
///
/// * `cluster_id`: The `cluster_id` property is a `u32` id.
///
/// * `tables`: The `tables` property is a vector of TableResult hold member table results.
///
/// * `field_clusters`: The `field_clusters` property represents the vector of member field results.
///
pub struct TableCluster {
    pub cluster_id: u32,
    pub tables: Vec<TableResult>,
    pub field_clusters: Vec<FieldCluster>,
}

#[derive(Debug, Clone)]
/// The `FieldCluster` represents the result of a field clustering process
///
/// Properties:
///
/// * `cluster_id`: The `cluster_id` property is a `u32` id.
///
/// * `fields`: The `fields` property represents the vector of member field results
pub struct FieldCluster {
    pub cluster_id: u32,
    pub fields: Vec<FieldResult>,
}

#[derive(Debug, Clone)]
/// The `TableResult` represents the identity information of a table
///
/// Properties:
///
/// * `id`: The `id` property is a `u32` id.
///
/// * `silo_id`: The `silo_id` property represents String Id of the silo in which the table is a member.
///
/// * `table_name`: The `table_name` property represents the Utf8/String name of the table.
pub struct TableResult {
    pub id: Uuid,
    pub silo_id: String,
    pub table_name: String,
}

impl ClusterItem for TableResult {
    type Def = TableDef;
    fn from_def(id: <Self::Def as Matchable>::Id, def: &Self::Def) -> Self {
        TableResult {
            id,
            silo_id: def.silo_id().to_string(),
            table_name: def.table_name().to_string(),
        }
    }
}

#[derive(Debug, Clone)]
/// The `FieldResult` represents the identity information of a field
///
/// Properties:
///
/// * `id`: The `cluster_id` property is a `u32` id.
///
/// * `silo_id`: The `silo_id` property represents String Id of the silo in which the table is a member.
///
/// * `table_name`: The `table_name` property represents the Utf8/String name of the table in which the field is a member.
///
/// * `field_name`: The `field_name` property represents the Utf8/String name of the field.
pub struct FieldResult {
    pub id: Uuid,
    pub silo_id: String,
    pub table_name: String,
    pub field_name: String,
}

impl ClusterItem for FieldResult {
    type Def = FieldDef;
    fn from_def(id: <Self::Def as Matchable>::Id, def: &Self::Def) -> Self {
        FieldResult {
            id,
            silo_id: def.silo_id().to_string(),
            table_name: def.table_name().to_string(),
            field_name: def.name().to_string(),
        }
    }
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
///
/// Properties:
///
/// * `schema`: The `schema` property holds the TableDef values as read from the latent store.
///
/// * `confidence`: The `confidence` property represents floating-point value cosine distance from the latent store.
pub struct TableMatch {
    pub schema: TableDef,
    pub confidence: f32,
}

impl MatchCandidate for TableMatch {
    type Id = Uuid;
    type Body = TableDef;

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn schema_id(&self) -> Self::Id {
        self.schema.id
    }

    fn schema_silo_id(&self) -> &str {
        &self.schema.silo_id
    }

    fn body(&self) -> &Self::Body {
        &self.schema
    }
}

/// The `FieldMatch` represents the field data as read from the latent store.
///
/// Properties:
///
/// * `schema`: The `schema` property holds the FieldDef values as read from the latent store.
///
/// * `confidence`: The `confidence` property represents floating-point value cosine distance from the latent store.
pub struct FieldMatch {
    pub schema: FieldDef,
    pub confidence: f32,
}

impl MatchCandidate for FieldMatch {
    type Id = Uuid;
    type Body = FieldDef;

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn schema_id(&self) -> Self::Id {
        self.schema.id
    }

    fn schema_silo_id(&self) -> &str {
        &self.schema.silo_id
    }

    fn body(&self) -> &Self::Body {
        &self.schema
    }
}
