use uuid::Uuid;

use crate::types::{FieldDef, MatchCandidate, Matchable, TableDef};

pub trait ClusterItem: Sized {
    type Def: ClusterDef;
    fn from_def(id: <Self::Def as Matchable>::Id, def: &Self::Def) -> Self;
}

pub trait ClusterDef: Matchable + Clone {
    fn name(&self) -> &str;
    fn table_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct TableCluster {
    pub cluster_id: u32,
    pub tables: Vec<TableResult>,
    pub field_clusters: Vec<FieldCluster>,
}

#[derive(Debug, Clone)]
pub struct FieldCluster {
    pub cluster_id: u32,
    pub fields: Vec<FieldResult>,
}

#[derive(Debug, Clone)]
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
