use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    reconciler::{report::TableMatch, retriever::Storable},
    types::field::FieldProfile,
};

#[derive(Debug, Clone)]
pub struct TableRep {
    /// Global unique Id for the table
    pub id: Uuid,
    /// Id of silo in which the table is member
    pub silo_id: String,
    /// Name of the schema in which the field is member
    pub table_schema: String,
    /// Name of the table
    pub name: String,
    /// Vec of the FieldDef associated with table
    pub fields: Vec<Uuid>,
}

impl From<&TableDef> for TableRep {
    fn from(value: &TableDef) -> Self {
        let fields: Vec<Uuid> = value.fields.iter().map(|f| f.field_def.id).collect();
        Self {
            id: value.id,
            silo_id: value.silo_id.clone(),
            table_schema: value.table_schema.clone(),
            name: value.name.clone(),
            fields,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableDef {
    /// Global unique Id for the table
    pub id: Uuid,
    /// Id of silo in which the table is member
    pub silo_id: String,
    /// Name of the schema in which the field is member
    pub table_schema: String,
    /// Name of the table
    pub name: String,
    /// Vec of the FieldDef associated with table
    pub fields: Vec<FieldProfile>,
}

impl Storable for TableRep {
    type SearchResult = TableMatch;

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn table_id(&self) -> &Uuid {
        &self.id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn schema(dim: usize) -> std::sync::Arc<arrow::datatypes::Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("table_schema", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "fields",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
                false,
            ),
            Field::new("num_fields", DataType::Int16, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
        ]))
    }

    fn result_columns() -> Vec<String> {
        vec![
            "id".to_string(),
            "silo_id".to_string(),
            "table_schema".to_string(),
            "name".to_string(),
            "fields".to_string(),
            "_distance".to_string(),
        ]
    }

    fn vtable_name() -> &'static str {
        "table_def"
    }
}
