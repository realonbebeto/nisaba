use arrow::datatypes::{DataType, Field};
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    error::NError,
    types::{FieldDef, TableDef},
};

pub trait StdSchema {
    fn try_into_table_defs(self) -> Result<Vec<TableDef>, NError>;
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct InfoSchema {
    pub silo_id: String,
    pub table_schema: String,
    pub table_name: String,
    pub column_name: String,
    pub column_default: Option<String>,
    pub is_nullable: String,
    pub data_type: String,
    pub character_maximum_length: String,
    pub udt_name: String,
}

impl StdSchema for Vec<InfoSchema> {
    fn try_into_table_defs(self) -> Result<Vec<TableDef>, NError> {
        let fields: Vec<FieldDef> = self
            .into_iter()
            .map(|v| {
                let mut metadata = HashSet::new();

                metadata.insert(v.column_default);
                metadata.insert(Some(v.is_nullable));
                metadata.insert(Some(v.character_maximum_length));

                FieldDef {
                    id: Uuid::now_v7(),
                    silo_id: v.silo_id,
                    name: v.column_name,
                    table_name: v.table_name,
                    canonical_type: sql_to_arrow_type(&v.udt_name),
                    metadata,
                    cardinality: None,
                    sample_values: [const { None }; 17],
                }
            })
            .collect();

        let mut map: Vec<TableDef> = Vec::new();

        for f in fields {
            map.push(TableDef {
                silo_id: f.silo_id.clone(),
                id: Uuid::now_v7(),
                name: f.table_name.clone(),
                fields: vec![],
            });
            // TODO: nested and max-depth will have to be updated later
        }

        Ok(map)
    }
}

fn sql_to_arrow_type(data_type: &str) -> DataType {
    match data_type.to_lowercase().trim() {
        // TODO: For array, the subtype can be picked
        "array" => DataType::List(Arc::new(Field::new("", DataType::Utf8, false))),
        "bool" | "boolean" => DataType::Boolean,
        "numeric" | "decimal" => DataType::Decimal128(38, 0),
        "integer" | "int" | "int4" | "serial4" | "serial" => DataType::Int32,
        "smallint" | "int2" | "serial2" | "smallserial" => DataType::Int16,
        "tinyint" => DataType::Int8,
        "bigint" => DataType::Int64,
        "float2" => DataType::Float16,
        "real" | "float4" => DataType::Float32,
        "double" | "float" | "float8" => DataType::Float64,
        "date" => DataType::Date32,
        "bytea" => DataType::Binary,
        "jsonb" => DataType::LargeBinary,
        "json" | "text" => DataType::LargeUtf8,
        "character" | "char" | "character varying" | "varchar" => DataType::Utf8,
        _ => DataType::Null,
    }
}
