use arrow::datatypes::{DataType, Field};

use std::sync::Arc;

mod field;
mod table;

pub use field::FieldDef;
pub use table::TableDef;

use crate::analyzer::retriever::Storable;

pub trait Matchable {
    type Id: Clone;
    type Match: MatchCandidate<Id = Self::Id>;

    fn id(&self) -> Self::Id;
    fn silo_id(&self) -> &str;
}

#[allow(unused)]
pub trait MatchCandidate {
    type Id: Clone;
    type Body: Storable;

    fn confidence(&self) -> f32;
    fn schema_id(&self) -> Self::Id;
    fn schema_silo_id(&self) -> &str;
    fn body(&self) -> &Self::Body;
}

pub fn get_field_defs() -> Vec<Field> {
    vec![
        Field::new("id", DataType::FixedSizeBinary(16), false),
        Field::new("silo_id", DataType::Utf8, false),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("canonical_type", DataType::Utf8, false),
        Field::new("type_confidence", DataType::Float32, false),
        Field::new("cardinality", DataType::Float32, true),
        Field::new("avg_byte_length", DataType::Float32, true),
        Field::new("is_monotonic", DataType::Boolean, false),
        Field::new(
            "char_class_signature",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            true,
        ),
        Field::new("column_default", DataType::Utf8, true),
        Field::new("is_nullable", DataType::Boolean, false),
        Field::new("char_max_length", DataType::Int32, true),
        Field::new("numeric_precision", DataType::Int32, true),
        Field::new("numeric_scale", DataType::Int32, true),
        Field::new("datetime_precision", DataType::Int32, true),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 384),
            true,
        ),
    ]
}
