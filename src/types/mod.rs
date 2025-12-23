use arrow::{
    array::{Array, AsArray, FixedSizeListArray, GenericListArray},
    datatypes::{DataType, Field},
};

use std::{collections::HashSet, sync::Arc};

mod field;
mod table;

pub use field::FieldDef;
pub use table::TableDef;

use crate::{analyzer::retriever::Storable, error::NError};

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
        Field::new("name", DataType::Utf8, false),
        Field::new("table_name", DataType::Utf8, false),
        Field::new("canonical_type", DataType::Utf8, false),
        Field::new(
            "metadata",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new(
            "sample_values",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Utf8, true)), 17),
            true,
        ),
        Field::new("cardinality", DataType::UInt64, true),
        Field::new(
            "cached_embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
            true,
        ),
    ]
}

pub fn extract_metadata(
    list_array: &GenericListArray<i64>,
    index: usize,
) -> Result<HashSet<Option<String>>, NError> {
    let values = list_array.value(index);
    let values = values.as_string::<i64>();

    let mut metadata = HashSet::new();
    for i in 0..values.len() {
        if values.is_null(i) {
            metadata.insert(None);
        } else {
            metadata.insert(Some(values.value(i).to_string()));
        }
    }

    Ok(metadata)
}

pub fn extract_sample_values(
    fixed_list: &FixedSizeListArray,
    index: usize,
) -> Result<[Option<String>; 17], NError> {
    let values = fixed_list.value(index);
    let values = values.as_string::<i64>();

    let mut sample_values: [Option<String>; 17] = Default::default();
    for (i, item) in values.iter().enumerate() {
        sample_values[i] = item.map(|v| v.to_string());
    }

    Ok(sample_values)
}
