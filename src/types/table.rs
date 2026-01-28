use arrow::datatypes::{DataType, Field, Schema};
use nalgebra::{DVector, SVector};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    analyzer::{calculation::deterministic_projection, report::TableMatch, retriever::Storable},
    types::FieldDef,
};

#[derive(Debug, Clone)]
pub struct TableRep {
    /// Global unique Id for the table
    pub id: Uuid,
    /// Id of silo in which the table is member
    pub silo_id: String,
    /// Name of the table
    pub name: String,
    /// Vec of the FieldDef associated with table
    pub fields: Vec<Uuid>,
}

impl From<&TableDef> for TableRep {
    fn from(value: &TableDef) -> Self {
        let fields: Vec<Uuid> = value.fields.iter().map(|f| f.id).collect();
        Self {
            id: value.id,
            silo_id: value.silo_id.clone(),
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
    /// Name of the table
    pub name: String,
    /// Vec of the FieldDef associated with table
    pub fields: Vec<FieldDef>,
}

impl Storable for TableRep {
    type SearchResult = TableMatch;

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn silo_id(&self) -> &str {
        &self.silo_id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn schema() -> std::sync::Arc<arrow::datatypes::Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                true,
            ),
            Field::new(
                "fields",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
                false,
            ),
        ]))
    }

    fn result_columns() -> Vec<String> {
        vec![
            "id".to_string(),
            "silo_id".to_string(),
            "name".to_string(),
            "fields".to_string(),
            "_distance".to_string(),
        ]
    }

    fn vtable_name() -> &'static str {
        "table_def"
    }
}

impl TableDef {
    /// The `structure` function profiles the fields of the TableDef by count and DataType
    /// for linear projection purposes
    ///
    /// Returns
    ///
    /// A `TableStats`
    pub fn structure(&self) -> TableStats {
        let mut structure = TableStats::default();

        for f in &self.fields {
            match f.canonical_type {
                DataType::List { .. }
                | DataType::LargeList { .. }
                | DataType::LargeListView { .. }
                | DataType::FixedSizeList { .. }
                | DataType::ListView { .. } => {
                    structure.num_array += 1.0;
                }
                DataType::Decimal32 { .. }
                | DataType::Decimal64 { .. }
                | DataType::Decimal256 { .. }
                | DataType::Decimal128 { .. } => {
                    structure.num_decimal128 += 1.0;
                }
                DataType::Int64 | DataType::UInt64 => {
                    structure.num_int64 += 1.0;
                }
                DataType::Int32 | DataType::UInt32 => {
                    structure.num_int32 += 1.0;
                }
                DataType::Int16 | DataType::UInt16 | DataType::Int8 | DataType::UInt8 => {
                    structure.num_int16 += 1.0;
                }
                DataType::Boolean => {
                    structure.num_bool += 1.0;
                }
                DataType::Null
                | DataType::Struct { .. }
                | DataType::Dictionary { .. }
                | DataType::Map { .. }
                | DataType::RunEndEncoded { .. }
                | DataType::Union { .. } => {
                    structure.num_unknown += 1.0;
                }
                DataType::Date32 | DataType::Date64 => {
                    structure.num_date32 += 1.0;
                }
                DataType::Time32 { .. }
                | DataType::Time64 { .. }
                | DataType::Duration { .. }
                | DataType::Interval { .. } => {
                    structure.num_time += 1.0;
                }
                DataType::Timestamp { .. } => {
                    structure.num_timestamps += 1.0;
                }
                DataType::Float16 => {
                    structure.num_float16 += 1.0;
                }
                DataType::Float32 => {
                    structure.num_float32 += 1.0;
                }
                DataType::Float64 => {
                    structure.num_float64 += 1.0;
                }
                DataType::Binary
                | DataType::FixedSizeBinary { .. }
                | DataType::LargeBinary
                | DataType::BinaryView => {
                    structure.num_binary += 1.0;
                }
                DataType::Utf8 | DataType::Utf8View => {
                    structure.num_utf8 += 1.0;
                }
                DataType::LargeUtf8 => {
                    structure.num_largeutf += 1.0;
                }
            }
        }

        structure.num_fields = self.fields.len() as f32;

        structure
    }
}

#[derive(Debug)]
/// The `TableStats` provides a numeric profile of how table is constituted by DataType
pub struct TableStats {
    pub num_array: f32,
    pub num_decimal128: f32,
    pub num_int32: f32,
    pub num_int16: f32,
    pub num_float16: f32,
    pub num_float32: f32,
    pub num_float64: f32,
    pub num_date32: f32,
    pub num_binary: f32,
    pub num_time: f32,
    pub num_timestamps: f32,
    pub num_int64: f32,
    pub num_int8: f32,
    pub num_uint8: f32,
    pub num_bool: f32,
    pub num_fixed_sized_binary: f32,
    pub num_utf8: f32,
    pub num_largeutf: f32,
    pub num_unknown: f32,
    pub num_fields: f32,
}

impl TableStats {
    /// The `embedding` function runs to generate an embedding from
    /// the numeric profile through linear projection
    ///
    /// Returns:
    ///
    /// A nalgebra Matrix of floats
    pub fn embedding(&self) -> DVector<f32> {
        let raw_embed: SVector<f32, 19> = SVector::from_vec(vec![
            self.num_array,
            self.num_decimal128,
            self.num_int32,
            self.num_int16,
            self.num_float16,
            self.num_float32,
            self.num_float64,
            self.num_date32,
            self.num_binary,
            self.num_timestamps,
            self.num_int64,
            self.num_int8,
            self.num_uint8,
            self.num_bool,
            self.num_fixed_sized_binary,
            self.num_utf8,
            self.num_largeutf,
            self.num_unknown,
            self.num_fields,
        ]);

        let raw_embed = raw_embed / self.num_fields;

        // Dimensionality expansion
        deterministic_projection::<19>(raw_embed, 384, 42)
    }
}

impl Default for TableStats {
    fn default() -> Self {
        TableStats {
            num_array: 0.0,
            num_decimal128: 0.0,
            num_int32: 0.0,
            num_int16: 0.0,
            num_float16: 0.0,
            num_float32: 0.0,
            num_float64: 0.0,
            num_date32: 0.0,
            num_binary: 0.0,
            num_time: 0.0,
            num_timestamps: 0.0,
            num_int64: 0.0,
            num_int8: 0.0,
            num_uint8: 0.0,
            num_bool: 0.0,
            num_fixed_sized_binary: 0.0,
            num_utf8: 0.0,
            num_largeutf: 0.0,
            num_fields: 0.0,
            num_unknown: 0.0,
        }
    }
}
