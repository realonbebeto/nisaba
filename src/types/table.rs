use arrow::{
    array::{
        Array, ArrayRef, AsArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array,
        GenericListArray, RecordBatch, StringArray, StructArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Fields, Float32Type, Int32Type, Schema},
    error::ArrowError,
};
use nalgebra::{DVector, Vector1};
use std::sync::Arc;
use uuid::Uuid;

use crate::{
    analyzer::{
        AnalyzerConfig,
        calculation::deterministic_projection,
        report::{ClusterDef, TableMatch},
        retriever::Storable,
    },
    error::NisabaError,
    types::{FieldDef, Matchable, get_field_defs},
};

#[derive(Debug)]
pub struct TableDef {
    pub id: Uuid,
    pub silo_id: String,
    pub name: String,
    pub fields: Vec<FieldDef>,
}

impl ClusterDef for TableDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn table_name(&self) -> &str {
        &self.name
    }
}

impl Matchable for TableDef {
    type Id = Uuid;
    type Match = TableMatch;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn silo_id(&self) -> &str {
        &self.silo_id
    }
}

impl Storable for TableDef {
    type SearchResult = TableMatch;

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn silo_id(&self) -> &str {
        &self.silo_id
    }

    fn schema() -> std::sync::Arc<arrow::datatypes::Schema> {
        let field_defs = Fields::from(get_field_defs());
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                true,
            ),
            Field::new("fields", DataType::Struct(field_defs), false),
        ]))
    }

    fn vtable_name() -> &'static str {
        "table_def"
    }

    fn embedding(&self, config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NisabaError> {
        let field_count = self.fields.len();

        // Field embedding average
        let mut field_embed = Vector1::from_iterator([0.0; 384]);
        let _ = self.fields.iter().map(|f| {
            field_embed += Vector1::from_vec(f.embedding(config.clone()).unwrap());
        });

        // Field Embedding
        let field_embed = field_embed / field_count as f32;

        // Table Stats Embedding
        let structure_embed = self.structure().embedding();

        // Combined embedding
        let combnd = (config.name_weight + config.type_weight) * field_embed
            + config.structure_weight * structure_embed;

        Ok(combnd.as_slice().to_vec())
    }

    fn to_record_batch(
        items: &[Self],
        schema: Arc<Schema>,
        config: Arc<AnalyzerConfig>,
    ) -> Result<RecordBatch, NisabaError>
    where
        Self: std::marker::Sized,
    {
        let ids = FixedSizeBinaryArray::try_from_iter(items.iter().map(|t| t.id.into_bytes()))?;

        let names = StringArray::from(
            items
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<String>>(),
        );

        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            items.iter().map(|item| {
                let embeeding: Vec<Option<f32>> = item
                    .embedding(config.clone())
                    .unwrap()
                    .iter()
                    .map(|&v| Some(v))
                    .collect();

                Some(embeeding.into_iter())
            }),
            384,
        );

        let fields = build_fields_array(items, config)?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(names), Arc::new(vectors), fields],
        )
        .unwrap();

        Ok(batch)
    }

    fn from_record_batches(
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<Self::SearchResult>, NisabaError> {
        let mut schemas = Vec::new();

        for batch in batches {
            // Get id values
            let id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or(ArrowError::CastError("Failed to downcast id column".into()))?;

            // Get silo_id values
            let silo_id_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast silo id column".into(),
                ))?;

            // Get name values
            let name_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast name column".into(),
                ))?;

            let fields_list = batch
                .column(3)
                .as_any()
                .downcast_ref::<GenericListArray<i64>>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast fields column".into(),
                ))?;

            let distances = batch
                .column(4)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast distance column".into(),
                ))?;

            for row_idx in 0..batch.num_rows() {
                let id = Uuid::from_slice(id_array.value(row_idx))?;
                let silo_id = silo_id_array.value(row_idx).to_string();

                let name = name_array.value(row_idx).to_string();

                let field_structs = fields_list.value(0);
                let field_structs = field_structs.as_any().downcast_ref::<StructArray>().ok_or(
                    ArrowError::CastError("Failed to downcast fields struct".into()),
                )?;

                let fields = extract_field_defs(field_structs)?;

                let confidence = distances.value(row_idx);

                let schema = TableDef {
                    id,
                    silo_id,
                    name,
                    fields,
                };

                schemas.push(TableMatch { schema, confidence });
            }
        }

        Ok(schemas)
    }
}

impl TableDef {
    fn structure(&self) -> TableStats {
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

        structure
    }
}

fn build_fields_array(
    tbl_schema: &[TableDef],
    config: Arc<AnalyzerConfig>,
) -> Result<ArrayRef, NisabaError> {
    // N number of fileds
    let capacity: usize = tbl_schema.iter().map(|s| s.fields.len()).sum();

    // Allocate with known capacity
    let mut ids = Vec::with_capacity(capacity * 16);

    let mut silo_ids = Vec::with_capacity(capacity);
    let mut names = Vec::with_capacity(capacity);
    let mut table_names = Vec::with_capacity(capacity);
    let mut canonical_types = Vec::with_capacity(capacity);

    // For cardinality
    let mut cardinalities = Vec::with_capacity(capacity);
    let mut cardinality_nulls = Vec::with_capacity(capacity);

    // For embeddings
    let mut embeding_values = Vec::new();
    let mut embedding_offsets = Vec::with_capacity(capacity);
    embedding_offsets.push(0i64);
    let mut embedding_nulls = Vec::with_capacity(capacity);

    // Outer list offsets for fields list
    let mut fields_offsets = Vec::with_capacity(tbl_schema.len() + 1);
    fields_offsets.push(0i64);

    let mut current_field_offset = 0i64;

    for scheme in tbl_schema {
        for field in &scheme.fields {
            ids.push(field.id.into_bytes());
            silo_ids.push(field.silo_id.clone());
            names.push(field.name.clone());
            table_names.push(field.table_name.clone());
            canonical_types.push(field.canonical_type.to_string());

            // Cardinality
            cardinalities.push(field.cardinality.unwrap_or_default());
            cardinality_nulls.push(field.cardinality.is_none());

            // Embeddding
            embeding_values.extend_from_slice(&field.embedding(config.clone())?);
            embedding_offsets.push(embeding_values.len() as i64);
            embedding_nulls.push(false);

            current_field_offset += 1;
        }

        fields_offsets.push(current_field_offset);
    }

    // Build arrays from collected data
    let ids_array = FixedSizeBinaryArray::try_from_iter(ids.iter())?;
    let silo_id_array = StringArray::from(silo_ids);
    let name_array = StringArray::from(names);
    let table_name_array = StringArray::from(table_names);
    let canonical_type_array = StringArray::from(canonical_types);

    // CardinalityArray
    let cardinality_nulls_buffer = NullBuffer::from(cardinality_nulls);
    let cardinality_array = Float32Array::new(
        ScalarBuffer::from(cardinalities),
        Some(cardinality_nulls_buffer),
    );

    // Embedding Arrays
    let embedding_values_array = Float32Array::from(embeding_values);
    let embedding_nulls_buffer = NullBuffer::from(embedding_nulls);
    let embedding_list = GenericListArray::<i64>::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        OffsetBuffer::new(ScalarBuffer::from(embedding_offsets)),
        Arc::new(embedding_values_array),
        Some(embedding_nulls_buffer),
    )?;

    let struct_array = StructArray::try_new(
        get_field_defs().into(),
        vec![
            Arc::new(ids_array),
            Arc::new(silo_id_array),
            Arc::new(name_array),
            Arc::new(table_name_array),
            Arc::new(canonical_type_array),
            Arc::new(cardinality_array),
            Arc::new(embedding_list),
        ],
        None,
    )?;

    let fields_list = GenericListArray::<i64>::try_new(
        Arc::new(Field::new(
            "item",
            DataType::Struct(get_field_defs().into()),
            false,
        )),
        OffsetBuffer::new(ScalarBuffer::from(fields_offsets)),
        Arc::new(struct_array),
        None,
    )?;

    Ok(Arc::new(fields_list))
}

fn extract_field_defs(struct_array: &StructArray) -> Result<Vec<FieldDef>, NisabaError> {
    let num_fields = struct_array.len();
    let mut field_defs = Vec::with_capacity(num_fields);

    // Extract all columns
    let id_col = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or(ArrowError::CastError(
            "Failed to downcast id (Uuid) column".into(),
        ))?;

    let silo_id_col = struct_array.column(1).as_string::<i64>();
    let name_col = struct_array.column(2).as_string::<i64>();
    let table_name_col = struct_array.column(3).as_string::<i64>();
    let canonical_type_col = struct_array.column(4).as_string::<i64>();

    let type_confidence_col = struct_array.column(5).as_primitive::<Float32Type>();
    let cardinality_col = struct_array.column(6).as_primitive::<Float32Type>();

    let avg_byte_len_col = struct_array.column(7).as_primitive::<Float32Type>();
    let is_monotonic_col = struct_array.column(8).as_boolean();

    let char_class_signature_col = struct_array.column(9).as_fixed_size_list();
    let column_default_col = struct_array.column(10).as_string::<i64>();
    let is_nullable_col = struct_array.column(11).as_boolean();

    let char_max_len_col = struct_array.column(12).as_primitive::<Int32Type>();

    let numeric_precision_col = struct_array.column(13).as_primitive::<Int32Type>();

    let numeric_scale_col = struct_array.column(14).as_primitive::<Int32Type>();

    let datetime_precision_col = struct_array.column(15).as_primitive::<Int32Type>();

    for i in 0..num_fields {
        let id = Uuid::from_slice(id_col.value(i))?;
        let silo_id = silo_id_col.value(i).to_string();
        let table_name = table_name_col.value(i).to_string();
        let name = name_col.value(i).to_string();
        let canonical_type: DataType = canonical_type_col.value(i).parse()?;

        let type_confidence = if type_confidence_col.is_null(i) {
            None
        } else {
            Some(type_confidence_col.value(i))
        };

        let cardinality = if cardinality_col.is_null(i) {
            None
        } else {
            Some(cardinality_col.value(i))
        };

        let avg_byte_length = if avg_byte_len_col.is_null(i) {
            None
        } else {
            Some(avg_byte_len_col.value(i))
        };

        let is_monotonic = is_monotonic_col.value(i);

        let char_class_signature = if char_class_signature_col.is_null(i) {
            None
        } else {
            let values = char_class_signature_col.value(i);
            let values = values.as_primitive::<Float32Type>();

            Some([
                values.value(0),
                values.value(1),
                values.value(2),
                values.value(3),
            ])
        };

        let column_default = if column_default_col.is_null(i) {
            None
        } else {
            Some(column_default_col.value(i).to_owned())
        };

        let is_nullable = is_nullable_col.value(i);

        let char_max_length = if char_max_len_col.is_null(i) {
            None
        } else {
            Some(char_max_len_col.value(i))
        };

        let numeric_precision = if numeric_precision_col.is_null(i) {
            None
        } else {
            Some(numeric_precision_col.value(i))
        };

        let numeric_scale = if numeric_scale_col.is_null(i) {
            None
        } else {
            Some(numeric_scale_col.value(i))
        };

        let datetime_precision = if datetime_precision_col.is_null(i) {
            None
        } else {
            Some(datetime_precision_col.value(i))
        };

        field_defs.push(FieldDef {
            id,
            silo_id,
            name,
            table_name,
            canonical_type,
            type_confidence,
            cardinality,
            avg_byte_length,
            is_monotonic,
            char_class_signature,
            column_default,
            is_nullable,
            char_max_length,
            numeric_precision,
            numeric_scale,
            datetime_precision,
        });
    }

    Ok(field_defs)
}

#[derive(Debug)]
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
    fn embedding(&self) -> DVector<f32> {
        let raw_embed = Vector1::from_vec(vec![
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
        deterministic_projection(raw_embed, 384, 42)
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
