use arrow::{
    array::{
        Array, ArrayRef, AsArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray,
        Float32Array, Int32Array, ListArray, RecordBatch, StringArray, StructArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Fields, Float32Type, Int32Type, Schema},
    error::ArrowError,
};
use nalgebra::{DVector, SVector};
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

    fn name(&self) -> &String {
        &self.name
    }

    fn schema() -> std::sync::Arc<arrow::datatypes::Schema> {
        let field_defs = Fields::from(get_field_defs());
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::FixedSizeBinary(16), false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                true,
            ),
            Field::new(
                "fields",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(field_defs),
                    false,
                ))),
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

    fn embedding(&self, config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NisabaError> {
        let field_count = self.fields.len();

        // Field embedding average
        let mut field_embed: SVector<f32, 384> = SVector::<f32, 384>::zeros();

        let _: Result<Vec<()>, NisabaError> = self
            .fields
            .iter()
            .map(|f| {
                let embed = f.embedding(config.clone())?;
                let vector: SVector<f32, 384> = SVector::from_vec(embed);
                field_embed += vector;

                Ok(())
            })
            .collect();

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

        let silo_ids = StringArray::from(
            items
                .iter()
                .map(|t| t.silo_id.clone())
                .collect::<Vec<String>>(),
        );
        let names = StringArray::from(
            items
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<String>>(),
        );

        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            items.iter().map(|item| {
                let embedding: Vec<Option<f32>> = item
                    .embedding(config.clone())
                    .unwrap()
                    .iter()
                    .map(|&v| Some(v))
                    .collect();

                Some(embedding)
            }),
            384,
        );

        let fields = build_fields_array(items, config)?;

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(silo_ids),
                Arc::new(names),
                Arc::new(vectors),
                fields,
            ],
        )?;

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

            let fields_list = batch.column(3).as_any().downcast_ref::<ListArray>().ok_or(
                ArrowError::CastError("Failed to downcast fields column".into()),
            )?;

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
    /// The `structure` function profiles the fields of the TableDef by count and DataType
    /// for linear projection purposes
    ///
    /// Returns
    ///
    /// A `TableStats`
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

        structure.num_fields = self.fields.len() as f32;

        structure
    }
}

/// The `build_fields_array` function takes
///
/// Arguments:
///
/// * `tbl_schema`: The `tbl_schema` parameter is a slice of TableDef
///
/// Returns:
///
/// A `Result` of ArrayRef (implements arrow's Array trait behind Arc pointer)
fn build_fields_array(
    tbl_schema: &[TableDef],
    config: Arc<AnalyzerConfig>,
) -> Result<ArrayRef, NisabaError> {
    // N number of fileds
    let capacity: usize = tbl_schema.iter().map(|s| s.fields.len()).sum();

    // Allocate with known capacity
    let mut ids = Vec::with_capacity(capacity * 16);

    let mut silo_ids = Vec::with_capacity(capacity);
    let mut table_names = Vec::with_capacity(capacity);
    let mut names = Vec::with_capacity(capacity);
    let mut canonical_types = Vec::with_capacity(capacity);

    // Type Confidence
    let mut type_confidences = Vec::with_capacity(capacity);

    // For cardinality
    let mut cardinalities = Vec::with_capacity(capacity);

    // Avg_byte_length
    let mut avg_byte_lengths = Vec::with_capacity(capacity);

    // Monotonicity
    let mut is_monotonics = Vec::with_capacity(capacity);

    // Char class signature
    let mut char_class_values = Vec::with_capacity(capacity * 4);
    let mut char_class_offsets = Vec::with_capacity(capacity);
    char_class_offsets.push(0i32);
    let mut char_class_nulls = Vec::with_capacity(capacity);

    // Column defaults
    let mut column_defaults = Vec::with_capacity(capacity);

    // Nullable
    let mut is_nullables = Vec::with_capacity(capacity);

    // Char max length
    let mut char_max_lengths = Vec::with_capacity(capacity);

    // Numeric precision
    let mut numeric_precisions = Vec::with_capacity(capacity);

    // Numeric scale
    let mut numeric_scales = Vec::with_capacity(capacity);

    // Datetime precision
    let mut datetime_precisions = Vec::with_capacity(capacity);

    // For embeddings
    let mut embedding_values = Vec::with_capacity(capacity * 384);

    // Outer list offsets for fields list
    let mut fields_offsets = Vec::with_capacity(tbl_schema.len() + 1);
    fields_offsets.push(0i32);

    let mut current_field_offset = 0i32;

    for scheme in tbl_schema {
        for field in &scheme.fields {
            ids.push(field.id.into_bytes());
            silo_ids.push(field.silo_id.clone());
            table_names.push(field.table_name.clone());
            names.push(field.name.clone());
            canonical_types.push(field.canonical_type.to_string());

            // Type confidence
            type_confidences.push(field.type_confidence);

            // Cardinality
            cardinalities.push(field.cardinality);

            // Avg Byte Length
            avg_byte_lengths.push(field.avg_byte_length);

            // Monotonicity
            is_monotonics.push(field.is_monotonic);

            // Char class signature
            if let Some(ccs) = field.char_class_signature {
                char_class_values.extend_from_slice(&ccs);
                char_class_offsets.push(char_class_values.len() as i32);
                char_class_nulls.push(true);
            } else {
                char_class_nulls.push(false);
                char_class_offsets.push(char_class_values.len() as i32);
            }

            // Column default
            column_defaults.push(field.column_default.clone());

            // Nullable
            is_nullables.push(field.is_nullable);

            // Char max length
            char_max_lengths.push(field.char_max_length);

            // Numeric precision
            numeric_precisions.push(field.numeric_precision);

            // Numeric scale
            numeric_scales.push(field.numeric_scale);

            // Datetime precision
            datetime_precisions.push(field.datetime_precision);

            // Embeddding
            embedding_values.extend_from_slice(&field.embedding(config.clone())?);

            current_field_offset += 1;
        }

        fields_offsets.push(current_field_offset);
    }

    // Build arrays from collected data
    let ids_array = FixedSizeBinaryArray::try_from_iter(ids.iter())?;
    let silo_id_array = StringArray::from(silo_ids);
    let table_name_array = StringArray::from(table_names);
    let name_array = StringArray::from(names);
    let canonical_type_array = StringArray::from(canonical_types);

    // Type confidence
    let type_confidence_array = Float32Array::from(type_confidences);

    // CardinalityArray
    let cardinality_array = Float32Array::from(cardinalities);

    // Avg byte length
    let avg_byte_len_array = Float32Array::from(avg_byte_lengths);

    // Monotonicity
    let is_monotonic_array = BooleanArray::from(is_monotonics);

    // Char class signature
    let char_class_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        OffsetBuffer::new(ScalarBuffer::from(char_class_offsets)),
        Arc::new(Float32Array::from(char_class_values)),
        Some(NullBuffer::from(char_class_nulls)),
    );

    // column default
    let column_default_array = StringArray::from(column_defaults);

    // Nullability
    let is_nullable_array = BooleanArray::from(is_nullables);

    // Char max length
    let char_max_length_array = Int32Array::from(char_max_lengths);

    // Numeric precision
    let numeric_precision_array = Int32Array::from(numeric_precisions);

    // Numeric scale
    let numeric_scale_array = Int32Array::from(numeric_scales);

    // Datetime precision
    let datetime_precision_array = Int32Array::from(datetime_precisions);

    // Embedding Array
    let embed_value_data = Float32Array::from(embedding_values);
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let embed_array = FixedSizeListArray::new(field, 384, Arc::new(embed_value_data), None);

    let struct_array = StructArray::try_new(
        get_field_defs().into(),
        vec![
            Arc::new(ids_array),
            Arc::new(silo_id_array),
            Arc::new(table_name_array),
            Arc::new(name_array),
            Arc::new(canonical_type_array),
            Arc::new(type_confidence_array),
            Arc::new(cardinality_array),
            Arc::new(avg_byte_len_array),
            Arc::new(is_monotonic_array),
            Arc::new(char_class_array),
            Arc::new(column_default_array),
            Arc::new(is_nullable_array),
            Arc::new(char_max_length_array),
            Arc::new(numeric_precision_array),
            Arc::new(numeric_scale_array),
            Arc::new(datetime_precision_array),
            Arc::new(embed_array),
        ],
        None,
    )?;

    let fields_list = ListArray::try_new(
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

/// The `extract_field_defs` functions processes a StructArray and returns a Vec of FieldDef
/// This is essential when reading TableDefs from latent store
///
/// Arguments:
///
/// *`struct_array`: The `struct_array` parameter is a reference that constitutes values
///   and fields associated with TableDefs
///
/// Returns:
///
/// A Vec of FieldDef
fn extract_field_defs(struct_array: &StructArray) -> Result<Vec<FieldDef>, NisabaError> {
    let num_fields = struct_array.len();
    let mut field_defs = Vec::with_capacity(num_fields);

    // Extract all columns
    let id_array = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or(ArrowError::CastError(
            "Failed to downcast id (Uuid) column".into(),
        ))?;

    let silo_id_array = struct_array.column(1).as_string::<i32>();

    let table_schema_array = struct_array.column(2).as_string::<i32>();

    let table_name_array = struct_array.column(3).as_string::<i32>();

    let name_array = struct_array.column(4).as_string::<i32>();

    let canonical_type_array = struct_array.column(5).as_string::<i32>();

    let type_confidence_array = struct_array.column(6).as_primitive::<Float32Type>();

    let cardinality_array = struct_array.column(7).as_primitive::<Float32Type>();

    let avg_byte_len_array = struct_array.column(8).as_primitive::<Float32Type>();

    let is_monotonic_array = struct_array.column(9).as_boolean();

    let char_class_signature_array = struct_array.column(10).as_list::<i32>();

    let column_default_array = struct_array.column(11).as_string::<i32>();

    let is_nullable_array = struct_array.column(12).as_boolean();

    let char_max_len_array = struct_array.column(13).as_primitive::<Int32Type>();

    let numeric_precision_array = struct_array.column(14).as_primitive::<Int32Type>();

    let numeric_scale_array = struct_array.column(15).as_primitive::<Int32Type>();

    let datetime_precision_array = struct_array.column(16).as_primitive::<Int32Type>();

    for i in 0..num_fields {
        let id = Uuid::from_slice(id_array.value(i))?;
        let silo_id = silo_id_array.value(i).to_string();
        let table_schema = table_schema_array.value(i).to_string();
        let table_name = table_name_array.value(i).to_string();
        let name = name_array.value(i).to_string();
        let canonical_type: DataType = canonical_type_array.value(i).parse()?;

        let type_confidence = if type_confidence_array.is_null(i) {
            None
        } else {
            Some(type_confidence_array.value(i))
        };

        let cardinality = if cardinality_array.is_null(i) {
            None
        } else {
            Some(cardinality_array.value(i))
        };

        let avg_byte_length = if avg_byte_len_array.is_null(i) {
            None
        } else {
            Some(avg_byte_len_array.value(i))
        };

        let is_monotonic = is_monotonic_array.value(i);

        let char_class_signature = if char_class_signature_array.is_null(i) {
            None
        } else {
            let values = char_class_signature_array.value(i);
            let values = values.as_primitive::<Float32Type>();

            Some([
                values.value(0),
                values.value(1),
                values.value(2),
                values.value(3),
            ])
        };

        let column_default = if column_default_array.is_null(i) {
            None
        } else {
            Some(column_default_array.value(i).to_owned())
        };

        let is_nullable = is_nullable_array.value(i);

        let char_max_length = if char_max_len_array.is_null(i) {
            None
        } else {
            Some(char_max_len_array.value(i))
        };

        let numeric_precision = if numeric_precision_array.is_null(i) {
            None
        } else {
            Some(numeric_precision_array.value(i))
        };

        let numeric_scale = if numeric_scale_array.is_null(i) {
            None
        } else {
            Some(numeric_scale_array.value(i))
        };

        let datetime_precision = if datetime_precision_array.is_null(i) {
            None
        } else {
            Some(datetime_precision_array.value(i))
        };

        field_defs.push(FieldDef {
            id,
            silo_id,
            table_name,
            name,
            table_schema,
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
    fn embedding(&self) -> DVector<f32> {
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
