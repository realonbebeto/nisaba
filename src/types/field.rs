use arrow::{
    array::{
        Array, AsArray, BooleanArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array,
        Int32Array, ListArray, RecordBatch, StringArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Float32Type, Int32Type, Schema},
    error::ArrowError,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::{str::FromStr, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        AnalyzerConfig,
        inference::FieldMetrics,
        report::{ClusterDef, FieldMatch},
        retriever::Storable,
    },
    error::NisabaError,
    types::Matchable,
};

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub id: Uuid,
    pub silo_id: String,
    pub table_name: String,
    pub name: String,
    pub canonical_type: DataType,
    pub type_confidence: Option<f32>,
    pub cardinality: Option<f32>,
    pub avg_byte_length: Option<f32>,
    pub is_monotonic: bool,
    pub char_class_signature: Option<[f32; 4]>, // [digit, alpha, whitespace, symbol]
    pub column_default: Option<String>,
    pub is_nullable: bool,
    pub char_max_length: Option<i32>,
    pub numeric_precision: Option<i32>,
    pub numeric_scale: Option<i32>,
    pub datetime_precision: Option<i32>,
}

impl std::fmt::Display for FieldDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "table name: {} field name: {} data type: {}, cardinality: {}",
            self.table_name,
            self.name,
            self.canonical_type,
            self.cardinality.unwrap_or_default()
        )
    }
}

impl FieldDef {
    pub fn enrich_from_arrow(&mut self, metrics: Option<&FieldMetrics>) {
        if let Some(m) = metrics {
            self.char_class_signature = Some(m.char_class_signature);
            self.is_monotonic = m.monotonicity;
            self.cardinality = Some(m.cardinality);
            self.avg_byte_length = m.avg_byte_length;
        }
    }
}

impl ClusterDef for FieldDef {
    fn name(&self) -> &str {
        &self.name
    }
    fn table_name(&self) -> &str {
        &self.table_name
    }
}

impl Matchable for FieldDef {
    type Id = Uuid;
    type Match = FieldMatch;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn silo_id(&self) -> &str {
        &self.silo_id
    }
}

impl Storable for FieldDef {
    type SearchResult = FieldMatch;

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn silo_id(&self) -> &str {
        &self.silo_id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::FixedSizeBinary(16), false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("canonical_type", DataType::Utf8, false),
            Field::new("type_confidence", DataType::Float32, true),
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
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                true,
            ),
        ]))
    }

    fn result_columns() -> Vec<String> {
        vec![
            "id".to_string(),
            "silo_id".to_string(),
            "table_name".to_string(),
            "name".to_string(),
            "canonical_type".to_string(),
            "type_confidence".to_string(),
            "cardinality".to_string(),
            "avg_byte_length".to_string(),
            "is_monotonic".to_string(),
            "char_class_signature".to_string(),
            "column_default".to_string(),
            "is_nullable".to_string(),
            "char_max_length".to_string(),
            "numeric_precision".to_string(),
            "numeric_scale".to_string(),
            "datetime_precision".to_string(),
            "_distance".to_string(),
        ]
    }

    fn vtable_name() -> &'static str {
        "field_def"
    }

    fn embedding(&self, _config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NisabaError> {
        let mut model =
            TextEmbedding::try_new(InitOptions::new(EmbeddingModel::MultilingualE5Small))?;

        // TODO Sample Embedding is part of field on embed
        let embeddings = &model.embed([self.value()], None)?[0];
        let embeddings = embeddings.to_owned();

        Ok(embeddings)
    }

    fn to_record_batch(
        items: &[Self],
        schema: Arc<Schema>,
        config: Arc<AnalyzerConfig>,
    ) -> Result<RecordBatch, NisabaError>
    where
        Self: std::marker::Sized,
    {
        let ids = FixedSizeBinaryArray::try_from_iter(items.iter().map(|f| f.id.into_bytes()))?;

        let silo_ids = StringArray::from(
            items
                .iter()
                .map(|f| f.silo_id.clone())
                .collect::<Vec<String>>(),
        );

        let table_names = StringArray::from(
            items
                .iter()
                .map(|f| f.table_name.clone())
                .collect::<Vec<String>>(),
        );

        let names = StringArray::from(
            items
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<String>>(),
        );

        let canonical_types = StringArray::from(
            items
                .iter()
                .map(|f| f.canonical_type.to_string())
                .collect::<Vec<String>>(),
        );

        let capacity = items.len();

        // For type confidence
        let type_confidence_array = Float32Array::from(
            items
                .iter()
                .map(|f| f.type_confidence)
                .collect::<Vec<Option<f32>>>(),
        );

        // For cardinality
        let cardinalities_array = Float32Array::from(
            items
                .iter()
                .map(|f| f.cardinality)
                .collect::<Vec<Option<f32>>>(),
        );

        // For average byte length
        let avg_byte_lens_array = Float32Array::from(
            items
                .iter()
                .map(|f| f.avg_byte_length)
                .collect::<Vec<Option<f32>>>(),
        );

        // For monotonicity
        let monotonic_flag_array =
            BooleanArray::from(items.iter().map(|f| f.is_monotonic).collect::<Vec<bool>>());

        // For char signature
        let mut char_class_values = Vec::with_capacity(capacity * 4);
        let mut char_class_offsets = Vec::with_capacity(capacity);
        char_class_offsets.push(0i32);
        let mut char_class_nulls = Vec::with_capacity(capacity);

        // Column Defaults
        let column_defaults_array = StringArray::from(
            items
                .iter()
                .map(|f| f.column_default.clone())
                .collect::<Vec<Option<String>>>(),
        );

        let nullable_array =
            BooleanArray::from(items.iter().map(|f| f.is_nullable).collect::<Vec<bool>>());

        let char_max_lengths_array = Int32Array::from(
            items
                .iter()
                .map(|f| f.char_max_length)
                .collect::<Vec<Option<i32>>>(),
        );

        let numeric_precision_array = Int32Array::from(
            items
                .iter()
                .map(|f| f.numeric_precision)
                .collect::<Vec<Option<i32>>>(),
        );

        let numeric_scale_array = Int32Array::from(
            items
                .iter()
                .map(|f| f.numeric_scale)
                .collect::<Vec<Option<i32>>>(),
        );

        let datetime_precision_array = Int32Array::from(
            items
                .iter()
                .map(|f| f.datetime_precision)
                .collect::<Vec<Option<i32>>>(),
        );

        for field in items {
            // Char class signature
            if let Some(ccs) = field.char_class_signature {
                char_class_values.extend_from_slice(&ccs);
                char_class_offsets.push(char_class_values.len() as i32);
                char_class_nulls.push(true);
            } else {
                char_class_nulls.push(false);
                char_class_offsets.push(char_class_values.len() as i32);
            }
        }

        let char_class_array = ListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            OffsetBuffer::new(ScalarBuffer::from(char_class_offsets)),
            Arc::new(Float32Array::from(char_class_values)),
            Some(NullBuffer::from(char_class_nulls)),
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

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(silo_ids),
                Arc::new(table_names),
                Arc::new(names),
                Arc::new(canonical_types),
                Arc::new(type_confidence_array),
                Arc::new(cardinalities_array),
                Arc::new(avg_byte_lens_array),
                Arc::new(monotonic_flag_array),
                Arc::new(char_class_array),
                Arc::new(column_defaults_array),
                Arc::new(nullable_array),
                Arc::new(char_max_lengths_array),
                Arc::new(numeric_precision_array),
                Arc::new(numeric_scale_array),
                Arc::new(datetime_precision_array),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        Ok(batch)
    }

    fn from_record_batches(
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<Self::SearchResult>, NisabaError> {
        let mut schemas = Vec::new();

        for batch in batches {
            let ids_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcat id (Uuid) column".into(),
                ))?;

            // Get silo_id values
            let silo_id_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast silo id column".into(),
                ))?;

            // Get table name values
            let table_name_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast table_name column".into(),
                ))?;

            // Get name values
            let name_array = batch
                .column(3)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast name column".into(),
                ))?;

            // Get canonical_type values
            let canonical_type_array = batch
                .column(4)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast canonical_type column".into(),
                ))?;

            //Type confidence
            let type_confidence_array = batch
                .column(5)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast type_confidence column".into(),
                ))?;

            // Cardinality
            let cardinality_array = batch.column(6).as_primitive::<Float32Type>();

            // Avg Byte Len
            let avg_byte_len_array = batch.column(7).as_primitive::<Float32Type>();

            // Monotonicity
            let monotonicity_array =
                batch
                    .column(8)
                    .as_boolean_opt()
                    .ok_or(ArrowError::CastError(
                        "Failed to downcast monotonicity".into(),
                    ))?;

            // Class signature
            let class_signature_array =
                batch
                    .column(9)
                    .as_list_opt::<i32>()
                    .ok_or(ArrowError::CastError(
                        "Failed to downcast class signature".into(),
                    ))?;

            // Column Default
            let column_default_array = batch.column(10).as_string::<i32>();

            // Nullable
            let nullable_array = batch
                .column(11)
                .as_boolean_opt()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast is nullable column".into(),
                ))?;

            // Char Max Length
            let char_max_len_array = batch.column(12).as_primitive::<Int32Type>();

            // Numeric precision
            let numeric_precision_array = batch.column(13).as_primitive::<Int32Type>();

            // Numeric scale
            let numeric_scale_array = batch.column(14).as_primitive::<Int32Type>();

            // Datetime precision
            let datetime_precision_array = batch.column(15).as_primitive::<Int32Type>();

            let distances = batch
                .column(16)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast distance column".into(),
                ))?;

            for row_idx in 0..batch.num_rows() {
                let id = Uuid::from_slice(ids_array.value(row_idx))?;
                let silo_id = silo_id_array.value(row_idx).to_string();

                let table_name = table_name_array.value(row_idx).to_string();

                let name = name_array.value(row_idx).to_string();

                let canonical_type = DataType::from_str(canonical_type_array.value(row_idx))?;

                let type_confidence = if type_confidence_array.is_null(row_idx) {
                    None
                } else {
                    Some(type_confidence_array.value(row_idx))
                };

                let cardinality = if cardinality_array.is_null(row_idx) {
                    None
                } else {
                    Some(cardinality_array.value(row_idx))
                };

                let avg_byte_length = if avg_byte_len_array.is_null(row_idx) {
                    None
                } else {
                    Some(avg_byte_len_array.value(row_idx))
                };

                let is_monotonic = monotonicity_array.value(row_idx);

                let char_class_signature = if class_signature_array.is_null(row_idx) {
                    None
                } else {
                    let values = class_signature_array.value(row_idx);
                    let values = values.as_primitive::<Float32Type>();

                    Some([
                        values.value(0),
                        values.value(1),
                        values.value(2),
                        values.value(3),
                    ])
                };

                let column_default = if column_default_array.is_null(row_idx) {
                    None
                } else {
                    Some(column_default_array.value(row_idx).to_owned())
                };

                let is_nullable = nullable_array.value(row_idx);

                let char_max_length = if char_max_len_array.is_null(row_idx) {
                    None
                } else {
                    Some(char_max_len_array.value(row_idx))
                };

                let numeric_precision = if numeric_precision_array.is_null(row_idx) {
                    None
                } else {
                    Some(numeric_precision_array.value(row_idx))
                };

                let numeric_scale = if numeric_scale_array.is_null(row_idx) {
                    None
                } else {
                    Some(numeric_scale_array.value(row_idx))
                };

                let datetime_precision = if datetime_precision_array.is_null(row_idx) {
                    None
                } else {
                    Some(datetime_precision_array.value(row_idx))
                };

                let confidence = distances.value(row_idx);

                let schema = FieldDef {
                    id,
                    silo_id,
                    table_name,
                    name,
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
                };

                schemas.push(FieldMatch { schema, confidence });
            }
        }

        Ok(schemas)
    }
}

impl FieldDef {
    fn value(&self) -> String {
        format!("{}", self)
    }
}
