use arrow::{
    array::{
        Array, AsArray, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, GenericListArray,
        RecordBatch, StringArray, UInt64Array,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Float32Type, Schema, UInt64Type},
    error::ArrowError,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::{collections::HashSet, str::FromStr, sync::Arc};
use uuid::Uuid;

use crate::{
    analyzer::{
        AnalyzerConfig,
        report::{ClusterDef, FieldMatch},
        retriever::Storable,
    },
    error::NError,
    types::{Matchable, extract_metadata, extract_sample_values},
};

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub id: Uuid,
    pub silo_id: String,
    pub table_name: String,
    pub name: String,
    pub canonical_type: DataType,
    pub metadata: HashSet<Option<String>>,
    pub sample_values: [Option<String>; 17],
    pub cardinality: Option<u64>,
}

impl std::fmt::Display for FieldDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let m = self
            .metadata
            .iter()
            .map(|n| n.to_owned())
            .collect::<Vec<Option<String>>>()
            .iter()
            .map(|s| s.clone().unwrap_or_default())
            .collect::<Vec<String>>()
            .join(" ");
        write!(
            f,
            "table name: {} field name: {} data type: {} metadata: {}, cardinality: {}",
            self.table_name,
            self.name,
            self.canonical_type,
            m,
            self.cardinality.unwrap_or_default()
        )
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

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
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
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
                true,
            ),
        ]))
    }

    fn vtable_name() -> &'static str {
        "field_def"
    }

    fn embedding(&self, _config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NError> {
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
    ) -> Result<RecordBatch, NError>
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

        // For metadata
        let mut metadata_values = Vec::new();
        let mut metadata_offsets = Vec::with_capacity(capacity + 1);
        metadata_offsets.push(0i64);

        // For sample values(fixed size list of 17)
        let mut sample_values = Vec::with_capacity(capacity * 17);
        let mut sample_nulls = vec![false; capacity * 17];

        // For cardinality
        let mut cardinalities = Vec::with_capacity(capacity);
        let mut cardinality_nulls = Vec::with_capacity(capacity);

        for field in items {
            // Metadata list
            for item in &field.metadata {
                metadata_values.push(item.clone());
            }
            metadata_offsets.push(metadata_values.len() as i64);

            // Sample values
            let start_idx = sample_values.len();
            for (i, sample) in field.sample_values.iter().enumerate() {
                sample_values.push(sample.clone());
                sample_nulls[start_idx + i] = sample.is_none();
            }

            // Cardinality
            cardinalities.push(field.cardinality.unwrap_or_default());
            cardinality_nulls.push(field.cardinality.is_none());
        }

        let metadata_values_array = StringArray::from(metadata_values);
        let metadata_list = GenericListArray::<i64>::try_new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            OffsetBuffer::new(ScalarBuffer::from(metadata_offsets)),
            Arc::new(metadata_values_array),
            None,
        )?;

        // SamplesArray
        let samples_values_array = StringArray::from(sample_values);
        let sample_nulls_buffer = NullBuffer::from(sample_nulls);
        let samples_list = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Utf8, true)),
            17,
            Arc::new(samples_values_array),
            Some(sample_nulls_buffer),
        )?;

        // CardinalityArray
        let cardinality_nulls_buffer = NullBuffer::from(cardinality_nulls);
        let cardinality_array = UInt64Array::new(
            ScalarBuffer::from(cardinalities),
            Some(cardinality_nulls_buffer),
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
                Arc::new(metadata_list),
                Arc::new(samples_list),
                Arc::new(cardinality_array),
                Arc::new(vectors),
            ],
        )
        .unwrap();

        Ok(batch)
    }

    fn from_record_batches(batches: Vec<RecordBatch>) -> Result<Vec<Self::SearchResult>, NError> {
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

            let metadata_array = batch.column(5).as_list::<i64>();

            let sample_values_array = batch.column(6).as_fixed_size_list();

            let cardinality_array = batch.column(7).as_primitive::<UInt64Type>();

            let distances = batch
                .column(8)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or(ArrowError::CastError(
                    "Failed to downcast distance column".into(),
                ))?;

            for row_idx in 0..batch.num_rows() {
                let id = Uuid::try_parse_ascii(ids_array.value(row_idx))?;
                let silo_id = silo_id_array.value(row_idx).to_string();

                let table_name = table_name_array.value(row_idx).to_string();

                let name = name_array.value(row_idx).to_string();

                let canonical_type = DataType::from_str(canonical_type_array.value(row_idx))?;

                let metadata = extract_metadata(metadata_array, row_idx)?;

                let sample_values = extract_sample_values(sample_values_array, row_idx)?;

                let cardinality = if cardinality_array.is_null(row_idx) {
                    None
                } else {
                    Some(cardinality_array.value(row_idx))
                };

                let confidence = distances.value(row_idx);

                let schema = FieldDef {
                    id,
                    silo_id,
                    table_name,
                    name,
                    canonical_type,
                    metadata,
                    sample_values,
                    cardinality,
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
