use arrow::{
    array::{RecordBatch, RecordBatchIterator},
    datatypes::Schema,
};
use futures::TryStreamExt;
use lancedb::{
    Connection, DistanceType,
    database::CreateTableMode,
    index::{Index, vector::IvfHnswPqIndexBuilder},
    query::{ExecutableQuery, QueryBase, Select},
};
use std::{marker::PhantomData, path::PathBuf, sync::Arc};
use uuid::Uuid;

use crate::{analyzer::AnalyzerConfig, error::NError};

pub trait Storable: Send + Sync {
    /// The result type returned from similarity search
    ///
    type SearchResult;

    /// Get the schema for this type
    fn schema() -> Arc<Schema>;

    fn get_id(&self) -> Uuid;

    /// Get silo_id
    fn silo_id(&self) -> &str;

    /// Get the name of the resource for this type
    fn vtable_name() -> &'static str;

    /// COnvert a batch of items to a RecordBatch
    fn to_record_batch(
        items: &[Self],
        schema: Arc<Schema>,
        config: Arc<AnalyzerConfig>,
    ) -> Result<RecordBatch, NError>
    where
        Self: std::marker::Sized;

    /// Convert RecordBatches to search results
    fn from_record_batches(batches: Vec<RecordBatch>) -> Result<Vec<Self::SearchResult>, NError>;

    /// Get the embedding vector for this item
    fn embedding(&self, config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NError>;
}

#[derive(Clone)]
pub struct LatentStore {
    conn: Connection,
}

impl LatentStore {
    /// The function creates a new database connection, defines a table schema with fields, and creates
    /// an index on a specific field in the table.
    ///
    /// Arguments:
    ///
    /// * `dir_path`: The `dir_path` parameter in the `new` function is an optional reference to a
    ///   string that represents the directory path where the database will be created or accessed. It
    ///   allows the user to specify a custom directory path for the database. If no directory path is
    ///   provided, the database will be created/access
    ///
    /// Returns:
    ///
    /// The `new` function is returning an instance of the `LatentStore` struct, which contains a
    /// connection to a database.
    pub async fn new(dir_path: Option<&str>) -> Self {
        let path = db_path(dir_path);
        let path = path.to_str().unwrap();

        let conn = lancedb::connect(path).execute().await.unwrap();

        LatentStore { conn }
    }

    pub fn table_handler<T: Storable>(&self, config: Arc<AnalyzerConfig>) -> TableHandler<T> {
        TableHandler {
            conn: self.conn.clone(),
            config,
            _phantom: PhantomData,
        }
    }
}

pub struct TableHandler<T: Storable> {
    conn: Connection,
    config: Arc<AnalyzerConfig>,
    _phantom: PhantomData<T>,
}

impl<T: Storable> TableHandler<T> {
    pub async fn initialize(&self) -> Result<(), NError> {
        let scheme = T::schema();

        let tbl = self
            .conn
            .create_empty_table(T::vtable_name(), scheme)
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await
            .unwrap();

        let idx_builder = IvfHnswPqIndexBuilder::default();
        let idx_builder = idx_builder.distance_type(DistanceType::Cosine);

        tbl.create_index(&["vector"], Index::IvfHnswPq(idx_builder))
            .execute()
            .await
            .unwrap();

        Ok(())
    }

    pub async fn store(&self, items: &[T]) -> Result<(), NError> {
        let tbl = self.conn.open_table(T::vtable_name()).execute().await?;

        let schema = T::schema();

        let batch = T::to_record_batch(items, schema.clone(), self.config.clone())?;

        let batch_iter = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        tbl.add(batch_iter).execute().await?;

        Ok(())
    }

    pub async fn search(
        &self,
        item: &T,
        config: Arc<AnalyzerConfig>,
        columns: Vec<String>,
    ) -> Result<Vec<T::SearchResult>, NError> {
        let embedding = item.embedding(self.config.clone())?;

        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        let batches = table
            .query()
            .nearest_to(embedding)?
            .limit(config.top_k.unwrap_or(7))
            .select(Select::Columns(columns))
            .distance_type(DistanceType::Cosine)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = T::from_record_batches(batches)?;

        Ok(results)
    }

    pub async fn clear_table(&self) -> Result<(), NError> {
        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        table.delete("true").await?;

        Ok(())
    }
}

fn db_path(dir_path: Option<&str>) -> PathBuf {
    match dir_path {
        Some(v) => {
            let mut path = PathBuf::new();
            path.push(v);
            path
        }
        None => {
            let mut path = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
            path.push("nisaba");
            path.push(Uuid::now_v7().to_string());
            path
        }
    }
}
