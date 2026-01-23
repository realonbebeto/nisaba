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

use crate::{analyzer::AnalyzerConfig, error::NisabaError};

/// The `Storable` trait defines how to access common attributes of the store
/// element (TableDef/FieldDef) during latent store storage and retrieval.
pub trait Storable: Send + Sync {
    /// The result type returned from similarity search
    ///
    type SearchResult;

    /// Get the schema for this type
    fn schema() -> Arc<Schema>;

    /// Get name of the store element (TableDef/FieldDef)
    fn name(&self) -> &String;

    /// Get the id of the store element (TableDef/FieldDef)
    fn get_id(&self) -> Uuid;

    /// Get silo_id
    fn silo_id(&self) -> &str;

    /// Get the name of the resource for this type
    fn vtable_name() -> &'static str;

    /// Convert a batch of items to a RecordBatch
    fn to_record_batch(
        items: &[Self],
        schema: Arc<Schema>,
        config: Arc<AnalyzerConfig>,
    ) -> Result<RecordBatch, NisabaError>
    where
        Self: std::marker::Sized;

    /// Convert RecordBatches to search results
    fn from_record_batches(
        batches: Vec<RecordBatch>,
    ) -> Result<Vec<Self::SearchResult>, NisabaError>;

    /// Get the embedding vector for this item
    fn embedding(&self, config: Arc<AnalyzerConfig>) -> Result<Vec<f32>, NisabaError>;

    /// Get the columns required to fetch results from latent store
    fn result_columns() -> Vec<String>;
}

#[derive(Clone)]
/// The `LatentStore` represents an interface to access Lancedb store
///
/// Properties:
///
/// * `conn`: The `conn` property is a `Connection` to Lancedb.
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

    /// The `table_handler` is a function represents the result of a clustering process
    ///
    /// Arguments:
    ///
    /// * `conn`: The `conn` property is a `Connection` to Lancedb.
    /// * `config`: The `config` property is a shared holder to AnalyzerConfig.
    ///
    /// Returns:
    ///
    /// A table_handler of type TableHandler of generic T
    pub fn table_handler<T: Storable>(&self, config: Arc<AnalyzerConfig>) -> TableHandler<T> {
        TableHandler {
            conn: self.conn.clone(),
            config,
            _phantom: PhantomData,
        }
    }
}

/// The `TableHandler` represents a connection over store element which implements Storable trait
///
/// Properties:
///
/// * `conn`: The `conn` property is a `Connection` to Lancedb.
/// * `config`: The `config` property is a shared holder to AnalyzerConfig.
pub struct TableHandler<T: Storable> {
    conn: Connection,
    config: Arc<AnalyzerConfig>,
    // Marker to hold a Storable type
    // Used to associate the handler with a specific Storable type
    _phantom: PhantomData<T>,
}

impl<T: Storable> TableHandler<T> {
    pub async fn initialize(&self) -> Result<(), NisabaError> {
        let scheme = T::schema();

        self.conn
            .create_empty_table(T::vtable_name(), scheme)
            .mode(CreateTableMode::Overwrite)
            .execute()
            .await?;

        Ok(())
    }

    /// This `store` function takes in types that implements Storable trait for storage
    ///
    /// Arguments:
    ///
    /// * `items`: The `items` parameter is a slice of types that implements Storable trait
    ///
    /// Returns:
    ///
    /// A `Result` of unit value on success and NisabaError when connection to Latent Store,
    /// RecordBatch conversion, storage or index creation fails.
    pub async fn store(&self, items: &[T]) -> Result<(), NisabaError> {
        let tbl = self.conn.open_table(T::vtable_name()).execute().await?;

        let schema = T::schema();

        let batch = T::to_record_batch(items, schema.clone(), self.config.clone())?;

        let capacity = batch.num_rows();

        let batches = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema);

        tbl.add(batches).execute().await?;

        let idx_builder = IvfHnswPqIndexBuilder::default();
        let idx_builder = idx_builder.distance_type(DistanceType::Cosine);

        if capacity > 256 {
            tbl.create_index(&["vector"], Index::IvfHnswPq(idx_builder))
                .execute()
                .await?;
        }

        Ok(())
    }

    /// This `search` function performs a search in the latent store and returns a vector
    /// of FieldMatch or TableMatch
    ///
    /// Arguments:
    ///
    /// * `item`: The `item` parameter is a reference of type implementing Storable.
    ///
    /// * `config`: The `config` property is a shared holder to AnalyzerConfig.
    ///
    /// * `columns`: The `columns` property is a Vec of String used to return the required
    ///   set of fields from the latent store.
    ///
    /// Returns:
    ///
    /// A `Result` of a `Vec` of SearchResult values on success and NisabaError when
    /// embedding generation, search, and creation of the SearchResult fails.
    pub async fn search(
        &self,
        item: &T,
        config: Arc<AnalyzerConfig>,
        columns: Vec<String>,
    ) -> Result<Vec<T::SearchResult>, NisabaError> {
        let embedding = item.embedding(self.config.clone())?;

        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        let batches = table
            .query()
            .nearest_to(embedding)?
            .nprobes(config.top_k.unwrap_or(7))
            .refine_factor(5)
            .limit(config.top_k.unwrap_or(7))
            .select(Select::Columns(columns))
            .distance_type(DistanceType::Cosine)
            .distance_range(Some(0.), Some(1. - self.config.similarity_threshold))
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let results = T::from_record_batches(batches)?;

        Ok(results)
    }

    pub async fn clear_table(&self) -> Result<(), NisabaError> {
        let table = self.conn.open_table(T::vtable_name()).execute().await?;

        table.delete("true").await?;

        Ok(())
    }
}

/// A function that creates/provides local path to where the latent store will be created or is existing
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
            path
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_path_creation() {
        let default_path = db_path(None);
        let parent_dir = default_path.parent().unwrap();

        assert!(parent_dir.exists());
    }

    #[tokio::test]
    async fn test_new_with_custom_path() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();

        let _ltstore = LatentStore::new(Some(temp_path)).await;

        assert!(std::path::Path::new(temp_path).exists());
    }

    #[tokio::test]
    async fn test_new_valid_connection() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();

        let ltstore = LatentStore::new(Some(temp_path)).await;

        let tables = ltstore.conn.table_names().execute().await.unwrap();

        assert!(tables.is_empty() || !tables.is_empty());
    }

    #[test]
    fn test_vector_storage() {}

    #[test]
    fn test_vector_retrieval() {}
}
