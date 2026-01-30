use arrow::{
    array::{
        ArrayRef, BooleanArray, Date64Array, Float64Array, Int64Array, NullArray, RecordBatch,
        StringArray,
    },
    csv::{ReaderBuilder, reader::Format},
    datatypes::{DataType, Field, Schema},
};
use calamine::{Data, Ods, Range, Reader, Table, Xls, Xlsb, Xlsx, XlsxError, open_workbook};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    fs::{self, File},
    io::Seek,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::Mutex;

use crate::{
    analyzer::{
        datastore::Source,
        inference::{SchemaInferenceEngine, SourceField, convert_into_table_defs},
        probe::InferenceStats,
    },
    error::NisabaError,
    types::{TableDef, TableRep},
};

// =================================================
// CSV Inference Engine
// =================================================
#[derive(Debug)]
pub struct CsvInferenceEngine;

impl Default for CsvInferenceEngine {
    fn default() -> Self {
        CsvInferenceEngine
    }
}

impl CsvInferenceEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn csv_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        workers: usize,
        on_table: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let dir_str = source
            .client
            .as_path()
            .ok_or(NisabaError::Missing("No csv dir path provided".into()))?;
        // Collect CSV paths first
        let csv_paths: Vec<_> = fs::read_dir(dir_str)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
            .collect();

        let table_reps = Mutex::new(Vec::new());

        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?
            .install(|| {
                csv_paths.par_iter().for_each(|path| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    {
                        let mut stats = rt.block_on(infer_stats.lock());
                        stats.tables_found += 1;
                    }

                    match self.infer_single_csv(path, source) {
                        Ok(table_def) => {
                            {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.tables_inferred += 1;
                                stats.fields_inferred += table_def.fields.len();
                            }

                            {
                                let mut table_reps = rt.block_on(table_reps.lock());
                                table_reps.push((&table_def).into());
                            }

                            if let Err(e) = rt.block_on(on_table(vec![table_def])) {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.errors.push(e.to_string());
                            }
                        }
                        Err(e) => {
                            let mut stats = rt.block_on(infer_stats.lock());
                            stats.errors.push(e.to_string());
                        }
                    }
                })
            });

        Ok(table_reps.into_inner())
    }

    fn infer_single_csv(&self, path: &PathBuf, source: &Source) -> Result<TableDef, NisabaError> {
        let mut file = File::open(path)?;

        // 1. Infer schema from file
        let (schema, _) = Format::default()
            .with_header(source.metadata.has_header)
            .infer_schema(&mut file, Some(source.metadata.num_rows))?;

        file.rewind()?;

        // 2. Read batch for stats and promotion(inference)
        let mut csv_reader = ReaderBuilder::new(Arc::new(schema.clone()))
            .with_header(source.metadata.has_header)
            .build(file)?;

        let record_batch = csv_reader.next();

        let table_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| NisabaError::Invalid(path.to_string_lossy().into()))?
            .to_string();

        // 3. Build source fields
        let source_fields: Vec<SourceField> = schema
            .fields()
            .iter()
            .map(|f| SourceField {
                silo_id: source.metadata.silo_id.clone(),
                table_schema: "csv".into(),
                table_name: table_name.clone(),
                column_name: f.name().clone(),
                udt_name: format!("{}", f.data_type()),
                data_type: f.data_type().to_string(),
                column_default: None,
                character_maximum_length: None,
                is_nullable: if f.is_nullable() {
                    "YES".into()
                } else {
                    "NO".into()
                },
                numeric_precision: None,
                numeric_scale: None,
                datetime_precision: None,
            })
            .collect();

        // 4. Convert to table_def
        let table_def = convert_into_table_defs(source_fields)?;
        let mut table_def = table_def
            .into_iter()
            .next()
            .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

        if let Some(Ok(mut batch)) = record_batch {
            self.enrich_table_def(&mut table_def, &mut batch)?;
        }

        Ok(table_def)
    }
}

impl SchemaInferenceEngine for CsvInferenceEngine {
    fn engine_name(&self) -> &str {
        "csv"
    }
}

// =================================================
// Excel Inference Engine
// =================================================
#[derive(Debug)]
pub struct ExcelInferenceEngine;

impl Default for ExcelInferenceEngine {
    fn default() -> Self {
        ExcelInferenceEngine
    }
}

impl ExcelInferenceEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn excel_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        workers: usize,
        on_tables: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let dir_str = source
            .client
            .as_path()
            .ok_or(NisabaError::Missing("No excel directory provided".into()))?;
        // Collect Excel paths first
        let excel_paths: Vec<_> = fs::read_dir(dir_str)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let ext = p.extension().and_then(|s| s.to_str());
                ext == Some("csv")
                    || ext == Some("xls")
                    || ext == Some("xlsx")
                    || ext == Some("xlsm")
                    || ext == Some("xlsb")
            })
            .collect();

        let table_reps = Mutex::new(Vec::new());

        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?
            .install(|| {
                excel_paths.par_iter().for_each(|path| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();

                    {
                        let mut stats = rt.block_on(infer_stats.lock());
                        stats.tables_found += 1;
                    }

                    match self.infer_single_excel(path, source) {
                        Ok(table_defs) => {
                            let tables_count = table_defs.len();
                            let fields_count: usize =
                                table_defs.iter().map(|f| f.fields.len()).sum();

                            {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.tables_inferred += tables_count;
                                stats.fields_inferred += fields_count;
                            }

                            {
                                let mut table_reps = rt.block_on(table_reps.lock());
                                table_reps.extend(table_defs.iter().map(|td| {
                                    let tr: TableRep = td.into();
                                    tr
                                }));
                            }

                            if let Err(e) = rt.block_on(on_tables(table_defs)) {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.errors.push(e.to_string());
                            }
                        }

                        Err(e) => {
                            let mut stats = rt.block_on(infer_stats.lock());
                            stats.errors.push(e.to_string());
                        }
                    }
                })
            });

        Ok(table_reps.into_inner())
    }

    fn infer_single_excel(
        &self,
        path: &PathBuf,
        source: &Source,
    ) -> Result<Vec<TableDef>, NisabaError> {
        let record_batches = read_excel_to_record_batch(
            path,
            Some(source.metadata.num_rows),
            Some(source.metadata.has_header),
        )?;

        let table_schema = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|t| t.to_string())
            .ok_or(NisabaError::Invalid("Invalid excel path file name".into()))?;
        let mut table_defs = Vec::new();

        for (mut batch, table_name) in record_batches {
            let schema = batch.schema();

            let source_fields: Vec<SourceField> = schema
                .fields()
                .iter()
                .map(|f| SourceField {
                    silo_id: source.metadata.silo_id.clone(),
                    table_schema: table_schema.clone(),
                    table_name: table_name.clone(),
                    column_name: f.name().clone(),
                    udt_name: f.data_type().to_string(),
                    data_type: f.data_type().to_string(),
                    column_default: None,
                    character_maximum_length: None,
                    is_nullable: if f.is_nullable() {
                        "YES".into()
                    } else {
                        "NO".into()
                    },
                    numeric_precision: None,
                    numeric_scale: None,
                    datetime_precision: None,
                })
                .collect();

            // 4. Convert to table_def
            let table_def = convert_into_table_defs(source_fields)?;

            let mut table_def = table_def
                .into_iter()
                .next()
                .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

            self.enrich_table_def(&mut table_def, &mut batch)?;

            table_defs.push(table_def);
        }

        Ok(table_defs)
    }
}

impl SchemaInferenceEngine for ExcelInferenceEngine {
    fn engine_name(&self) -> &str {
        "excel"
    }
}

// =================================================
// Parquet Inference Engine
// =================================================
#[derive(Debug)]
pub struct ParquetInferenceEngine;

impl Default for ParquetInferenceEngine {
    fn default() -> Self {
        ParquetInferenceEngine
    }
}

impl ParquetInferenceEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn parquet_store_infer<F, Fut>(
        &self,
        source: &Source,
        infer_stats: Arc<Mutex<InferenceStats>>,
        workers: usize,
        on_tables: F,
    ) -> Result<Vec<TableRep>, NisabaError>
    where
        F: Fn(Vec<TableDef>) -> Fut + Sync,
        Fut: Future<Output = Result<(), NisabaError>> + Send,
    {
        let dir_str = source
            .client
            .as_path()
            .ok_or(NisabaError::Missing("No parquet directory provided".into()))?;

        // Collect Parquet paths first
        let parquet_paths: Vec<_> = fs::read_dir(dir_str)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("parquet"))
            .collect();

        let table_reps = Mutex::new(Vec::new());

        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?
            .install(|| {
                parquet_paths.par_iter().for_each(|path| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    {
                        let mut stats = rt.block_on(infer_stats.lock());
                        stats.tables_found += 1;
                    }

                    match self.infer_single_parquet(path, source) {
                        Ok(table_def) => {
                            {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.tables_inferred += 1;
                                stats.fields_inferred += 1;
                                let mut table_reps = rt.block_on(table_reps.lock());
                                table_reps.push((&table_def).into());
                            }

                            if let Err(e) = rt.block_on(on_tables(vec![table_def])) {
                                let mut stats = rt.block_on(infer_stats.lock());
                                stats.errors.push(e.to_string());
                            }
                        }
                        Err(e) => {
                            let mut stats = rt.block_on(infer_stats.lock());
                            stats.errors.push(e.to_string());
                        }
                    }
                })
            });

        Ok(table_reps.into_inner())
    }

    fn infer_single_parquet(
        &self,
        path: &PathBuf,
        source: &Source,
    ) -> Result<TableDef, NisabaError> {
        let file = File::open(path)?;
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file)?.with_limit(source.metadata.num_rows);
        let schema = builder.schema();

        let table_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|t| t.to_string())
            .ok_or(NisabaError::Invalid("Invalid excel path file name".into()))?;

        let result: Vec<SourceField> = schema
            .fields()
            .iter()
            .map(|field| SourceField {
                silo_id: source.metadata.silo_id.clone(),
                table_schema: "default".into(),
                table_name: table_name.clone(),
                column_name: field.name().into(),
                udt_name: format!("{}", field.data_type()),
                data_type: field.extension_type_name().unwrap_or_default().to_string(),
                column_default: None,
                character_maximum_length: None,
                is_nullable: format!("{}", field.is_nullable()),
                numeric_precision: None,
                numeric_scale: None,
                datetime_precision: None,
            })
            .collect();

        let table_def = convert_into_table_defs(result)?;

        let mut table_def = table_def
            .into_iter()
            .next()
            .ok_or_else(|| NisabaError::NoTableDefGenerated)?;

        let mut batch = builder.build()?.next().ok_or(NisabaError::Missing(
            "Parquet RecordBatch not generated on read".into(),
        ))??;

        self.enrich_table_def(&mut table_def, &mut batch)?;

        Ok(table_def)
    }
}

impl SchemaInferenceEngine for ParquetInferenceEngine {
    fn engine_name(&self) -> &str {
        "parquet"
    }
}

// ===================
// Read Excel Files
// ===================

enum Workbook {
    Ods(Ods<std::io::BufReader<std::fs::File>>),
    Xls(Xls<std::io::BufReader<std::fs::File>>),
    Xlsx(Xlsx<std::io::BufReader<std::fs::File>>),
    Xlsb(Xlsb<std::io::BufReader<std::fs::File>>),
}

impl Workbook {
    fn open(path: &PathBuf) -> Result<Self, NisabaError> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("ods") => Ok(Workbook::Ods(open_workbook(path)?)),
            Some("xls") => Ok(Workbook::Xls(open_workbook(path)?)),
            Some("xlsb") => Ok(Workbook::Xlsb(open_workbook(path)?)),
            Some("xlsm") | Some("xlsx") => Ok(Workbook::Xlsx(open_workbook(path)?)),
            _ => Err(NisabaError::Unsupported(path.to_string_lossy().into())),
        }
    }
    fn load_tables(&mut self) -> Result<(), NisabaError> {
        match self {
            Workbook::Ods(_) => Ok(()),
            Workbook::Xls(_) => Ok(()),
            Workbook::Xlsb(_) => Ok(()),
            Workbook::Xlsx(wb) => Ok(wb.load_tables()?),
        }
    }
    fn sheet_names(&mut self) -> Vec<String> {
        match self {
            Workbook::Ods(wb) => wb.sheet_names(),
            Workbook::Xls(wb) => wb.sheet_names(),
            Workbook::Xlsb(wb) => wb.sheet_names(),
            Workbook::Xlsx(wb) => wb.sheet_names(),
        }
    }

    fn table_names_in_sheet(&self, sheet_name: &str) -> Vec<&String> {
        match self {
            Workbook::Ods(_) => vec![],
            Workbook::Xls(_) => vec![],
            Workbook::Xlsb(_) => vec![],
            Workbook::Xlsx(wb) => wb.table_names_in_sheet(sheet_name),
        }
    }

    fn table_by_name(&mut self, table_name: &str) -> Result<Table<Data>, NisabaError> {
        match self {
            Workbook::Ods(_) => Err(NisabaError::Xlsx(XlsxError::TableNotFound(
                table_name.into(),
            ))),
            Workbook::Xls(_) => Err(NisabaError::Xlsx(XlsxError::TableNotFound(
                table_name.into(),
            ))),
            Workbook::Xlsb(_) => Err(NisabaError::Xlsx(XlsxError::TableNotFound(
                table_name.into(),
            ))),
            Workbook::Xlsx(wb) => Ok(wb.table_by_name(table_name)?),
        }
    }

    fn worksheet_range(&mut self, name: &str) -> Result<Range<Data>, NisabaError> {
        match self {
            Workbook::Ods(wb) => Ok(wb.worksheet_range(name)?),
            Workbook::Xls(wb) => Ok(wb.worksheet_range(name)?),
            Workbook::Xlsb(wb) => Ok(wb.worksheet_range(name)?),
            Workbook::Xlsx(wb) => Ok(wb.worksheet_range(name)?),
        }
    }
}

fn read_excel_to_record_batch(
    path: &PathBuf,
    limit: Option<usize>,
    has_header: Option<bool>,
) -> Result<Vec<(RecordBatch, String)>, NisabaError> {
    // Open workbook
    let mut workbook = Workbook::open(path)?;
    let sheet_names = workbook.sheet_names();

    workbook.load_tables()?;

    let mut dfs = Vec::new();

    // Preference to handle tables in sheets
    for sheet_name in &sheet_names {
        // Get the table names in the current sheet.
        let table_names: Vec<String> = workbook
            .table_names_in_sheet(sheet_name)
            .iter()
            .copied()
            .map(|f| f.to_string())
            .collect();

        if !table_names.is_empty() {
            for tbl in table_names {
                let table = workbook.table_by_name(&tbl)?;
                let headers = table.columns();

                let range = table.data();

                let result = read_range_to_arrow(range, headers, 0, limit)?;

                dfs.extend(result.into_iter().map(|f| (f, tbl.clone())));
            }
        } else {
            let range = workbook.worksheet_range(sheet_name)?;

            let rows: Vec<_> = range.rows().collect();

            let headers: Vec<String> = rows[0].iter().map(|cell| cell.to_string()).collect();

            let skip_rows = has_header.unwrap_or(false) as usize;

            let batch = read_range_to_arrow(&range, &headers, skip_rows, limit)?;

            dfs.extend(batch.into_iter().map(|f| (f, sheet_name.clone())));
        }
    }

    Ok(dfs)
}

fn read_range_to_arrow(
    range: &Range<Data>,
    headers: &[String],
    skip: usize,
    limit: Option<usize>,
) -> Result<Option<RecordBatch>, NisabaError> {
    let (height, width) = range.get_size();

    let height = limit.map(|v| v.min(height)).unwrap_or(height);

    if height > skip {
        let mut columns: Vec<ArrayRef> = Vec::new();
        let mut fields: Vec<Field> = Vec::new();

        for (index, col) in headers.iter().enumerate().take(width) {
            let values: Vec<Option<&Data>> = (skip..height)
                .map(|row| range.get_value((row as u32, index as u32)))
                .collect();

            let check_vals: Vec<&Data> = values.iter().cloned().flatten().collect();

            let dtype = match calamine_type_to_arrow(&check_vals) {
                DataType::Null => {
                    let arr = NullArray::new(height);
                    columns.push(Arc::new(arr));
                    Some(DataType::Null)
                }
                DataType::Boolean => {
                    let vals: Vec<Option<bool>> = values
                        .iter()
                        .map(|val| {
                            if let Some(v) = val {
                                match v {
                                    Data::Bool(v) => return Some(*v),
                                    _ => return None,
                                }
                            }
                            None
                        })
                        .collect();

                    let arr: BooleanArray = vals.into();
                    columns.push(Arc::new(arr));

                    Some(DataType::Boolean)
                }
                DataType::Date64 => {
                    let vals: Vec<Option<i64>> = values
                        .iter()
                        .map(|val| {
                            if let Some(v) = val {
                                match v {
                                    Data::DateTime(v) => {
                                        let edt = v.as_datetime();
                                        match edt {
                                            Some(ed) => {
                                                return Some(ed.and_utc().timestamp_millis());
                                            }
                                            None => return None,
                                        }
                                    }
                                    _ => return None,
                                }
                            }
                            None
                        })
                        .collect();

                    columns.push(Arc::new(Date64Array::from(vals)));

                    Some(DataType::Date64)
                }
                DataType::Int64 => {
                    let vals: Vec<Option<i64>> = values
                        .iter()
                        .map(|val| {
                            if let Some(v) = val {
                                match v {
                                    Data::Int(v) => return Some(*v),
                                    _ => return None,
                                }
                            }
                            None
                        })
                        .collect();

                    columns.push(Arc::new(Int64Array::from(vals)));
                    Some(DataType::Int64)
                }
                DataType::Float64 => {
                    let vals: Vec<Option<f64>> = values
                        .iter()
                        .map(|val| {
                            if let Some(v) = val {
                                match v {
                                    Data::Float(v) => return Some(*v),
                                    _ => return None,
                                }
                            }
                            None
                        })
                        .collect();

                    columns.push(Arc::new(Float64Array::from(vals)));

                    Some(DataType::Float64)
                }
                DataType::Utf8 => {
                    let vals: Vec<Option<String>> = values
                        .iter()
                        .map(|val| {
                            if let Some(v) = val {
                                match v {
                                    Data::String(v) => return Some(v.clone()),
                                    Data::Error(v) => return Some((*v).to_string()),
                                    Data::DurationIso(v) => return Some(v.clone()),
                                    _ => return None,
                                }
                            }
                            None
                        })
                        .collect();

                    columns.push(Arc::new(StringArray::from(vals)));

                    Some(DataType::Utf8)
                }
                _ => None,
            };

            if let Some(dtype) = dtype {
                fields.push(Field::new(col.clone(), dtype, true))
            }
        }

        let schema = Arc::new(Schema::new(fields));

        return Ok(Some(RecordBatch::try_new(schema, columns)?));
    }

    Ok(None)
}

fn calamine_type_to_arrow(values: &[&Data]) -> DataType {
    if values.iter().all(|val| matches!(**val, Data::Bool(_))) {
        DataType::Boolean
    } else if values.iter().all(|val| **val == Data::Empty) {
        DataType::Null
    } else if values.iter().all(|val| matches!(**val, Data::DateTime(_))) {
        DataType::Date64
    } else if values.iter().all(|val| matches!(**val, Data::Int(_))) {
        DataType::Int64
    } else if values.iter().all(|val| matches!(**val, Data::Float(_))) {
        if values.iter().all(|val| match val {
            Data::Float(v) => {
                v.is_finite() && v.fract().abs() < f64::MIN_POSITIVE && v.abs() < i64::MAX as f64
            }
            _ => false,
        }) {
            return DataType::Int64;
        }
        DataType::Float64
    } else {
        DataType::Utf8
    }
}

#[cfg(test)]
mod tests {

    use crate::{AnalyzerConfig, LatentStore, analyzer::datastore::FileStoreType};

    use super::*;

    #[tokio::test]
    async fn test_csv_inference() {
        let source = Source::files(FileStoreType::Csv)
            .path("./assets/csv")
            .num_rows(1000)
            .has_header(true)
            .build()
            .unwrap();

        let csv_inference = CsvInferenceEngine::new();

        let stats = Arc::new(Mutex::new(InferenceStats::default()));

        let latent_store = Arc::new(
            LatentStore::builder()
                .analyzer_config(Arc::new(AnalyzerConfig::default()))
                .build()
                .await
                .unwrap(),
        );

        let table_handler = latent_store.table_handler::<TableRep>();

        let result = csv_inference
            .csv_store_infer(&source, stats, 4, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .unwrap();

        assert_eq!(result.len(), 9);
    }

    #[tokio::test]
    async fn test_xlsx_inference() {
        let source = Source::files(FileStoreType::Excel)
            .path("./assets/xlsx")
            .num_rows(1000)
            .has_header(true)
            .build()
            .unwrap();

        let excel_inference = ExcelInferenceEngine::new();

        let stats = Arc::new(Mutex::new(InferenceStats::default()));

        let latent_store = Arc::new(
            LatentStore::builder()
                .analyzer_config(Arc::new(AnalyzerConfig::default()))
                .build()
                .await
                .unwrap(),
        );
        let table_handler = latent_store.table_handler::<TableRep>();

        let result = excel_inference
            .excel_store_infer(&source, stats, 4, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .unwrap();

        assert_eq!(result.len(), 9);
    }

    #[tokio::test]
    async fn test_parquet_inference() {
        let source = Source::files(FileStoreType::Parquet)
            .path("./assets/parquet")
            .num_rows(1000)
            .build()
            .unwrap();

        let parquet_inference = ParquetInferenceEngine::new();

        let stats = Arc::new(Mutex::new(InferenceStats::default()));

        let latent_store = Arc::new(
            LatentStore::builder()
                .analyzer_config(Arc::new(AnalyzerConfig::default()))
                .build()
                .await
                .unwrap(),
        );
        let table_handler = latent_store.table_handler::<TableRep>();

        let result = parquet_inference
            .parquet_store_infer(&source, stats, 4, |table_defs| async {
                table_handler.store_tables(table_defs).await?;
                Ok(())
            })
            .unwrap();

        assert_eq!(result.len(), 9);
    }
}
