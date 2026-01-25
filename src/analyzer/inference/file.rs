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
use std::{
    fs::{self, File},
    io::Seek,
    path::PathBuf,
    sync::Arc,
};
use uuid::Uuid;

use crate::{
    analyzer::{
        catalog::{StorageBackend, StorageConfig},
        inference::{
            SchemaInferenceEngine, SourceField, compute_field_metrics, convert_into_table_defs,
            promote::{ColumnStats, TypeLatticeResolver, cast_utf8_column},
        },
    },
    error::NisabaError,
    types::TableDef,
};

// =================================================
// Flat-File Inference Engine
// =================================================

/// File inference engine for common data file formats
#[derive(Debug)]
pub struct FileInferenceEngine {
    sample_size: usize,
    csv_has_header: bool,
}

impl Default for FileInferenceEngine {
    fn default() -> Self {
        Self {
            sample_size: 1000,
            csv_has_header: true,
        }
    }
}

impl FileInferenceEngine {
    pub fn new() -> Self {
        FileInferenceEngine::default()
    }

    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    pub fn with_csv_hasheader(mut self, has_header: bool) -> Self {
        self.csv_has_header = has_header;
        self
    }

    fn infer_from_csv(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        let dir_str = config.connection_string()?;
        let silo_id = format!("{}-{}", config.backend, Uuid::now_v7());

        let entries = fs::read_dir(dir_str)?;

        let mut table_defs: Vec<TableDef> = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("csv") {
                let mut file = File::open(&path)?;

                let (schema, _) = Format::default()
                    .with_header(self.csv_has_header)
                    .infer_schema(&mut file, Some(self.sample_size))?;

                file.rewind()?;

                let mut csv_reader = ReaderBuilder::new(Arc::new(schema.clone()))
                    .with_header(self.csv_has_header)
                    .build(file)?;

                let record_batch = csv_reader.next();

                if let Some(v) = &path.file_name().and_then(|s| s.to_str()) {
                    let table_name = (*v).to_string();

                    let result: Vec<SourceField> = schema
                        .fields()
                        .iter()
                        .map(|f| SourceField {
                            silo_id: silo_id.clone(),
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

                    // There will be generation of only one table def
                    let mut table_def = convert_into_table_defs(result)?;

                    if let Some(batch) = record_batch {
                        let mut batch = batch?;

                        // Promotion
                        let schema = batch.schema();

                        for (index, field) in schema.fields().iter().enumerate() {
                            let column = batch.column(index);

                            match column.data_type() {
                                DataType::LargeUtf8
                                | DataType::Utf8
                                | DataType::Int32
                                | DataType::Int64 => {
                                    let stats = ColumnStats::new(column);
                                    let resolver = TypeLatticeResolver::new();
                                    let resolved_result =
                                        resolver.promote(column.data_type(), &stats)?;

                                    if let Some(ff) = table_def[0]
                                        .fields
                                        .iter_mut()
                                        .find(|f| f.name == *field.name())
                                    {
                                        ff.type_confidence = Some(resolved_result.confidence);

                                        // Update char_max_length
                                        match (&ff.canonical_type, &resolved_result.dest_type) {
                                            (DataType::Utf8, DataType::Utf8)
                                            | (DataType::LargeUtf8, DataType::LargeUtf8)
                                            | (DataType::Utf8View, DataType::Utf8View) => {
                                                ff.char_max_length =
                                                    resolved_result.character_maximum_length;
                                            }

                                            (_, _) => {}
                                        }

                                        // Update type related signals when there is a mismatch on types
                                        if ff.canonical_type != resolved_result.dest_type {
                                            ff.canonical_type = resolved_result.dest_type;
                                            ff.type_confidence = Some(resolved_result.confidence);
                                            ff.is_nullable = resolved_result.nullable;
                                            ff.char_max_length =
                                                resolved_result.character_maximum_length;
                                            ff.numeric_precision =
                                                resolved_result.numeric_precision;
                                            ff.numeric_scale = resolved_result.numeric_scale;
                                            ff.datetime_precision =
                                                resolved_result.datetime_precision;

                                            // Very important for field values in batch to be updated
                                            cast_utf8_column(
                                                &mut batch,
                                                &ff.name,
                                                &ff.canonical_type,
                                            )?;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }

                        let metrics = compute_field_metrics(&batch)?;

                        let _ = table_def[0].fields.iter_mut().map(|f| {
                            let fmetrics = metrics.get(&f.name);

                            if let Some(m) = fmetrics {
                                f.char_class_signature = Some(m.char_class_signature);
                                f.is_monotonic = m.monotonicity;
                                f.cardinality = Some(m.cardinality);
                                f.avg_byte_length = m.avg_byte_length
                            }
                        });
                    }

                    table_defs.extend(table_def);
                }
            }
        }

        Ok(table_defs)
    }

    fn infer_from_excel(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        let dir_str = &config.dir_path.clone().ok_or(NisabaError::Missing(
            "Directory with Excel workbooks not provided".into(),
        ))?;
        // Get all excel filenames
        let entries = fs::read_dir(dir_str)?;
        let silo_id = format!("{}-{}", config.backend, Uuid::now_v7());

        let mut table_defs: Vec<TableDef> = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| {
                    ["xls", "xlsx", "xlsm", "xlsb"]
                        .iter()
                        .any(|sufx| s.ends_with(sufx))
                })
                .unwrap_or(false)
            {
                let record_batches = read_excel_to_record_batch(&path)?;

                for (mut batch, table_name) in record_batches {
                    let schema = batch.schema();

                    if let Some(v) = &path.file_name().and_then(|s| s.to_str()) {
                        let table_schema = (*v).to_string();

                        let result: Vec<SourceField> = schema
                            .fields()
                            .iter()
                            .map(|f| SourceField {
                                silo_id: silo_id.clone(),
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

                        // There will be generation of only one table def
                        let mut table_def = convert_into_table_defs(result)?;

                        for (index, field) in schema.fields().iter().enumerate() {
                            let column = batch.column(index);

                            match column.data_type() {
                                DataType::LargeUtf8
                                | DataType::Utf8
                                | DataType::Int32
                                | DataType::Int64 => {
                                    let stats = ColumnStats::new(column);

                                    let resolver = TypeLatticeResolver::new();
                                    let resolved_result =
                                        resolver.promote(column.data_type(), &stats)?;

                                    if let Some(ff) = table_def[0]
                                        .fields
                                        .iter_mut()
                                        .find(|f| f.name == *field.name())
                                    {
                                        ff.type_confidence = Some(resolved_result.confidence);
                                        // Update char_max_length
                                        match (&ff.canonical_type, &resolved_result.dest_type) {
                                            (DataType::Utf8, DataType::Utf8)
                                            | (DataType::LargeUtf8, DataType::LargeUtf8)
                                            | (DataType::Utf8View, DataType::Utf8View) => {
                                                ff.char_max_length =
                                                    resolved_result.character_maximum_length;
                                            }

                                            (_, _) => {}
                                        }
                                        if ff.canonical_type != resolved_result.dest_type {
                                            ff.canonical_type = resolved_result.dest_type.clone();
                                            ff.type_confidence = Some(resolved_result.confidence);
                                            ff.is_nullable = resolved_result.nullable;
                                            ff.char_max_length =
                                                resolved_result.character_maximum_length;
                                            ff.numeric_precision =
                                                resolved_result.numeric_precision;
                                            ff.numeric_scale = resolved_result.numeric_scale;
                                            ff.datetime_precision =
                                                resolved_result.datetime_precision;

                                            // Very important for field values in batch to be updated
                                            cast_utf8_column(
                                                &mut batch,
                                                &ff.name,
                                                &resolved_result.dest_type,
                                            )?;
                                        }
                                    }
                                }
                                _ => {}
                            }

                            let metrics = compute_field_metrics(&batch)?;

                            let _ = table_def[0].fields.iter_mut().map(|f| {
                                let fmetrics = metrics.get(&f.name);

                                if let Some(m) = fmetrics {
                                    f.char_class_signature = Some(m.char_class_signature);
                                    f.is_monotonic = m.monotonicity;
                                    f.cardinality = Some(m.cardinality);
                                    f.avg_byte_length = m.avg_byte_length
                                }
                            });
                        }

                        table_defs.extend(table_def);
                    }
                }
            }
        }

        Ok(table_defs)
    }

    fn infer_from_parquet(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        let dir_str = config.connection_string()?;

        // Get all parquet filenames
        let entries = fs::read_dir(dir_str)?;
        let silo_id = format!("{}-{}", config.backend, Uuid::now_v7());

        let mut table_defs: Vec<TableDef> = Vec::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("parquet") {
                let file = File::open(&path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
                let schema = builder.schema();

                if let Some(v) = &path.file_name().and_then(|s| s.to_str()) {
                    let table_name = (*v).to_string();
                    let result: Vec<SourceField> = schema
                        .fields()
                        .iter()
                        .map(|field| SourceField {
                            silo_id: silo_id.clone(),
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

                    // There will be generation of only one table def
                    let mut table_def = convert_into_table_defs(result)?;

                    let record_batch = builder.build()?.next();

                    if let Some(batch) = record_batch {
                        let mut batch = batch?;

                        // Promotion
                        let schema = batch.schema();

                        for (index, field) in schema.fields().iter().enumerate() {
                            let column = batch.column(index);

                            match column.data_type() {
                                DataType::LargeUtf8
                                | DataType::Utf8
                                | DataType::Int32
                                | DataType::Int64 => {
                                    let stats = ColumnStats::new(column);
                                    let resolver = TypeLatticeResolver::new();
                                    let resolved_result =
                                        resolver.promote(column.data_type(), &stats)?;

                                    if let Some(ff) = table_def[0]
                                        .fields
                                        .iter_mut()
                                        .find(|f| f.name == *field.name())
                                    {
                                        ff.type_confidence = Some(resolved_result.confidence);

                                        // Update char_max_length
                                        match (&ff.canonical_type, &resolved_result.dest_type) {
                                            (DataType::Utf8, DataType::Utf8)
                                            | (DataType::LargeUtf8, DataType::LargeUtf8)
                                            | (DataType::Utf8View, DataType::Utf8View) => {
                                                ff.char_max_length =
                                                    resolved_result.character_maximum_length;
                                            }

                                            (_, _) => {}
                                        }

                                        if ff.canonical_type != resolved_result.dest_type {
                                            ff.canonical_type = resolved_result.dest_type;
                                            ff.type_confidence = Some(resolved_result.confidence);
                                            ff.is_nullable = resolved_result.nullable;
                                            ff.char_max_length =
                                                resolved_result.character_maximum_length;
                                            ff.numeric_precision =
                                                resolved_result.numeric_precision;
                                            ff.numeric_scale = resolved_result.numeric_scale;
                                            ff.datetime_precision =
                                                resolved_result.datetime_precision;

                                            // Very important for field values in batch to be updated
                                            cast_utf8_column(
                                                &mut batch,
                                                &ff.name,
                                                &ff.canonical_type,
                                            )?;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }

                        let metrics = compute_field_metrics(&batch)?;

                        let _ = table_def[0].fields.iter_mut().map(|f| {
                            let fmetrics = metrics.get(&f.name);

                            if let Some(m) = fmetrics {
                                f.char_class_signature = Some(m.char_class_signature);
                                f.is_monotonic = m.monotonicity;
                                f.cardinality = Some(m.cardinality);
                                f.avg_byte_length = m.avg_byte_length
                            }
                        });
                    }

                    table_defs.extend(table_def);
                }
            }
        }

        Ok(table_defs)
    }
}

impl SchemaInferenceEngine for FileInferenceEngine {
    fn infer_schema(&self, config: &StorageConfig) -> Result<Vec<TableDef>, NisabaError> {
        match config.backend {
            StorageBackend::Csv => self.infer_from_csv(config),
            StorageBackend::Excel => self.infer_from_excel(config),
            StorageBackend::Parquet => self.infer_from_parquet(config),
            _ => Err(NisabaError::Unsupported(format!(
                "{:?} file store unsupported by File engine",
                config.backend
            )))?,
        }
    }

    fn can_handle(&self, backend: &StorageBackend) -> bool {
        matches!(backend, |StorageBackend::Csv| StorageBackend::Excel
            | StorageBackend::Parquet)
    }

    fn engine_name(&self) -> &str {
        "File"
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

fn read_excel_to_record_batch(path: &PathBuf) -> Result<Vec<(RecordBatch, String)>, NisabaError> {
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

                let result = read_range_to_arrow(range, headers, 0)?;

                dfs.extend(result.into_iter().map(|f| (f, tbl.clone())));
            }
        } else {
            let range = workbook.worksheet_range(sheet_name)?;

            let rows: Vec<_> = range.rows().collect();

            let headers: Vec<String> = rows[0].iter().map(|cell| cell.to_string()).collect();

            let result = read_range_to_arrow(&range, &headers, 1)?;

            dfs.extend(result.into_iter().map(|f| (f, sheet_name.clone())));
        }
    }

    Ok(dfs)
}

fn read_range_to_arrow(
    range: &Range<Data>,
    headers: &[String],
    skip: usize,
) -> Result<Option<RecordBatch>, NisabaError> {
    let (height, width) = range.get_size();

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
    use super::*;

    #[test]
    fn test_csv_inference() {
        let config = StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap();

        let csv_inference = FileInferenceEngine::default();

        let result = csv_inference.infer_from_csv(&config).unwrap();

        assert_eq!(result.len(), 9);

        // Fetched at is read in as Integer but parquet seems to store the semantic type
        let fetched_at = result
            .iter()
            .find(|t| t.name == "albums.csv")
            .unwrap()
            .fields
            .iter()
            .find(|f| f.name == "fetched_at")
            .unwrap();

        assert!(matches!(
            fetched_at.canonical_type,
            DataType::Timestamp(_, _)
        ));

        // release_date is read in as Integer but parquet seems to store the semantic type
        let release_date = result
            .iter()
            .find(|t| t.name == "albums.csv")
            .unwrap()
            .fields
            .iter()
            .find(|f| f.name == "release_date")
            .unwrap();

        assert!(matches!(release_date.canonical_type, DataType::Date32));
    }

    #[test]
    fn test_xlsx_inference() {
        let config =
            StorageConfig::new_file_backend(StorageBackend::Excel, "./assets/xlsx").unwrap();

        let excel_inference = FileInferenceEngine::default();

        let result = excel_inference.infer_from_excel(&config).unwrap();

        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_parquet_inference() {
        let config =
            StorageConfig::new_file_backend(StorageBackend::Excel, "./assets/parquet").unwrap();

        let parquet_inference = FileInferenceEngine::default();

        let result = parquet_inference.infer_from_parquet(&config).unwrap();

        assert_eq!(result.len(), 9);
    }
}
