use arrow::{
    array::{
        Array, AsArray, BinaryViewArray, BooleanArray, Date32Array, FixedSizeBinaryArray,
        Float64Array, GenericBinaryArray, GenericStringArray, Int8Array, Int16Array, Int32Array,
        Int64Array, LargeBinaryArray, LargeStringArray, OffsetSizeTrait, StringArray,
        StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    },
    compute::cast,
    datatypes::{DataType, Float64Type, Int64Type, TimeUnit},
    error::ArrowError,
};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

use core::f64;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};
use uuid::Uuid;

use crate::{
    error::NisabaError,
    reconciler::calculation::{autocorrelation, histogram_entropy, kurtosis, skewness},
};

const TIME_UNITS: [TimeUnit; 4] = [
    TimeUnit::Second,
    TimeUnit::Millisecond,
    TimeUnit::Microsecond,
    TimeUnit::Nanosecond,
];

const TIME_FMTS: &[&str] = &[
    "%p%I%M%S%.f",
    "%H:%M:%S%.fZ",
    "%H:%M:%S%.6f",
    "%H:%M:%S%.f%:z",
    "%H:%M:%S%.f",
    "%H:%M:%S",
    "%H:%M:%S%.f GMT",
    "%H:%M:%S%.f %z",
    "%H%M%S%.f",
    "%-I:%M %p",
];

const DATE_FMTS: &[&str] = &[
    "%d%b%Y",
    "%y/%m/%d",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%a, %d %b %Y",
    "%d/%b/%Y",
    "%Y%m%d",
];

const TIMESTAMP_FMTS: &[&str] = &[
    "%d%b%Y%p%I%M%S",
    "%y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    // RFC / logs
    "%a, %d %b %Y %H:%M:%S",
    "%d/%b/%Y:%H:%M:%S",
    "%Y%m%d%H%M%S",
    // --
    "%d%b%Y%p%I%M%S%.f",
    // ISO-8601 (T separator)
    "%Y-%m-%dT%H:%M:%S%.fZ",
    "%Y-%m-%dT%H:%M:%S%.f%:z",
    "%Y-%m-%dT%H:%M:%S%.f",
    // Space-separated
    "%Y-%m-%d %H:%M:%S%.f%:z",
    // RFC / logs
    "%a, %d %b %Y %H:%M:%S%.f GMT",
    "%d/%b/%Y:%H:%M:%S%.f %z",
    "%Y-%m-%d %H:%M:%S%.f",
    "%y/%m/%d %H:%M:%S%.f",
    "%Y%m%d%H%M%S%.f",
];

const BOOLEAN_FMTS: &[&str] = &[
    "t", "tr", "tru", "true", "y", "ye", "yes", "on", "1", "1.0", "f", "fa", "fal", "fals",
    "false", "n", "no", "of", "off", "0", "0.0",
];

#[allow(unused)]
#[derive(Debug, Clone)]
/// The `FieldStats` represents statistical and profile information about a column/field in a dataset.
pub struct FieldStats {
    pub table_name: String,
    pub field_name: String,
    pub source: DataType,
    /// number of values in the column that were used to calculate the statistics.
    pub sample_size: usize,
    /// number of null values present in the column for which the statistics are being calculated.
    pub null_count: usize,
    /// number of unique or distinct values present in the column for which the statistics are being calculated.
    pub distinct_count: usize,
    /// average of values in the column for numeric values and lengths for Utf8
    pub avg: f64,
    /// median of values in the column for numeric values abd lengths for Utf8
    pub median: f64,
    pub p10: Option<f64>,
    pub p90: Option<f64>,
    /// minimum value found in the column for which these statistics are calculated.
    pub min_val: Option<f64>,
    /// maximum value found in the column for which these statistics are calculated.
    pub max_val: Option<f64>,
    /// measure for "tailedness" or "peakedness"
    pub kurtosis: Option<f64>,
    /// measure for asymmetry of the distribution
    pub skewness: Option<f64>,
    /// different percentiles for the data
    pub quantiles_f64: Option<Vec<f64>>, // p01, ..., p99
    /// measure of the amount of uncertainty or randomness in the data
    pub entropy: f32,
    /// measure of the amount of uncertainty related to temporal(time related) data
    pub histogram_entropy: Option<f64>,
    /// maximum length of characters in the column for any value within the column
    pub character_max_length: Option<i32>,
    /// minimum length of characters in the column for any value within the column
    pub character_min_length: Option<i32>,
    /// number of digits that can be stored, both to the left and right of the decimal point.
    pub numeric_precision: Option<i32>,
    /// number of digits to the right of the decimal point in a number
    pub numeric_scale: Option<i32>,
    /// precision of datetime values in the column
    pub datetime_precision: Option<i32>,
    /// Flag if the values are boolean integers
    pub all_int_bools: Option<bool>,
    /// Flag if the values are boolean utf8
    pub all_utf8_bools: Option<bool>,
    /// Flag if the values are semantically correct integers where values are less than 0 and more than 1
    pub all_integers: Option<bool>,
    /// Flag if string values are UUIDs
    pub all_uuid: Option<bool>,
    /// Flag if string values depict array pattern
    pub all_array: Option<bool>,
    /// Flag if string values depict json pattern
    pub all_json: Option<bool>,
    /// Potent datetime
    pub all_time: Option<(DataType, bool)>,
    pub all_date32: Option<(DataType, bool)>,
    pub all_timestamp: Option<(DataType, bool)>,
    /// Relationship between a data series and a lagged version of itself, revealing patterns or trends over time
    pub autocorrelation: Option<f64>,
    /// Optional floating-point measure of the uniquness/distinctness
    pub cardinality: Option<f64>,
    /// Boolean flag if the field is sequential for integer/time/timestamp/date field
    pub is_monotonic: bool,
    /// Flag if field allows null values
    pub is_null: bool,
    // Optional character class signature array of a String field
    pub char_class_signature: Option<ColSignature>, // [digit, alpha, whitespace, punctuation, special]
    /// GCD of consecutive differences (intrinsic resolution)
    pub delta_gcd: Option<i64>,
    /// Time bands as per digit sizes of integer timestamps 5-10
    pub timestamp_bands: Option<HashMap<TimeUnit, [usize; 6]>>,
    /// Flag if timestamp values can be cast down to date32
    pub all_timestamp_date32: Option<HashMap<TimeUnit, bool>>,
}

impl FieldStats {
    /// The function `new` creates an instance of `FieldStats` from an Arrow array
    /// by means of taking a refence of a type that implements Array trait.
    ///
    /// Dispatches to a type-specific extraction method for every Arrow `DataType`
    /// family. Types that are not yet analysed (nested, union, etc.) are left with
    /// their default `None` / zero values and a `char_class_signature` of
    /// `ColSignature::Opaque` so callers can distinguish "no data" from "not
    /// applicable".
    pub fn calculate(
        array: &dyn Array,
        table_name: &str,
        field_name: &str,
    ) -> Result<FieldStats, NisabaError> {
        let null_count = array.null_count();
        // `sample_size` counts only the non-null rows that will feed statistics.
        let sample_size = array.len() - null_count;

        let mut stats = Self {
            table_name: table_name.to_string(),
            field_name: field_name.to_string(),
            source: array.data_type().clone(),
            sample_size,
            null_count,
            // Everything else starts at its "not computed" default
            distinct_count: 0,
            avg: 0.0,
            median: 0.0,
            p10: None,
            p90: None,
            min_val: None,
            max_val: None,
            kurtosis: None,
            skewness: None,
            quantiles_f64: None,
            entropy: 0.0,
            histogram_entropy: None,
            character_max_length: None,
            character_min_length: None,
            numeric_precision: None,
            numeric_scale: None,
            datetime_precision: None,
            all_int_bools: None,
            all_utf8_bools: None,
            all_integers: None,
            all_uuid: None,
            all_array: None,
            all_json: None,
            all_time: None,
            all_date32: None,
            all_timestamp: None,
            autocorrelation: None,
            cardinality: None,
            is_monotonic: false,
            is_null: null_count > 0,
            char_class_signature: None,
            delta_gcd: None,
            timestamp_bands: None,
            all_timestamp_date32: None,
        };

        if sample_size == 0 || sample_size == null_count {
            return Ok(stats);
        }

        // Cardinality is meaningful for every non-empty type
        stats.cardinality = Some(compute_cardinality(array)?);
        stats.avg = compute_avg_byte_length(array)?.unwrap_or(-1.0);

        // Dispatch to the appropriate extraction method.
        // Each method updates relevant metrics and errors are propagated
        match array.data_type() {
            // String metric dispatch
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                Self::extract_stats_from_string_array(&mut stats, array)?;
            }
            // Numeric metric dispatch
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float64
            | DataType::Float32
            | DataType::Float16 => {
                Self::extract_stats_from_numeric_array(&mut stats, array)?;
            }

            // Capture precision and scale
            DataType::Decimal128(p, s) | DataType::Decimal256(p, s) => {
                stats.numeric_precision = Some(*p as i32);
                stats.numeric_scale = Some(*s as i32);

                Self::extract_stats_from_numeric_array(&mut stats, array)?;
            }

            // Boolean
            DataType::Boolean => {
                Self::extract_stats_from_boolean_array(&mut stats, array)?;
            }

            DataType::Date32 => {
                Self::extract_stats_from_date32_array(&mut stats, array)?;
            }

            DataType::Time32(_) => {
                Self::extract_stats_from_time_array(&mut stats, array)?;
            }

            DataType::Timestamp(tu, _) => {
                Self::extract_stats_from_timestamp_array(&mut stats, array, *tu)?;
            }

            // Binary
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => {
                Self::extract_stats_from_binary_array(&mut stats, array)?;
            }
            _ => stats.char_class_signature = Some(ColSignature::Opaque),
        }

        Ok(stats)
    }

    pub fn re_calculate(&mut self, array: &dyn Array) -> Result<FieldStats, NisabaError> {
        Self::calculate(array, &self.table_name, &self.field_name)
    }

    /// The method `null_ratio` calculates the null proportion in sampled values.
    ///
    /// Arguments:
    /// * `self`: reference of ColumnStats
    ///
    /// Returns:
    ///
    /// The functions returns a floating-point number representing the null proportion of
    /// the sampled data
    pub fn null_ratio(&self) -> f32 {
        if self.sample_size == 0 {
            0.0
        } else {
            self.null_count as f32 / self.sample_size as f32
        }
    }

    /// The function `extract_stats_from_numeric_array` extracts statistics such as distinct count, maximum
    /// value, minimum value, and entropy from an integer array.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a mutable reference to a `ColumnStats` struct or object.
    ///
    /// Returns:
    ///
    /// The function `extract_stats_from_numeric_array` is returning a `Result<(), NisabaError>`.
    fn extract_stats_from_numeric_array(
        stats: &mut FieldStats,
        vals: &dyn Array,
    ) -> Result<(), NisabaError> {
        let arr = cast(vals, &DataType::Float64)?;
        let values = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to Float64Array".into(),
            ))?;

        let mut values: Vec<f64> = values.iter().flatten().collect();

        if values.is_empty() {
            return Ok(());
        }

        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        stats.delta_gcd = Self::compute_delta_gcd(&values);

        let all_bool = values
            .iter()
            .all(|v| v.fract() < 1e-9 && matches!(v.round() as i64, 0 | 1));

        let all_integers = values.iter().all(|v| v.fract().abs() < f64::EPSILON)
            && !values.iter().all(|v| (0..=1).contains(&(v.round() as i64)));

        // Distinct count
        let mut distinct_set = HashSet::with_capacity(values.len());
        let mut max_val = f64::NEG_INFINITY;
        let mut min_val = f64::INFINITY;
        for val in &values {
            distinct_set.insert(val.to_bits());
            max_val = max_val.max(*val);
            min_val = min_val.min(*val);
        }

        // Quantiles
        if stats.sample_size > 20 {
            stats.quantiles_f64 = Some(Self::compute_quantiles(&values, 11));
        }

        stats.p10 = Some(Self::percentile(&values, 0.1));
        stats.p90 = Some(Self::percentile(&values, 0.9));

        // Timestamp
        let mut time_bands: HashMap<TimeUnit, [usize; 6]> = HashMap::with_capacity(4);

        const OFFSET: usize = 5;
        // Map to bands and check for out-of-range values in one pass for each time unit

        for tu in &TIME_UNITS {
            let mut band_counts = [0usize; 6];
            let mut all_in_range = true;

            for val in &values {
                match Self::ladder_band(*val, tu) {
                    Some(band) => band_counts[(band - OFFSET as u8) as usize] += 1,
                    None => {
                        all_in_range = false;
                        break;
                    }
                }
            }

            if all_in_range {
                time_bands.insert(*tu, band_counts);
            }
        }

        stats.timestamp_bands = Some(time_bands);

        // Scalar stats
        let total = values.iter().sum::<f64>();

        stats.is_monotonic = detect_monotonicity(vals)?;
        stats.distinct_count = distinct_set.len();
        stats.max_val = Some(max_val);
        stats.min_val = Some(min_val);
        stats.all_int_bools = Some(all_bool);
        stats.all_integers = Some(all_integers);
        stats.avg = total as f64 / stats.sample_size as f64;

        // Temper the entropy with distinct ratio
        stats.entropy = Self::normalized_float_entropy(values.iter().copied())
            * (distinct_set.len() as f32 / stats.sample_size as f32);

        stats.kurtosis = kurtosis(&values, stats.sample_size, stats.avg);
        stats.skewness = skewness(&values, stats.sample_size, stats.avg);
        stats.histogram_entropy = histogram_entropy(
            &values,
            stats.sample_size,
            Some(6),
            stats.max_val,
            stats.min_val,
        );

        stats.autocorrelation = autocorrelation(&values, stats.sample_size, stats.avg, 1);

        Ok(())
    }

    /// The function `extract_stats_from_string_array` calculates various statistics from a string
    /// array, such as distinct count, character lengths, average length, and entropy.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a mutable reference to a struct or object of type `ColumnStats`.
    ///
    /// Returns:
    ///
    /// The function `extract_stats_from_string_array` is returning a `Result<(), NisabaError>`.
    fn extract_stats_from_string_array(
        stats: &mut FieldStats,
        vals: &dyn Array,
    ) -> Result<(), NisabaError> {
        let values = collect_string_values(vals)?;

        if values.is_empty() {
            return Ok(());
        }

        // [[1, 2], [3, 4]]
        // [1, 2, 3, 4]
        // [1, "text", true]
        // [{"a": 1}, {"b": 2}]
        let all_array = values.iter().all(|s| {
            let val = s.trim();
            if (val.starts_with("[") && val.ends_with("]"))
                || (val.starts_with("[[") && val.ends_with("]]"))
            {
                return true;
            }
            false
        });

        let all_json = values.iter().all(|s| {
            let val = s.trim();
            if val.starts_with("{") && val.ends_with("}") {
                return true;
            }
            false
        });

        // Values that parse as finite `f64` support Float64.
        // Values that are strictly 0.0 or 1.0 also support Boolean
        // Non-parseable values are against both numeric types
        let parsed_floats: Vec<f64> = values
            .iter()
            .filter_map(|v| v.parse::<f64>().ok().filter(|v| v.is_finite()))
            .collect();

        if parsed_floats.len() == values.len() {
            // Classifify: are they all integers? all 0/1?
            let all_integers = parsed_floats.iter().all(|v| v.fract().abs() < f64::EPSILON)
                && !parsed_floats
                    .iter()
                    .all(|v| (0..=1).contains(&(v.round() as i64)));

            stats.all_integers = Some(all_integers);
        }

        // Temporal format checks
        let all_time = values.iter().all(|s| {
            TIME_FMTS
                .iter()
                .any(|fmt| NaiveTime::parse_from_str(s, fmt).is_ok())
        });

        let all_date32 = values.iter().all(|s| {
            DATE_FMTS
                .iter()
                .any(|fmt| NaiveDate::parse_from_str(s, fmt).is_ok())
        });

        let all_stamp = values.iter().all(|s| {
            TIMESTAMP_FMTS
                .iter()
                .any(|fmt| NaiveDateTime::parse_from_str(s, fmt).is_ok())
        });

        // UUID check
        let all_uuid = values.iter().all(|s| {
            let val = s
                .replace("urn:uuid:", "")
                .replace("-", "")
                .replace("{", "")
                .replace("}", "");

            let val = val.trim();

            Uuid::parse_str(val).is_ok()
        });

        // Boolean check
        let all_utf8_bools = values.iter().all(|s| BOOLEAN_FMTS.contains(s));

        // Length statistics
        let mut lengths = values.iter().map(|v| v.len() as f64).collect::<Vec<f64>>();
        lengths.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if stats.sample_size > 20 {
            stats.quantiles_f64 = Some(Self::compute_quantiles(&lengths, 11))
        }

        let mut distinct_set = HashSet::with_capacity(values.len());
        let mut total_len = 0.0;
        let mut max_len = usize::MIN;
        let mut min_len = usize::MAX;
        for val in lengths {
            distinct_set.insert(val as usize);
            total_len += val;
            max_len = max_len.max(val as usize);
            min_len = min_len.min(val as usize);
        }

        stats.char_class_signature = Some(compute_char_class_signature(vals));

        stats.distinct_count = distinct_set.len();
        stats.character_max_length = Some(max_len as i32);
        stats.character_min_length = Some(min_len as i32);
        stats.avg = total_len / stats.sample_size as f64;
        stats.all_uuid = Some(all_uuid);
        stats.all_array = Some(all_array);
        stats.all_utf8_bools = Some(all_utf8_bools);
        stats.all_json = Some(all_json);
        stats.all_time = Some((DataType::Time32(TimeUnit::Millisecond), all_time));
        stats.all_date32 = Some((DataType::Date32, all_date32));
        stats.all_timestamp = Some((DataType::Timestamp(TimeUnit::Second, None), all_stamp));
        stats.entropy = Self::normalized_string_entropy(values.iter().map(|v| Some(*v)));

        Ok(())
    }

    fn extract_stats_from_timestamp_array(
        stats: &mut FieldStats,
        array: &dyn Array,
        tu: TimeUnit,
    ) -> Result<(), NisabaError> {
        // Date32 alignment check for this specific time unit
        Self::extract_date32_flag_from_timestamp_array(stats, array, tu)?;

        // Re-using numeric path
        Self::extract_stats_from_numeric_array(stats, array)?;

        // Monotonicity on the typed array (avoids a second cast)
        stats.is_monotonic = detect_monotonicity(array)?;

        Ok(())
    }

    fn extract_stats_from_boolean_array(
        stats: &mut FieldStats,
        array: &dyn Array,
    ) -> Result<(), NisabaError> {
        let arr = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to BooleanArray".into(),
            ))?;

        let mut true_count = 0usize;
        let mut false_count = 0usize;

        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            if arr.value(i) {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }

        let has_true = true_count > 0;
        let has_false = false_count > 0;

        stats.distinct_count = has_true as usize + has_false as usize;
        stats.all_int_bools = Some(true);
        stats.char_class_signature = Some(ColSignature::Boolean);

        // Represent true=1.0, false=0.0 so min/max are meaningful
        stats.min_val = Some(if has_false { 0.0 } else { 1.0 });
        stats.max_val = Some(if has_true { 1.0 } else { 0.0 });

        // Shannon entropy over the two outcome probabilities
        let n = stats.sample_size as f32;
        let entropy =
            [true_count, false_count]
                .iter()
                .filter(|c| **c > 0)
                .fold(0.0f32, |acc, c| {
                    let p = *c as f32 / n;
                    acc - (p * p.log2())
                });

        stats.entropy = entropy;

        Ok(())
    }

    fn extract_stats_from_date32_array(
        stats: &mut FieldStats,
        array: &dyn Array,
    ) -> Result<(), NisabaError> {
        let arr = array
            .as_any()
            .downcast_ref::<Date32Array>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to Date32Array".into(),
            ))?;

        let mut values: Vec<f64> = arr.iter().flatten().map(|v| v as f64).collect();

        if values.is_empty() {
            return Ok(());
        }

        let mut distinct: HashSet<i32> = HashSet::with_capacity(values.len());

        for i in 0..arr.len() {
            if !arr.is_null(i) {
                distinct.insert(arr.value(i));
            }
        }

        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_v = values[0];
        let max_v = *values.last().unwrap();

        stats.distinct_count = distinct.len();
        stats.max_val = Some(max_v);
        stats.min_val = Some(min_v);
        stats.is_monotonic = detect_monotonicity(array)?;
        stats.char_class_signature = Some(ColSignature::Temporal);
        stats.histogram_entropy = histogram_entropy(
            &values,
            stats.sample_size,
            Some(6),
            Some(max_v),
            Some(min_v),
        );

        if stats.sample_size > 20 {
            stats.quantiles_f64 = Some(Self::compute_quantiles(&values, 11));
        }

        stats.p10 = Some(Self::percentile(&values, 0.1));
        stats.p90 = Some(Self::percentile(&values, 0.9));

        Ok(())
    }

    fn extract_stats_from_time_array(
        stats: &mut FieldStats,
        array: &dyn Array,
    ) -> Result<(), NisabaError> {
        let as_i64 = cast(array, &DataType::Int64)?;
        let arr = as_i64
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or(ArrowError::CastError("Failed to cast to Int64Array".into()))?;

        let mut values: Vec<f64> = arr.iter().flatten().map(|v| v as f64).collect();

        if values.is_empty() {
            return Ok(());
        }

        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut distinct: HashSet<i64> = HashSet::with_capacity(values.len());
        for i in 0..arr.len() {
            if !arr.is_null(i) {
                distinct.insert(arr.value(i));
            }
        }

        let min_v = values[0];
        let max_v = *values.last().unwrap();

        stats.distinct_count = distinct.len();
        stats.max_val = Some(max_v);
        stats.min_val = Some(min_v);
        stats.is_monotonic = detect_monotonicity(array)?;
        stats.char_class_signature = Some(ColSignature::Temporal);
        stats.histogram_entropy = histogram_entropy(
            &values,
            stats.sample_size,
            Some(6),
            Some(max_v),
            Some(min_v),
        );

        if stats.sample_size > 20 {
            stats.quantiles_f64 = Some(Self::compute_quantiles(&values, 11));
        }

        stats.p10 = Some(Self::percentile(&values, 0.1));
        stats.p90 = Some(Self::percentile(&values, 0.9));

        Ok(())
    }

    fn extract_stats_from_binary_array(
        stats: &mut FieldStats,
        array: &dyn Array,
    ) -> Result<(), NisabaError> {
        macro_rules! gather {
            ($ty:ty) => {{
                let arr = array
                    .as_any()
                    .downcast_ref::<$ty>()
                    .ok_or(ArrowError::CastError(
                        concat!("Failed to cast to ", stringify!($ty)).into(),
                    ))?;

                let mut lengths: Vec<usize> = Vec::with_capacity(arr.len());
                let mut distinct: HashSet<&[u8]> = HashSet::new();
                let mut total_len = 0usize;
                let mut max_len = 0usize;
                let mut min_len = usize::MAX;

                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }

                    let b = arr.value(i);
                    distinct.insert(b);
                    let l = b.len();
                    total_len += l;
                    lengths.push(l);

                    max_len = max_len.max(l);
                    min_len = min_len.min(l);
                }

                (distinct.len(), total_len, max_len, min_len, lengths)
            }};
        }

        let (distinct_count, total_len, max_len, min_len, lengths) = match array.data_type() {
            DataType::Binary => gather!(GenericBinaryArray<i32>),
            DataType::LargeBinary => gather!(LargeBinaryArray),
            DataType::BinaryView => gather!(BinaryViewArray),
            DataType::FixedSizeBinary(_) => gather!(FixedSizeBinaryArray),
            other => {
                return Err(NisabaError::Invalid(format!(
                    "extract_stats_from_binary_array: unexpected type {other:?}"
                )));
            }
        };

        if lengths.is_empty() {
            return Ok(());
        }

        stats.distinct_count = distinct_count;
        stats.avg = total_len as f64 / lengths.len() as f64;
        stats.character_max_length = Some(max_len as i32);
        stats.character_min_length = Some(min_len as i32);

        stats.char_class_signature = Some(compute_char_class_signature(array));

        // Entropy over byte-length distribution (a reasonable proxy for binary data)
        let mut len_counts: HashMap<usize, usize> = HashMap::new();
        for l in &lengths {
            *len_counts.entry(*l).or_insert(0) += 1;
        }

        let total = lengths.len() as f32;
        let raw_entropy = len_counts.values().fold(0.0f32, |acc, c| {
            let p = *c as f32 / total;
            if p > 0.0 { acc - p * p.log2() } else { acc }
        });

        let n_buckets = len_counts.len() as f32;
        stats.entropy = if n_buckets > 1.0 {
            raw_entropy / n_buckets.log2()
        } else {
            0.0
        };

        Ok(())
    }

    fn extract_date32_flag_from_timestamp_array(
        stats: &mut FieldStats,
        array: &dyn Array,
        tu: TimeUnit,
    ) -> Result<(), NisabaError> {
        let mut all_time_date32: HashMap<TimeUnit, bool> = HashMap::with_capacity(4);

        let units_per_day = match tu {
            TimeUnit::Second => 86_400i64,
            TimeUnit::Millisecond => 86_400_000,
            TimeUnit::Microsecond => 86_400_000_000,
            TimeUnit::Nanosecond => 86_400_000_000_000,
        };

        macro_rules! check {
            ($ty:ty, $val:expr, $msg:literal, ) => {{
                let arr = array
                    .as_any()
                    .downcast_ref::<$ty>()
                    .ok_or(ArrowError::CastError($msg.into()))?;
                arr.iter().flatten().all(|v| v % $val == 0)
            }};
        }

        let aligned = match tu {
            TimeUnit::Second => check!(
                TimestampSecondArray,
                units_per_day,
                "Failed to cast to TimestampSecondArray",
            ),
            TimeUnit::Millisecond => check!(
                TimestampMillisecondArray,
                units_per_day,
                "Failed to cast to TimestampMillisecondArray",
            ),
            TimeUnit::Microsecond => check!(
                TimestampMicrosecondArray,
                units_per_day,
                "Failed to cast to TimestampMicrosecondArray",
            ),
            TimeUnit::Nanosecond => check!(
                TimestampNanosecondArray,
                units_per_day,
                "Failed to cast to TimestampNanosecondArray",
            ),
        };

        all_time_date32.insert(tu, aligned);
        stats.all_timestamp_date32 = Some(all_time_date32);

        Ok(())
    }

    /// The function flattens the iterator to iterate over the actual strings and then calculates the entropy based
    /// on the character frequencies in those strings.
    ///
    /// Arguments:
    ///
    /// * `values`: The `values` parameter is an iterator that yields `Option<&str>` values.
    ///
    /// Returns:
    ///
    /// The function `normalized_string_entropy` returns a floating-point value representing the
    /// normalized entropy of the input strings provided as an iterator of optional string references.
    fn normalized_string_entropy<'b>(values: impl Iterator<Item = Option<&'b str>>) -> f32 {
        let mut incident_counts: HashMap<char, usize> = HashMap::new();
        let mut total_incidents = 0;

        for val in values.flatten() {
            for ch in val.chars() {
                *incident_counts.entry(ch).or_insert(0) += 1;
                total_incidents += 1;
            }
        }

        if total_incidents == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;

        for count in incident_counts.values() {
            let p = *count as f32 / total_incidents as f32;

            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy / (incident_counts.len() as f32).log2()
    }

    /// The function `normalized_numeric_entropy` calculates the normalized entropy of a sequence of
    /// integers based on the frequency of different elements.
    ///
    /// Arguments:
    ///
    /// * `values`: This parameter is an iterator that yields integer values.
    ///
    /// Returns:
    ///
    /// The function `normalized_numeric_entropy` returns a floating-point value (`f32`) representing the
    /// normalized entropy of the input integer values provided by the iterator.
    fn normalized_numeric_entropy<T: DeltaKey + Copy>(values: impl Iterator<Item = T>) -> f32 {
        let mut incident_counts: HashMap<T::Key, usize> = HashMap::new();
        let mut total_incidents = 0;

        let mut iter = values.into_iter();
        let mut prev = match iter.next() {
            Some(v) => v,
            None => return 0.0,
        };

        for curr in iter {
            let delta = T::delta(curr, prev);
            *incident_counts.entry(delta).or_insert(0) += 1;
            total_incidents += 1;
            prev = curr;
        }

        if total_incidents == 0 {
            return 0.0;
        }

        if incident_counts.len() as f32 <= 1.0 {
            return 0.0;
        }

        let mut entropy = 0.0f32;

        for count in incident_counts.values() {
            let p = *count as f32 / total_incidents as f32;

            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy / (incident_counts.len() as f32).log2()
    }

    fn normalized_float_entropy(values: impl Iterator<Item = f64>) -> f32 {
        Self::normalized_numeric_entropy(values)
    }

    fn compute_quantiles(sorted: &[f64], n_quantiles: usize) -> Vec<f64> {
        let n = sorted.len();

        (1..=n_quantiles)
            .map(|i| {
                let rank = i as f64 * (n as f64 - 1.0) / (n_quantiles as f64 + 1.0);
                let lower = rank.floor() as usize;
                let upper = (lower + 1).min(n - 1);
                let frac = rank - lower as f64;
                sorted[lower] * (1.0 - frac) + sorted[upper] * frac
            })
            .collect()
    }

    /// Returns the value at the given quantile (0.0-1.0) from sorted slice
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);

        sorted[idx]
    }

    fn compute_delta_gcd(values: &[f64]) -> Option<i64> {
        values
            .windows(2)
            .filter_map(|w| {
                let delta = (w[1] as i64) - (w[0] as i64);
                if delta != 0 { Some(delta.abs()) } else { None }
            })
            .reduce(Self::gcd)
    }

    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let tmp = b;
            b = a % b;
            a = tmp;
        }
        a.abs()
    }

    const fn ladder_band(x: f64, unit: &TimeUnit) -> Option<u8> {
        let scale = match unit {
            TimeUnit::Second => 1.0,
            TimeUnit::Millisecond => 1_000.0,
            TimeUnit::Microsecond => 1_000_000.0,
            TimeUnit::Nanosecond => 1_000_000_000.0,
        };

        let abs_x = x.abs();

        match abs_x / scale {
            val if val >= 0.0 && val <= 99_999.0 => Some(5),
            val if val >= 100_000.0 && val <= 999_999.0 => Some(6),
            val if val >= 1_000_000.0 && val <= 9_999_999.0 => Some(7),
            val if val >= 10_000_000.0 && val <= 99_999_999.0 => Some(8),
            val if val >= 100_000_000.0 && val <= 999_999_999.0 => Some(9),
            val if val >= 1_000_000_000.0 && val <= 9_999_936_000.0 => Some(10),
            _ => None,
        }
    }
}

trait DeltaKey {
    type Key: Hash + Eq + Copy;
    fn delta(a: Self, b: Self) -> Self::Key;
}

impl DeltaKey for i64 {
    type Key = i64;
    fn delta(a: Self, b: Self) -> Self::Key {
        a - b
    }
}

impl DeltaKey for f64 {
    type Key = u64;
    fn delta(a: Self, b: Self) -> Self::Key {
        (a - b).to_bits()
    }
}

// ===============================
// Metric Computation Functions
// ===============================

/// Description of character-class composition of an array's values
#[derive(Clone, Debug, PartialEq)]
pub enum ColSignature {
    /// Column contains boolean values.
    Boolean,
    /// Ratio of [digits, alpha/letters, whitespace, punctuation, special_chars] chars; slots sum to 1.0.
    /// Produced for all string-like and binary-interpreted-as-text types.
    CharClass([f32; 5]),
    /// Column is a nested or structured type (list, struct, map, union).
    /// Char-class analysis would require recursive descent; not attempted here.
    Nested,
    /// Column contains numeric data (integers, floats, decimals).
    /// Char-class ratios are undefined.
    Numeric,
    /// Null-only column, or a type we genuinely cannot inspect.
    Opaque,
    /// Column contains temporal data (dates, times, timestamps, durations, intervals).
    Temporal,
}

impl ColSignature {
    pub fn discriminant(&self) -> &'static str {
        match self {
            Self::Boolean => "bool",
            Self::CharClass(_) => "charclass",
            Self::Nested => "nested",
            Self::Numeric => "numeric",
            Self::Opaque => "opaque",
            Self::Temporal => "temporal",
        }
    }

    /// Deserialize from the two component values produced `to_arrow_arrays`
    pub fn from_arrow(kind: &str, char_class: Option<[f32; 5]>) -> Result<Self, NisabaError> {
        match kind {
            "bool" => Ok(Self::Boolean),
            "charclass" => Ok(Self::CharClass(char_class.unwrap())),
            "nested" => Ok(Self::Nested),
            "numeric" => Ok(Self::Numeric),
            "opaque" => Ok(Self::Opaque),
            "temporal" => Ok(Self::Temporal),
            other => Err(NisabaError::Invalid(format!(
                "Unknown ColSignature kind: {}",
                other
            ))),
        }
    }
}

/// The function `compute_char_class_signature` calculates the distribution of character classes in a
/// string array.
///
/// Arguments:
///
/// * `samples`: The function `compute_char_class_signature` takes a reference to a trait object
///   `samples` that implements the `Array` trait. The goal of this function is to compute a signature
///   based on the character classes present in the data contained in the `samples` array.
///
/// Returns:
///
/// The function `compute_char_class_signature` returns an array of 4 floating-point numbers `[f32; 4]`.
/// The array contains the ratios of different character classes (digits, alphabetic characters,
/// whitespace characters, and other characters) found in the input samples.
pub fn compute_char_class_signature(samples: &dyn Array) -> ColSignature {
    classify_array(samples)
}

fn classify_array(samples: &dyn Array) -> ColSignature {
    match samples.data_type() {
        // Null
        DataType::Null => ColSignature::Opaque,
        // Boolean
        DataType::Boolean => ColSignature::Boolean,

        // Numeric
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => ColSignature::Numeric,

        // Temporal
        DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Date32
        | DataType::Date64
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Timestamp(_, _) => ColSignature::Temporal,

        // Strings
        DataType::Utf8 => char_class_from_string_array::<i32>(samples),
        DataType::LargeUtf8 => char_class_from_string_array::<i64>(samples),
        DataType::Utf8View => char_class_from_string_view_array(samples),

        // Binary (attempt Utf8 interpretation)
        DataType::Binary => char_class_from_binary_array::<i32>(samples),
        DataType::LargeBinary => char_class_from_binary_array::<i64>(samples),
        DataType::BinaryView => char_class_from_binary_view_array(samples),
        DataType::FixedSizeBinary(_) => char_class_from_fixed_binary_array(samples),

        // Nested / Structure
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::Struct(_)
        | DataType::Union(_, _)
        | DataType::Map(_, _)
        | DataType::Dictionary(_, _)
        | DataType::RunEndEncoded(_, _) => ColSignature::Nested,
    }
}

fn char_class_from_iter<'a>(
    len: usize,
    get_value: impl Fn(usize) -> Option<&'a str>,
    require_any_utf8: bool,
) -> ColSignature {
    let mut totals = [0f32; 5];
    let mut char_count = 0f32;
    let mut any_valid = false;

    for i in 0..len {
        if let Some(s) = get_value(i) {
            any_valid = true;
            count_chars(s, &mut totals, &mut char_count);
        }
    }

    if require_any_utf8 && !any_valid {
        return ColSignature::Opaque;
    }
    finalise_col_signature(totals, char_count)
}

fn char_class_from_string_array<O: OffsetSizeTrait>(samples: &dyn Array) -> ColSignature {
    let Some(arr) = samples.as_any().downcast_ref::<GenericStringArray<O>>() else {
        return ColSignature::Opaque;
    };

    char_class_from_iter(
        arr.len(),
        |i| (!arr.is_null(i)).then(|| arr.value(i)),
        false,
    )
}

fn char_class_from_string_view_array(samples: &dyn Array) -> ColSignature {
    let Some(arr) = samples.as_any().downcast_ref::<StringViewArray>() else {
        return ColSignature::Opaque;
    };

    char_class_from_iter(
        arr.len(),
        |i| (!arr.is_null(i)).then(|| arr.value(i)),
        false,
    )
}

fn char_class_from_binary_array<O: OffsetSizeTrait>(samples: &dyn Array) -> ColSignature {
    let Some(arr) = samples.as_any().downcast_ref::<GenericBinaryArray<O>>() else {
        return ColSignature::Opaque;
    };

    char_class_from_iter(
        arr.len(),
        |i| {
            (!arr.is_null(i))
                .then(|| std::str::from_utf8(arr.value(i)).ok())
                .flatten()
        },
        true,
    )
}

fn char_class_from_binary_view_array(samples: &dyn Array) -> ColSignature {
    let Some(arr) = samples.as_any().downcast_ref::<BinaryViewArray>() else {
        return ColSignature::Opaque;
    };

    char_class_from_iter(
        arr.len(),
        |i| {
            (!arr.is_null(i))
                .then(|| std::str::from_utf8(arr.value(i)).ok())
                .flatten()
        },
        true,
    )
}

fn char_class_from_fixed_binary_array(samples: &dyn Array) -> ColSignature {
    let Some(arr) = samples.as_any().downcast_ref::<FixedSizeBinaryArray>() else {
        return ColSignature::Opaque;
    };

    char_class_from_iter(
        arr.len(),
        |i| {
            (!arr.is_null(i))
                .then(|| std::str::from_utf8(arr.value(i)).ok())
                .flatten()
        },
        true,
    )
}

fn finalise_col_signature(totals: [f32; 5], char_count: f32) -> ColSignature {
    if char_count == 0.0 {
        // All values were null or empty strings - still a string column
        // but nothing to measure
        return ColSignature::CharClass([0.0; 5]);
    }

    ColSignature::CharClass(totals.map(|t| t / char_count))
}

fn count_chars(s: &str, totals: &mut [f32; 5], char_count: &mut f32) {
    for ch in s.chars() {
        *char_count += 1.0;

        if ch.is_ascii_digit() {
            totals[0] += 1.0;
        } else if ch.is_alphabetic() {
            totals[1] += 1.0
        } else if ch.is_ascii_whitespace() {
            totals[2] += 1.0
        } else if ch.is_ascii_punctuation() {
            totals[3] += 1.0;
        } else {
            totals[4] += 1.0
        }
    }
}

/// The function `detect_monotonicity` checks if the given array of samples is monotonically increasing.
///
/// Arguments:
///
/// * `samples`: The `detect_monotonicity` function takes a reference to a trait object `Array` as
///   input, which represents an array of samples with different data types. The function checks if the
///   samples are monotonically increasing based on their data type.
///
/// Returns:
///
/// The `detect_monotonicity` function returns a boolean value indicating whether the samples provided
/// in the input array are monotonic (i.e., always increasing) or not.
pub fn detect_monotonicity(samples: &dyn Array) -> Result<bool, NisabaError> {
    if samples.is_empty() {
        return Ok(false);
    }

    if samples.len() < 2 {
        return Ok(false);
    }

    match samples.data_type() {
        DataType::Int8
        | DataType::UInt8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64
        | DataType::Timestamp(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => {
            let samples = cast(samples, &DataType::Float64)?;
            let samples = samples.as_primitive::<Float64Type>();

            let mut values: Vec<f64> = (0..samples.len())
                .filter(|&i| !samples.is_null(i))
                .map(|i| samples.value(i))
                .collect();

            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));

            Ok(values.windows(2).all(|w| w[1] >= w[0]))
        }

        DataType::Date32 | DataType::Date64 => {
            let samples = cast(samples, &DataType::Int64)?;
            let timestamps = samples.as_primitive::<Int64Type>();

            let mut values: Vec<i64> = (0..timestamps.len())
                .filter(|&i| !timestamps.is_null(i))
                .map(|i| timestamps.value(i))
                .collect();

            values.sort_unstable();
            Ok(values.windows(2).all(|w| w[1] >= w[0]))
        }
        _ => Ok(false),
    }
}

/// The function `compute_cardinality` calculates the cardinality ratio of unique values in a given
/// array of data samples.
///
/// Arguments:
///
/// * `samples`: The function `compute_cardinality` takes a reference to a dynamic array `samples` as
///   input and calculates the cardinality of the data in the array. The function uses different logic
///   based on the data type of the array elements to determine the unique count of values in the array.
///
/// Returns:
///
/// The function `compute_cardinality` returns a `Result<f32, NisabaError>`, where the `Ok` variant
/// contains the calculated cardinality as a floating-point number (`f64`).
pub fn compute_cardinality(samples: &dyn Array) -> Result<f64, NisabaError> {
    if samples.is_empty() {
        return Ok(0.0);
    }

    if samples.len() < 2 {
        return Ok(1.0);
    }

    macro_rules! count_unique_vals {
        ($arr:expr, $array_type:ty) => {{
            let arr = $arr
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or(ArrowError::CastError(format!(
                    "Failed to cast to {}",
                    stringify!($array_type)
                )))?;

            let mut values: HashSet<_> = HashSet::new();

            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.insert(arr.value(i));
                }
            }

            values.len()
        }};
    }

    let unique_count = match samples.data_type() {
        DataType::Null => 0,
        DataType::Binary => {
            let values: HashSet<&[u8]> = samples.as_binary::<i32>().iter().flatten().collect();
            values.len()
        }
        DataType::BinaryView => {
            let values: HashSet<&[u8]> = samples.as_binary_view().iter().flatten().collect();

            values.len()
        }
        DataType::LargeBinary => count_unique_vals!(samples, LargeBinaryArray),
        DataType::Boolean => 2,
        DataType::Utf8 | DataType::Utf8View => {
            let values: HashSet<_> = samples.as_string::<i32>().iter().flatten().collect();

            values.len()
        }
        DataType::LargeUtf8 => count_unique_vals!(samples, LargeStringArray),
        DataType::Int8 => count_unique_vals!(samples, Int8Array),
        DataType::Int16 => count_unique_vals!(samples, Int16Array),
        DataType::Int32 => count_unique_vals!(samples, Int32Array),
        DataType::Int64 => count_unique_vals!(samples, Int64Array),
        DataType::UInt8 => count_unique_vals!(samples, UInt8Array),
        DataType::UInt16 => count_unique_vals!(samples, UInt16Array),
        DataType::UInt32 => count_unique_vals!(samples, UInt32Array),
        DataType::UInt64 => count_unique_vals!(samples, UInt64Array),
        DataType::Date32 => count_unique_vals!(samples, Date32Array),
        DataType::FixedSizeBinary(_) => count_unique_vals!(samples, FixedSizeBinaryArray),

        // Safe assumption that cardinality in these types is near perfect if not perfect
        DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Dictionary(_, _)
        | DataType::FixedSizeList(_, _)
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::RunEndEncoded(_, _)
        | DataType::Struct(_)
        | DataType::Union(_, _)
        | DataType::Duration(_)
        | DataType::Map(_, _)
        | DataType::List(_)
        | DataType::ListView(_)
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Interval(_)
        | DataType::LargeListView(_)
        | DataType::LargeList(_)
        | DataType::Time32(_)
        | DataType::Time64(_) => samples.len(),
    };

    let cc = unique_count as f64 / samples.len() as f64;

    Ok(cc)
}

/// The function `compute_avg_byte_length` calculates the average byte length of strings or binary data
/// in an array.
///
/// Arguments:
///
/// * `samples`: The `compute_avg_byte_length` function takes a reference to a trait object `samples`
///   that implements the `Array` trait. The function calculates the average byte length of the elements
///   in the array based on the data type of the array.
///
/// Returns:
///
/// The function `compute_avg_byte_length` returns a `Result` containing an `Option<f64>` or a
/// `NisabaError`.
pub fn compute_avg_byte_length(samples: &dyn Array) -> Result<Option<f64>, NisabaError> {
    match samples.data_type() {
        DataType::Utf8 => {
            let string_arr =
                samples
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or(ArrowError::CastError(
                        "Failed to cast array to string".into(),
                    ))?;
            let total_len: usize = string_arr.iter().flatten().map(|s| s.len()).sum();

            let count = string_arr.len() - string_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f64 / count as f64))
        }
        DataType::LargeUtf8 => {
            let string_arr = samples.as_any().downcast_ref::<LargeStringArray>().ok_or(
                ArrowError::CastError("Failed to cast array to large string".into()),
            )?;

            let total_len: usize = string_arr.iter().flatten().map(|s| s.len()).sum();

            let count = string_arr.len() - string_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f64 / count as f64))
        }
        DataType::Binary => {
            let binary_arr = samples
                .as_any()
                .downcast_ref::<GenericBinaryArray<i32>>()
                .ok_or(ArrowError::CastError(
                    "Failed to cast array to binary array".into(),
                ))?;

            let total_len: usize = binary_arr.iter().flatten().map(|s| s.len()).sum();

            let count = binary_arr.len() - binary_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f64 / count as f64))
        }

        DataType::LargeBinary => {
            let binary_arr = samples.as_any().downcast_ref::<LargeBinaryArray>().ok_or(
                ArrowError::CastError("Failed to cast array to binary array".into()),
            )?;

            let total_len: usize = binary_arr.iter().flatten().map(|s| s.len()).sum();

            let count = binary_arr.len() - binary_arr.null_count();

            if count == 0 {
                return Ok(None);
            }

            Ok(Some(total_len as f64 / count as f64))
        }

        _ => Ok(None),
    }
}

fn collect_string_values(array: &dyn Array) -> Result<Vec<&str>, NisabaError> {
    match array.data_type() {
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::CastError(
                    "Failed to cast to StringArray".into(),
                ))?;

            Ok(arr.iter().flatten().collect())
        }
        DataType::LargeUtf8 => {
            let arr =
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or(ArrowError::CastError(
                        "Failed to cast to LargeStringArray".into(),
                    ))?;

            Ok(arr.iter().flatten().collect())
        }
        DataType::Utf8View => {
            let arr =
                array
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or(ArrowError::CastError(
                        "Failed to cast to StringViewArray".into(),
                    ))?;

            Ok(arr.iter().flatten().collect())
        }
        other => Err(NisabaError::Invalid(format!(
            "collect_string_values: unsupported type {other:?}"
        ))),
    }
}
