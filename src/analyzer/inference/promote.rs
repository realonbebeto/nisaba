use arrow::{
    array::{Array, ArrayRef, FixedSizeBinaryBuilder, Int64Array, RecordBatch, StringArray},
    compute::{CastOptions, cast_with_options},
    datatypes::{DataType, Field, Schema, TimeUnit},
    error::ArrowError,
};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use uuid::Uuid;

use crate::error::NisabaError;

// ====================================
// Casting Safety Analysis
// ====================================
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum CastSafety {
    /// No data loss, always succeedds
    /// No failure, no semantic change, but representation differs
    /// e.g. Date32 ↔ Timestamp(day)
    Safe,
    /// Potential precision loss or truncation
    /// May succeed, May fail, depending on values, but no loss if it succeeds but still information can be lost
    /// e.g. f64 → f32, i64 → i32 (truncate)
    /// e.g. string → uuid, string → date
    /// e.g. string → int, float → int
    FallibleLossy,
    /// May fail at runtime
    // Semantically dangerous even if it succeeds
    Unsafe,
}

impl CastSafety {
    /// Convert to numeric score for graph weighting (.0, 1.)
    pub fn score(&self) -> f32 {
        match self {
            Self::Safe => 0.98,
            Self::FallibleLossy => 0.86,
            &Self::Unsafe => 0.48,
        }
    }
}

/// The function `cast_safety` in determines the safety of casting between different data types
/// based on specific rules for numeric, timestamp, date, and string/JSON conversions.
///
/// Arguments:
///
/// * `from`: The `from` parameter in the `cast_safety` function represents the data type that you want
///   to cast from. It is a reference to a `DataType` enum which specifies the original data type of the
///   value you want to convert.
/// * `to`: The `to` parameter in the `cast_safety` function represents the data type that you want to
///   cast to.
///
/// Returns:
///
/// The function `cast_safety` returns a value of type `CastSafety`, which indicates the safety level of
/// casting from one data type to another. The possible return values are `Safe`, `FallibleLossy`, or
/// `Unsafe` based on the specific conversion rules defined in the function.
pub fn cast_safety(from: &DataType, to: &DataType) -> CastSafety {
    // Exact match is always safe
    if from == to {
        return CastSafety::Safe;
    }

    match (from, to) {
        //Widening Numeric Conversions
        (DataType::Int8, DataType::Int16 | DataType::Int32 | DataType::Int64) => CastSafety::Safe,

        (DataType::Int16, DataType::Int32 | DataType::Int64) => CastSafety::Safe,

        (DataType::Int32, DataType::Int64) => CastSafety::Safe,

        // Narrowing Numeric Conversions
        (DataType::Int64, DataType::Int32 | DataType::Int16 | DataType::Int8) => {
            CastSafety::FallibleLossy
        }

        (
            DataType::Decimal32(_, _),
            DataType::Decimal64(_, _) | DataType::Decimal128(_, _) | DataType::Decimal256(_, _),
        ) => CastSafety::Safe,

        //Timestamp conversions
        (DataType::Int64, DataType::Timestamp { .. }) => CastSafety::FallibleLossy,
        (DataType::Timestamp { .. }, DataType::Int64) => CastSafety::Safe,
        (DataType::Timestamp(u1, ..), DataType::Timestamp(u2, ..)) if u1 == u2 => CastSafety::Safe,
        (DataType::Timestamp { .. }, DataType::Timestamp { .. }) => CastSafety::FallibleLossy,

        // Date conversions
        (DataType::Int32, DataType::Date32) => CastSafety::FallibleLossy,
        (DataType::Date32, DataType::Int32) => CastSafety::Safe,
        (DataType::Date32, DataType::Date64) => CastSafety::Safe,
        (DataType::Date64, DataType::Date32) => CastSafety::FallibleLossy,

        // String/JSON conversions
        (DataType::Utf8, DataType::List { .. }) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Struct { .. }) => CastSafety::FallibleLossy,
        (DataType::Binary, DataType::Utf8) => CastSafety::Unsafe,
        (DataType::Utf8, DataType::Binary) => CastSafety::Safe,

        (DataType::FixedSizeBinary(16), DataType::Utf8) => CastSafety::Safe,
        (DataType::Utf8, DataType::FixedSizeBinary(16)) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Timestamp(_, _)) => CastSafety::FallibleLossy,
        (
            DataType::Utf8,
            DataType::Date32 | DataType::Date64 | DataType::Time32(_) | DataType::Time64(_),
        ) => CastSafety::FallibleLossy,
        (DataType::Binary, DataType::FixedSizeBinary(16)) => CastSafety::FallibleLossy,

        _ => CastSafety::Unsafe,
    }
}

#[derive(Debug, Clone)]
/// The `DeltaStats`  contains fields for various statistical ratios and values related to
/// delta calculations.
///
/// Properties:
///
/// * `small_delta_ratio`: The `small_delta_ratio` property in `DeltaStats` represents the
///   ratio of deltas that have an absolute value less than or equal to a specified threshold, typically
///   denoted as `small_threshold`.
/// * `mode_ratio`: The `mode_ratio` property in `DeltaStats` represents the fraction of
///   deltas that are equal to the mode_delta value. This value indicates how frequently the mode_delta
///   value appears in the dataset compared to other delta values.
/// * `long_run_ratio`: The `long_run_ratio` property in `DeltaStats` represents the fraction
///   of rows belonging to the longest uninterrupted delta run. This means it indicates the proportion of
///   consecutive rows in the data where the delta values remain the same without interruption.
/// * `median_abs_delta`: The `median_abs_delta` property in `DeltaStats` represents the
///   median value of the absolute deltas in a dataset. This value is calculated by arranging all the
///   absolute delta values in ascending order and then selecting the middle value.
pub struct DeltaStats {
    /// Ratio of |delta| ≤ small_threshold (usually 1 or unit-sized)
    pub small_delta_ratio: f32,
    /// Fraction of deltas equal to mode_delta
    pub mode_ratio: f32,
    /// Fraction of rows belonging to longest uninterrupted delta run
    pub long_run_ratio: f32,
    /// Median of absolute deltas
    pub median_abs_delta: f32,
}

/// The `ColumnStats` represents statistical information about a column in a dataset.
///
/// Properties:
///
/// * `sample_size`: The `sample_size` property in `ColumnStats` represents the total number
///   of values in the column that were used to calculate the statistics.
/// * `null_count`: The `null_count` property in `ColumnStats` represents the number of null
///   values present in the column for which the statistics are being calculated.
/// * `distinct_count`: The `distinct_count` property in `ColumnStats` represents the number
///   of unique or distinct values present in the column for which the statistics are being calculated.
/// * `avg_length`: The `avg_length` property in `ColumnStats` represents the average length
///   of values in the column. It is an `Option<f32>`, which means it can either contain the average
///   length as a floating-point number or be `None` if the average length when applicable.
/// * `sample_values`: The `sample_values` property in `ColumnStats` is a reference to a
///   dynamic array (`dyn Array`) with an associated lifetime `'a`. This allows you to store a collection
///   of values of unknown type that implement the `Array` trait.
/// * `min_val`: The `min_val` property in `ColumnStats` represents the minimum value found
///   in the column for which these statistics are calculated. It is an optional field, meaning it may or
///   may not have a value..
/// * `max_val`: The `max_val` property in `ColumnStats` represents the maximum value found
///   in the column for which the statistics are being calculated. It is an `Option<i64>`, meaning it can
///   either contain the maximum value as an `i64` or be `None`.
/// * `quantiles_i32`: The `quantiles_i32` property in `ColumnStats` represents an optional
///   array of 7 integers. These integers correspond to different percentiles for the data in the column.
/// * `longest_run_ratio`: The `longest_run_ratio` property in `ColumnStats` represents the
///   ratio of the length of the longest consecutive run of identical values to the total number of values
///   in the column. It can be used to analyze the presence of patterns or repeated values within the data.
/// * `delta_stats`: The `delta_stats` property in `ColumnStats` likely represents statistics
///   related to the differences or changes between consecutive values in the column.
/// * `entropy`: The `entropy` property in `ColumnStats` represents the entropy value of the
///   column. Entropy is a measure of the amount of uncertainty or randomness in the data. In the context
///   of data analysis, entropy is often used to quantify the information content or predictability of a
///   data.
/// * `temporal_mod_entropy`: The `temporal_mod_entropy` property in `ColumnStats` represents
///   the modulo entropy value for temporal(time related) data in the column.
/// * `character_max_length`: The `character_max_length` property in `ColumnStats` represents
///   the maximum length of characters in the column. It indicates the maximum number of characters
///   present in any value within the column.
/// * `character_min_length`: The `character_min_length` property in `ColumnStats` represents
///   the minimum length of characters in the column. It is an optional field, meaning it may or may not
///   have a value associated with it.
/// * `numeric_precision`: The `numeric_precision` property in `ColumnStats` represents the
///   precision of numeric values in the column. It indicates the total number of digits that can be
///   stored, both to the left and right of the decimal point.
/// * `numeric_scale`: The `numeric_scale` property in `ColumnStats` represents the scale of
///   a numeric value. In the context of numeric data types, scale refers to the number of digits to the
///   right of the decimal point in a number.
/// * `datetime_precision`: The `datetime_precision` property in `ColumnStats` represents the
///   precision of datetime values in the column. This property is an `Option<i32>`, meaning it can either
///   contain an integer value representing the precision or be `None` when applicable.
pub struct ColumnStats<'a> {
    pub sample_size: usize,
    pub null_count: usize,
    pub distinct_count: usize,
    pub avg_length: Option<f32>,
    pub sample_values: &'a dyn Array,
    pub min_val: Option<i64>,
    pub max_val: Option<i64>,
    pub quantiles_i32: Option<[i32; 7]>, // p01, p05, p25, p50, p75, p95, p99
    pub longest_run_ratio: Option<f32>,
    pub delta_stats: Option<DeltaStats>,
    pub entropy: f32,
    pub temporal_mod_entropy: Option<f32>,
    pub character_max_length: Option<i32>,
    pub character_min_length: Option<i32>,
    pub numeric_precision: Option<i32>,
    pub numeric_scale: Option<i32>,
    pub datetime_precision: Option<i32>,
}

impl<'a> ColumnStats<'a> {
    /// The function `new` creates an instance of `ColumnStats` from an Arrow array
    /// by means of taking a refence of a type that implements Array trait.
    pub fn new(array: &'a dyn Array) -> Self {
        let sample_size = array.len();
        let null_count = array.null_count();
        let mut stats = Self {
            sample_size,
            null_count,
            distinct_count: 0,
            avg_length: None,
            sample_values: array,
            min_val: None,
            max_val: None,
            quantiles_i32: None,
            longest_run_ratio: None,
            delta_stats: None,
            entropy: 0.0,
            temporal_mod_entropy: None,
            character_max_length: None,
            character_min_length: None,
            numeric_precision: None,
            numeric_scale: None,
            datetime_precision: None,
        };

        if sample_size == 0 || sample_size == null_count {
            return stats;
        }

        match array.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let _ = Self::extract_stats_from_string_array(&mut stats);
            }
            DataType::Int32 | DataType::Int64 => {
                let _ = Self::extract_stats_from_int_array(&mut stats);
            }
            _ => {}
        }

        stats
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

    /// The function `extract_stats_from_int_array` extracts statistics such as distinct count, maximum
    /// value, minimum value, and entropy from an integer array.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `extract_stats_from_int_array` function is a
    ///   mutable reference to a `ColumnStats` struct or object. This function extracts statistics
    ///   from an array of integer values stored in the `sample_values` field of the `ColumnStats` object
    ///
    /// Returns:
    ///
    /// The function `extract_stats_from_int_array` is returning a `Result<(), NisabaError>`.
    fn extract_stats_from_int_array(stats: &mut ColumnStats) -> Result<(), NisabaError> {
        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or(ArrowError::CastError("Failed to cast to Int64Array".into()))?;

        let mut distinct_set = HashSet::new();
        let mut max_val = usize::MIN;
        let mut min_val = usize::MAX;
        for val in values.iter().flatten() {
            distinct_set.insert(val);
            max_val = max_val.max(val as usize);
            min_val = min_val.min(val as usize);
        }

        stats.distinct_count = distinct_set.len();
        stats.max_val = Some(max_val as i64);
        stats.min_val = Some(min_val as i64);
        stats.entropy = Self::normalized_int_entropy(values.iter().flatten());

        Ok(())
    }

    /// The function `extract_stats_from_string_array` calculates various statistics from a string
    /// array, such as distinct count, character lengths, average length, and entropy.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `extract_stats_from_string_array` function seems to be a
    ///   mutable reference to a struct or object of type `ColumnStats`. This function is designed to
    ///   extract various statistics from a sample of string values stored in the `stats` object.
    ///
    /// Returns:
    ///
    /// The function `extract_stats_from_string_array` is returning a `Result<(), NisabaError>`.
    fn extract_stats_from_string_array(stats: &mut ColumnStats) -> Result<(), NisabaError> {
        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

        let mut distinct_set = HashSet::new();
        let mut total_len = 0;
        let mut max_len = usize::MIN;
        let mut min_len = usize::MAX;
        for val in values.iter().flatten() {
            distinct_set.insert(val);
            total_len += val.len();
            max_len = max_len.max(val.len());
            min_len = min_len.min(val.len());
        }

        stats.distinct_count = distinct_set.len();
        stats.character_max_length = Some(max_len as i32);
        stats.character_min_length = Some(min_len as i32);
        stats.avg_length = Some(total_len as f32 / (stats.sample_size - stats.null_count) as f32);
        stats.entropy = Self::normalized_string_entropy(values.iter());

        Ok(())
    }

    /// The function calculates the normalized entropy of a collection of strings.
    ///
    /// Arguments:
    ///
    /// * `values`: The `normalized_string_entropy` function calculates the entropy of a collection of
    ///   strings. The `values` parameter is an iterator that yields `Option<&str>` values.
    ///
    /// The function flattens the iterator to iterate over the actual strings and then calculates the entropy based
    /// on the character frequencies in those strings.
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

    /// The function `normalized_int_entropy` calculates the normalized entropy of a sequence of
    /// integers based on the frequency of different elements.
    ///
    /// Arguments:
    ///
    /// * `values`: The function `normalized_int_entropy` calculates the normalized entropy of a
    ///   sequence of integers. The input parameter `values` is an iterator that yields integer values.
    ///   The function processes the values to compute the entropy based on the frequency of integer values.
    ///
    /// Returns:
    ///
    /// The function `normalized_int_entropy` returns a floating-point value (`f32`) representing the
    /// normalized entropy of the input integer values provided by the iterator.
    fn normalized_int_entropy(values: impl Iterator<Item = i64>) -> f32 {
        let mut incident_counts: HashMap<i64, usize> = HashMap::new();
        let mut total_incidents = 0;

        let mut iter = values.into_iter();
        let mut prev = match iter.next() {
            Some(v) => v,
            None => return 0.0,
        };

        for curr in iter {
            let delta = curr - prev;
            *incident_counts.entry(delta).or_insert(0) += 1;
            total_incidents += 1;
            prev = curr;
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
}

#[derive(Debug, Clone)]
/// The `PromotionResult` represents the result of a data type promotion operation,
/// including destination type, confidence level, and optional metadata.
///
/// Properties:
///
/// * `dest_type`: The `dest_type` property in `PromotionResult` represents the data type to
///   which a value is being promoted or converted.
/// * `confidence`: The `confidence` property in `PromotionResult` represents the level of
///   certainty or belief in the promotion result. It is a floating-point number (`f32`) typically ranging
///   from 0.0 to 1.0, where 1.0 indicates full confidence in the result.
/// * `nullable`: The `nullable` property in `PromotionResult` indicates whether the
///   corresponding data type can accept NULL values or not. If `nullable` is `true`, it means that NULL
///   values are allowed for that data type.
/// * `character_maximum_length`: The `character_maximum_length` property in the `PromotionResult`
///   represents the maximum length of characters for a data type. It is an optional field, meaning
///   it may or may not have a value depending on the data type being described.
/// * `numeric_precision`: The `numeric_precision` property in `PromotionResult` represents
///   the precision of a numeric data type. It specifies the total number of digits that can be stored,
///   including both the digits before and after the decimal point.
/// * `numeric_scale`: The `numeric_scale` property in `PromotionResult` represents the scale
///   of a numeric value. In the context of numeric data types, scale refers to the number of digits to
///   the right of the decimal point in a number.
/// * `datetime_precision`: The `datetime_precision` property in `PromotionResult` represents
///   the precision of a datetime data type. This property is an `Option<i32>`, meaning it can either
///   contain an integer value representing the precision or be `None` if the precision is not applicable
///   or not specified.
pub struct PromotionResult {
    pub dest_type: DataType,
    pub confidence: f32,
    pub nullable: bool,
    pub character_maximum_length: Option<i32>,
    pub numeric_precision: Option<i32>,
    pub numeric_scale: Option<i32>,
    pub datetime_precision: Option<i32>,
}

pub struct TypeLatticeResolver;

impl TypeLatticeResolver {
    pub fn new() -> Self {
        Self
    }

    pub fn promote(
        &self,
        source: &DataType,
        stats: &ColumnStats,
    ) -> Result<PromotionResult, NisabaError> {
        let mut dest_type: DataType = source.clone();
        let mut confidence = 0.98;
        let mut datetime_precision = stats.datetime_precision;
        let mut numeric_scale = stats.numeric_scale;
        let mut numeric_precision = stats.numeric_precision;
        let mut character_maximum_length = stats.character_max_length;

        match source {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let logp1 = self.detect_type_from_string(stats, &DataType::Utf8)?;
                let logp2 = self.detect_type_from_string(stats, &DataType::FixedSizeBinary(16))?;
                let logp3 = self.detect_type_from_string(stats, &DataType::Date32)?;
                let logp4 = self.detect_type_from_string(stats, &DataType::Date64)?;
                let logp5 =
                    self.detect_type_from_string(stats, &DataType::Time32(TimeUnit::Millisecond))?;
                let logp6 = self.detect_type_from_string(
                    stats,
                    &DataType::Timestamp(TimeUnit::Microsecond, None),
                )?;
                let logp7 = self.detect_type_from_string(
                    stats,
                    &DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                )?;

                let result = self.decide(&[
                    (DataType::Utf8, logp1),
                    (DataType::FixedSizeBinary(16), logp2),
                    (DataType::Date32, logp3),
                    (DataType::Date64, logp4),
                    (DataType::Time32(TimeUnit::Millisecond), logp5),
                    (DataType::Timestamp(TimeUnit::Microsecond, None), logp6),
                    (
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        logp7,
                    ),
                ]);

                if let Some((potent, conf)) = result {
                    let agg_conf = (cast_safety(source, &potent).score() * conf).sqrt();
                    if agg_conf >= 0.65 {
                        dest_type = potent;
                        match dest_type {
                            DataType::Date32 => {
                                datetime_precision = Some(0);
                                numeric_precision = None;
                                numeric_scale = None;
                                character_maximum_length = None;
                            }
                            DataType::Time32(TimeUnit::Millisecond) => datetime_precision = Some(3),
                            DataType::Date64 | DataType::Timestamp(_, _) => {
                                datetime_precision = Some(6);
                                numeric_precision = None;
                                numeric_scale = None;
                                character_maximum_length = None;
                            }
                            _ => {}
                        }
                        confidence = agg_conf;
                    }
                }
            }
            DataType::Int32 => {
                // Scouting potential destination types: Obvious expectation is that date in int32 format will generate Date32
                let logp1 = self.detect_time_from_int(stats, &DataType::Date32);
                let logp2 = self.detect_time_from_int(stats, &DataType::Date64);
                let logp3 = self.detect_time_from_int(stats, &DataType::Time32(TimeUnit::Second));
                let logp4 =
                    self.detect_time_from_int(stats, &DataType::Time32(TimeUnit::Millisecond));
                let logp5 =
                    self.detect_time_from_int(stats, &DataType::Time64(TimeUnit::Microsecond));
                let logp6 =
                    self.detect_time_from_int(stats, &DataType::Time64(TimeUnit::Nanosecond));

                let result = self.decide(&[
                    (DataType::Date32, logp1),
                    (DataType::Date64, logp2),
                    (DataType::Time32(TimeUnit::Second), logp3),
                    (DataType::Time32(TimeUnit::Millisecond), logp4),
                    (DataType::Time64(TimeUnit::Microsecond), logp5),
                    (DataType::Time64(TimeUnit::Nanosecond), logp6),
                ]);

                if let Some((potent, conf)) = result {
                    let agg_conf = (cast_safety(source, &potent).score() * conf).sqrt();
                    if agg_conf >= 0.65 {
                        dest_type = potent;
                        match dest_type {
                            DataType::Date32 | DataType::Time32(TimeUnit::Second) => {
                                datetime_precision = Some(0);
                                numeric_precision = None;
                                numeric_scale = None;
                            }
                            DataType::Time32(TimeUnit::Millisecond) => datetime_precision = Some(3),
                            DataType::Date64 | DataType::Time64(TimeUnit::Microsecond) => {
                                datetime_precision = Some(6);
                                numeric_precision = None;
                                numeric_scale = None;
                            }
                            _ => {}
                        }
                        confidence = agg_conf;
                    }
                }
            }

            DataType::Int64 => {
                let logp1 = self.detect_time_from_int(stats, &DataType::Date32);
                let logp2 = self.detect_time_from_int(stats, &DataType::Date64);
                let logp3 =
                    self.detect_time_from_int(stats, &DataType::Timestamp(TimeUnit::Second, None));
                let logp4 = self
                    .detect_time_from_int(stats, &DataType::Timestamp(TimeUnit::Millisecond, None));
                let logp5 = self
                    .detect_time_from_int(stats, &DataType::Timestamp(TimeUnit::Microsecond, None));
                let logp6 = self
                    .detect_time_from_int(stats, &DataType::Timestamp(TimeUnit::Nanosecond, None));

                let result = self.decide(&[
                    (DataType::Date32, logp1),
                    (DataType::Date64, logp2),
                    (DataType::Timestamp(TimeUnit::Second, None), logp3),
                    (DataType::Timestamp(TimeUnit::Millisecond, None), logp4),
                    (DataType::Timestamp(TimeUnit::Microsecond, None), logp5),
                    (DataType::Timestamp(TimeUnit::Nanosecond, None), logp6),
                ]);

                if let Some((potent, conf)) = result {
                    let agg_conf = (cast_safety(source, &potent).score() * conf).sqrt();
                    if agg_conf >= 0.65 {
                        dest_type = potent;
                        confidence = agg_conf;
                    }
                }
            }
            _ => {}
        }

        let nullable = stats.null_ratio() > 0.0;

        Ok(PromotionResult {
            dest_type,
            confidence,
            nullable,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            datetime_precision,
        })
    }

    /// The function `detect_type_from_string` in calculates a confidence score based on the
    /// likelihood of different data types for a given column.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `detect_type_from_string` returns a `Result` containing a floating-point number
    /// (`f32`) representing the confidence level of the inferred data type based on the provided
    /// statistics and destination data type.
    fn detect_type_from_string(
        &self,
        stats: &ColumnStats,
        dest: &DataType,
    ) -> Result<f32, NisabaError> {
        let mut logp = match dest {
            &DataType::Utf8 => -0.9,
            DataType::FixedSizeBinary(16) => -1.61,
            DataType::Date32 => -2.12,
            DataType::Time32(TimeUnit::Millisecond) => -3.22,
            DataType::Date64 => -2.81,
            DataType::Timestamp(TimeUnit::Microsecond, None) => -1.90,
            DataType::List(_) => -3.51,
            _ => 0.0,
        };

        // Uuid signal
        logp += self.uuid_likelihood(stats)?;

        // Date32 signal
        // Date64 signal
        // Timestamp signal
        // Time32 signal
        logp += self.datetime_likelihood(stats, dest)?;

        // Json signal
        logp += self.json_likelihood(stats)?;

        // Array/List signal
        logp += self.array_likelihood(stats)?;

        // Null penalty
        // Logic: too few values give less signal to trust inference
        let confidence = 1.0 - stats.null_ratio();

        Ok(confidence * logp)
    }

    /// The function `decide` performs log-sum-exp normalization on log priors thus converting the priors to
    /// probablities and selecting the most likely DataType. It required thresholds are not met then there is
    /// no likely DataType
    ///
    /// Arguments:
    ///
    /// * `posteriors`: The `posteriors` parameter in the `detect_type_from_string` function represents a slice
    ///   of resultant tuples of `DataType` and f32 confidence.
    ///
    /// Returns:
    ///
    /// The function `decide` returns a `Option` containing a tuple of DataType representing the destination type and a floating-point number
    /// (`f32`) representing the confidence level in probable terms of the inferred data type.
    fn decide(&self, posteriors: &[(DataType, f32)]) -> Option<(DataType, f32)> {
        let mut sorted = posteriors.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Mormalizaton - log-sum-exp normalization
        // 1. Pick the maximum
        // 2. Perform a maximum shift by doing a difference with all elements
        // 3. Exponent on all elements
        // 4. Probability is the ratio of the result over the sum of all elements
        let m = sorted
            .iter()
            .map(|(_, b)| *b)
            .fold(f32::NEG_INFINITY, f32::max);

        let sum_m: f32 = sorted.iter().map(|(_, b)| ((*b) - m).exp()).sum();

        let sorted: Vec<(DataType, f32)> = sorted
            .into_iter()
            .map(|(a, b)| {
                let conf = (b - m).exp() / sum_m;

                (a, conf)
            })
            .collect();

        let (best_t, best_p) = &sorted[0];
        let (_, second_p) = sorted[1];

        // 2 points more than the p-value conventional threshold
        if (best_p - second_p) < 0.07 {
            None
        } else {
            Some((best_t.clone(), *best_p))
        }
    }

    /// The function `uuid_likelihood` checks the likelihood of sample values in a column to
    /// be parsed as UUID, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    ///
    /// Returns:
    ///
    /// The function `uuid_likelihood` returns a `Result<f32, NisabaError>`. The possible return
    /// values are `Ok` with a floating-point number representing the likelihood of the UUID format
    /// being correct, or an `Err` containing a `NisabaError` if there was an issue during the execution
    /// of the function.
    fn uuid_likelihood(&self, stats: &ColumnStats) -> Result<f32, NisabaError> {
        if stats.character_min_length < Some(32) || stats.character_max_length != Some(45) {
            return Ok(0.0);
        }

        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

        if values.iter().flatten().all(|s| {
            let val = s
                .replace("urn:uuid:", "")
                .replace("-", "")
                .replace("{", "")
                .replace("}", "");

            let val = val.trim();

            Uuid::parse_str(val).is_ok()
        }) {
            // 15% markup as a show of confidence
            return Ok(1.61 * 1.15);
        }

        Ok(0.0)
    }

    /// The function `array_likelihood` checks the likelihood of sample values in a column to
    /// be parsed as list/array, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    ///
    /// Returns:
    ///
    /// The function `array_likelihood` returns a `Result<f32, NisabaError>`. The possible return
    /// values are `Ok` with a floating-point number representing the likelihood of the list/array format
    /// being correct, or an `Err` containing a `NisabaError` if there was an issue during the execution
    /// of the function.
    fn array_likelihood(&self, stats: &ColumnStats) -> Result<f32, NisabaError> {
        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

        // [[1, 2], [3, 4]]
        // [1, 2, 3, 4]
        // [1, "text", true]
        // [{"a": 1}, {"b": 2}]
        if values.iter().flatten().all(|s| {
            let val = s.trim();
            if (val.starts_with("[") && val.ends_with("]"))
                || (val.starts_with("[[") && val.ends_with("]]"))
            {
                return true;
            }
            false
        }) {
            return Ok(3.51);
        }

        Ok(0.0)
    }

    /// The function `json_likelihood` checks the likelihood of sample values in a column to
    /// be parsed as JSON, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    ///
    /// Returns:
    ///
    /// The function `json_likelihood` returns a `Result<f32, NisabaError>`. The possible return
    /// values are `Ok` with a floating-point number representing the likelihood of the JSON format
    /// being correct, or an `Err` containing a `NisabaError` if there was an issue during the execution
    /// of the function.
    fn json_likelihood(&self, stats: &ColumnStats) -> Result<f32, NisabaError> {
        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

        //
        if values.iter().flatten().all(|s| {
            let val = s.trim();
            if val.starts_with("{") && val.ends_with("}") {
                return true;
            }
            false
        }) {
            return Ok(0.9);
        }

        Ok(0.0)
    }

    /// The function `datetime_likelihood` checks the likelihood of sample values in a column to
    /// be parsed as specific date and time formats, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `datetime_likelihood` returns a `Result<f32, NisabaError>`. The possible return
    /// values are `Ok` with a floating-point number representing the likelihood of the datetime format
    /// being correct, or an `Err` containing a `NisabaError` if there was an issue during the execution
    /// of the function.
    fn datetime_likelihood(
        &self,
        stats: &ColumnStats,
        dest: &DataType,
    ) -> Result<f32, NisabaError> {
        let values = stats
            .sample_values
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| ArrowError::CastError("Failed to cast to StringArray".into()))?;

        let values = values.into_iter().flatten().collect::<Vec<&str>>();

        match dest {
            DataType::Date32 => {
                let formats = [
                    "%d%b%Y",
                    "%y/%m/%d",
                    "%Y-%m-%d",
                    "%d-%m-%Y",
                    "%a, %d %b %Y",
                    "%d/%b/%Y",
                    "%Y%m%d",
                ];

                if values.iter().all(|s| {
                    formats
                        .iter()
                        .find_map(|fmt| NaiveDate::parse_from_str(s, fmt).ok())
                        .is_some()
                }) {
                    return Ok(2.12);
                }
            }
            DataType::Date64 => {
                let formats = [
                    "%d%b%Y%p%I%M%S",
                    "%y/%m/%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    // RFC / logs
                    "%a, %d %b %Y %H:%M:%S",
                    "%d/%b/%Y:%H:%M:%S",
                    "%Y%m%d%H%M%S",
                ];

                if values.iter().all(|s| {
                    formats
                        .iter()
                        .find_map(|fmt| NaiveDate::parse_from_str(s, fmt).ok())
                        .is_some()
                }) {
                    return Ok(2.81);
                }
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                let formats = [
                    "%p%I%M%S%.f",
                    "%H:%M:%S%.fZ",
                    "%H:%M:%S%.f%:z",
                    "%H:%M:%S%.f",
                    "%H:%M:%S",
                    "%H:%M:%S%.f GMT",
                    "%H:%M:%S%.f %z",
                    "%H%M%S%.f",
                ];

                if values.iter().all(|s| {
                    formats
                        .iter()
                        .find_map(|fmt| NaiveTime::parse_from_str(s, fmt).ok())
                        .is_some()
                }) {
                    return Ok(3.22);
                }
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                let formats = [
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
                if values.iter().all(|s| {
                    formats
                        .iter()
                        .find_map(|fmt| NaiveDateTime::parse_from_str(s, fmt).ok())
                        .is_some()
                }) {
                    // 10% Markup as a show of confidence
                    return Ok(1.90 * 1.10);
                }
            }
            _ => return Ok(0.0),
        }

        Ok(0.0)
    }

    /// The function `detect_time_from_int` calculates a likelihood score for inferring
    /// time-related data types based on statistical properties of the input data.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `detect_time_from_int` returns a floating-point number (`f32`) which represents the
    /// likelihood of the given data type being a time-related data type based on various statistical
    /// calculations and likelihood calculations performed within the function. The final result is the
    /// product of the calculated likelihood and a confidence factor based on the null ratio of the
    /// data.
    fn detect_time_from_int(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let mut logp = match dest {
            DataType::Date32 => -2.12,
            DataType::Date64 => -2.81,
            DataType::Timestamp(TimeUnit::Second, _) => -1.90,
            DataType::Timestamp(TimeUnit::Millisecond, _) => -1.7,
            DataType::Timestamp(TimeUnit::Microsecond, _) => -1.8,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => -2.5,
            _ => 0.0,
        };

        logp += self.epoch_likelihood(stats, dest);
        logp += self.span_likelihood(stats, dest);
        logp += self.quantile_cv_likelihood(stats, dest);
        logp += self.delta_likelihood(stats, dest);
        logp += self.modulo_entropy_likelihood(stats, dest);
        logp += self.entropy_likelihood(stats, dest);
        logp += self.delta_regularity_likelihood(stats, dest);
        logp += self.delta_scale_likelihood(stats, dest);
        logp += self.monotonic_run_likelihood(stats, dest);

        // Null penalty
        // Logic: too few values give less signal to trust inference
        let confidence = 1.0 - stats.null_ratio();

        confidence * logp
    }

    /// The function `epoch_likelihood` checks the likelihood of sample values in a column to
    /// be parsed as epoch(time) from int64 values, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `epoch_likelihood` returns a `Result<f32, NisabaError>`. The possible return
    /// values are `Ok` with a floating-point number representing the likelihood of the epoch(time) format
    /// being correct, or an `Err` containing a `NisabaError` if there was an issue during the execution
    /// of the function.
    fn epoch_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        // Destructing min and max to find out
        let (min, max) = match (stats.max_val, stats.max_val) {
            (Some(min), Some(max)) if min < max => (min, max),
            _ => return 0.0,
        };

        let in_range = match dest {
            DataType::Time32(TimeUnit::Second) => {
                (0..=86400).contains(&min) && (0..=86400).contains(&max)
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                (0..=86_400_000).contains(&min) && (0..=86_400_000).contains(&max)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                (0..=86_400_000_000).contains(&min) && (0..=86_400_000_000).contains(&max)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                (0..=86_400_000_000_000).contains(&min) && (0..=86_400_000_000_000).contains(&max)
            }
            DataType::Date32 => (-25567..=47482).contains(&min) && (-25567..=47482).contains(&max),
            DataType::Date64 => {
                (-2_208_988_800..=4_102_444_800).contains(&min)
                    && (-2_208_988_800..=4_102_444_800).contains(&max)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                (-2_208_988_800..=4_102_444_800).contains(&min)
                    && (-2_208_988_800..=4_102_444_800).contains(&max)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                (-2_208_988_800_000..=4_102_444_800_000).contains(&min)
                    && (-2_208_988_800_000..=4_102_444_800_000).contains(&max)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                (-2_208_988_800_000_000..=4_102_444_800_000_000).contains(&min)
                    && (-2_208_988_800_000_000..=4_102_444_800_000_000).contains(&max)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                (-2_208_988_800_000_000_000..=4_102_444_800_000_000_000).contains(&min)
                    && (-2_208_988_800_000_000_000..=4_102_444_800_000_000_000).contains(&max)
            }
            _ => false,
        };

        if in_range { 1.2 } else { -2.0 }
    }

    /// The function `span_likelihood` checks the likelihood of sample values in a column to
    /// exist within the desired range differences, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `span_likelihood` returns a floating-point number as the log prior confidence.
    fn span_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        // Destructing min and max to find out
        let (min, max) = match (stats.max_val, stats.max_val) {
            (Some(min), Some(max)) if min < max => (min as f64, max as f64),
            _ => return 0.0,
        };

        // Span Sanity
        // Logic: time spans cluster around human scales
        // counters grow arbitrarily
        let span = (max - min).abs().max(1.0);

        let expected = match dest {
            DataType::Time32(TimeUnit::Second) => 86_400., // 1 day
            DataType::Time32(TimeUnit::Millisecond) => 86_400_000.,
            DataType::Time64(TimeUnit::Microsecond) => 86_400_000_000.,
            DataType::Time64(TimeUnit::Nanosecond) => 86_400_000_000_000.,
            DataType::Date32 => 30.0, // 30 days
            DataType::Date64 | DataType::Timestamp(TimeUnit::Second, _) => 3600., // 1h
            DataType::Timestamp(TimeUnit::Millisecond, _) => 3_600_000.,
            DataType::Timestamp(TimeUnit::Microsecond, _) => 3_600_000_000.,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => 3_600_000_000_000.,

            _ => return 0.0,
        };

        let ratio = span / expected;

        -((ratio.ln()).powi(2) as f32)
    }

    /// The function calculates the coefficient of variation for quantile spacing consistency and
    /// returns a likelihood score based on the data type.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `quantile_cv_likelihood` returns a `f32` value, which is either 0.6, -0.4, or 0.0
    /// based on the conditions specified in the match statement for the `dest` parameter.
    fn quantile_cv_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let q = match stats.quantiles_i32 {
            Some(q) => q,
            None => return 0.0,
        };

        // Quantile spacing consistency {logic: time sampling tends to be regular(cron jobs), semi-regular,
        // or noisy but continous while categorical or bucketed data tends to be uneven or with large plateaus}
        // Penalizes buckets, enum ordinals, or zipf distributed ids
        // Promotes logs, measurements, time series

        let deltas: Vec<f32> = q
            .windows(2)
            .map(|w| (w[1] - w[0]) as f32)
            .filter(|d| *d > 0.0)
            .collect();

        let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let var = deltas.iter().map(|d| (*d - mean).powi(2)).sum::<f32>() / deltas.len() as f32;

        let cv = var.sqrt() / mean.max(1.0);

        match dest {
            DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time64(_)
            | DataType::Time32(_) => {
                if cv < 0.5 {
                    0.6
                } else {
                    -0.4
                }
            }
            _ => 0.0,
        }
    }

    /// The function `delta_likelihood` checks the likelihood of sample values in a column to
    /// exhibit a certain delta behavior when considering integer time or timestamps, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `delta_likelihood` returns a `f32` value, which is either 0.8, -0.4, 1, -0.5, or 0.0
    /// based on the conditions specified in the match statement for the `dest` parameter.
    fn delta_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let ds = match &stats.delta_stats {
            Some(d) => d,
            None => return 0.0,
        };

        // Delta behaviour (day steps) {logic many +1 delta long runs depict counter,
        // most small deltas with breaks depict time, and (or) occassional jumps depict logging}
        // Penalizes counters as they exhibit perfect monotonicity
        // Promotes time patterns as they are monotonic with noise

        match dest {
            DataType::Date32 => {
                if ds.small_delta_ratio > 0.7 {
                    0.8
                } else {
                    -0.4
                }
            }
            DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_) => {
                if ds.small_delta_ratio > 0.85 {
                    1.0
                } else {
                    -0.5
                }
            }
            _ => 0.0,
        }
    }

    /// The function calculates the likelihood of entropy based on the type of data and its temporal
    /// characteristics.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `modulo_entropy_likelihood` returns a `f32` value based on the conditions specified
    /// in the code snippet. If the `dest` parameter matches one of the specified data types (`Date32`,
    /// `Date64`, `Timestamp`, `Time32`, `Time64`), it will return either `1.2` or `-0.6` based on the
    /// value of `h`.
    fn modulo_entropy_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let h = match stats.temporal_mod_entropy {
            Some(h) => h,
            None => return 0.0,
        };

        // Modulo entropy (temporal unit signal)
        // Logic: time has periodicity as counters dont wrap. Lower entropy shows cyclic structure of time
        // Rejects IDs, Hashes, Random ints (high entropy)
        // Accepts date, logs (low entropy)

        match dest {
            DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_) => {
                if h < 0.8 {
                    1.2
                } else {
                    -0.6
                }
            }
            _ => 0.0,
        }
    }

    /// The function `entropy_likelihood` calculates the likelihood of entropy based on the destination
    /// data type.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `entropy_likelihood` returns a `f32` value based on the conditions specified in the
    /// match statement for the `dest` parameter. If the `dest` matches one of the specified data types
    /// (`Date32`, `Date64`, `Timestamp`, `Time32`, `Time64`), it checks the entropy value in the
    /// `stats` parameter. If the entropy is less than
    fn entropy_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        match dest {
            DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_) => {
                if stats.entropy < 0.70 {
                    0.2
                } else {
                    -0.2
                }
            }
            _ => 0.0,
        }
    }

    /// This Rust function calculates the likelihood of delta regularity based on column statistics and
    /// data type.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `delta_regularity_likelihood` returns a floating-point number (f32) representing
    /// the likelihood of regularity based on the provided statistics and destination data type.
    fn delta_regularity_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let ds = match &stats.delta_stats {
            Some(ds) => ds,
            None => return 0.0,
        };

        let mut logp = 0.0;

        // Many small deltas imply time
        if ds.small_delta_ratio > 0.6 {
            logp += 0.6;
        }

        // Strong modal step implies clocked process
        if ds.mode_ratio > 0.5 {
            logp += 0.6;
        }

        // Long uninterrupted run implies logging window
        if ds.long_run_ratio > 0.2 && ds.long_run_ratio < 0.8 {
            logp += 0.4
        }

        match dest {
            DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_) => logp,
            _ => -logp * 0.7,
        }
    }

    /// The function `delta_scale_likelihood` checks the likelihood of sample values in a column to
    /// exhibit a certain delta scale behavior when considering integer time or timestamps, returning a confidence score.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter in the `detect_type_from_string` function represents the a reference
    ///   to actual data and statistics of a column, such as the number of null values, unique values,
    ///   data distribution, etc. It is used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter in the `detect_type_from_string` function represents the desired
    ///   data type that you want to detect based on the input string and column statistics. It is of type
    ///   `DataType`, which is an enum that can have various variants like `Utf8`, `FixedSizeBinary`,
    ///   `Date etc.
    ///
    /// Returns:
    ///
    /// The function `delta_likelihood` returns a `f32` value, which is the negated natural logarithm of
    /// the ratio of median absolute value and expected value time scale based on the conditions
    /// specified in the match statement for the `dest` parameter.
    fn delta_scale_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let ds = match &stats.delta_stats {
            Some(ds) => ds,
            None => return 0.0,
        };

        let median = ds.median_abs_delta.max(1.0);

        let expected = match dest {
            DataType::Date32 => 0.000012,
            DataType::Date64
            | DataType::Timestamp(TimeUnit::Second, _)
            | DataType::Time32(TimeUnit::Second) => 1.0,
            DataType::Timestamp(TimeUnit::Millisecond, _)
            | DataType::Time32(TimeUnit::Millisecond) => 1_000.0,
            DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Time64(TimeUnit::Microsecond) => 1_000_000.0,
            DataType::Timestamp(TimeUnit::Nanosecond, _)
            | DataType::Time64(TimeUnit::Nanosecond) => 1_000_000.0,
            _ => 0.0,
        };

        let ratio = median / expected;

        -ratio.ln().abs()
    }

    fn monotonic_run_likelihood(&self, stats: &ColumnStats, dest: &DataType) -> f32 {
        let lrr = match &stats.longest_run_ratio {
            Some(ms) => ms,
            None => return 0.0,
        };

        match dest {
            DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_) => {
                if *lrr > 0.3 {
                    0.7
                } else {
                    -0.2
                }
            }
            _ => {
                if *lrr > 0.3 {
                    -0.6
                } else {
                    0.0
                }
            }
        }
    }
}

/// The function `cast_utf8_column` casts a column in the mutable RecordBatch to the dest DataType
///
/// Arguments:
///
/// * `batch`: The `batch` parameter in the `cast_utf8_column` function represents a mutable reference
///   to a RecordBatch.
/// * `column_name`: The `column_name` parameter in the `cast_utf8_column` function represents the desired
///   column in the RecordBatch that is the target of a cast.
/// * `dest`: The `dest` parameter in the `cast_utf8_column` function represents the desired DataType the column
///   will be cast to.
///
/// Returns:
///
/// The function `cast_utf8_column` returns a Result of unit value on success and NisabaError when not successful.
pub fn cast_utf8_column(
    batch: &mut RecordBatch,
    column_name: &str,
    dest: &DataType,
) -> Result<(), NisabaError> {
    let schema = batch.schema();

    let index = schema
        .column_with_name(column_name)
        .ok_or(ArrowError::SchemaError(format!(
            "Column {} not found",
            column_name
        )))?;

    let string_array = batch.column(index.0);

    let cast_array = match dest {
        DataType::FixedSizeBinary(16) => {
            // Handling for UUID assuming 16-byte fixed size
            utf8_to_uuid(string_array)?
        }
        DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => {
            // Arrow inbuilt casting
            let cast_options = CastOptions {
                safe: false,
                format_options: Default::default(),
            };

            cast_with_options(string_array, dest, &cast_options)?
        }
        _ => Err(ArrowError::CastError(format!(
            "Unsupported cast from UTF8 to {:?}",
            dest
        )))?,
    };

    // Create new schema with updated field type
    let mut fields: Vec<Field> = schema.fields.iter().map(|f| (**f).clone()).collect();
    fields[index.0] = Field::new(column_name, dest.clone(), fields[index.0].is_nullable());

    let updated_schema = Arc::new(Schema::new(fields));

    // Create new columns array with the cast column
    let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();

    columns[index.0] = cast_array;

    *batch = RecordBatch::try_new(updated_schema, columns)?;

    Ok(())
}

// Casting Utf8 to FixedSizeBinary(16) - UUID

/// The function `utf8_to_uuid` casts a column/array to UUID
///
/// Arguments:
///
/// * `array`: The `array` parameter in the `utf8_to_uuid` function represents a reference to
///   an ArrowArray.
///
/// Returns:
///
/// The function `cast_utf8_column` returns a Result of ArrayRef after a successful cast and NisabaError when not successful.
pub fn utf8_to_uuid(array: &ArrayRef) -> Result<ArrayRef, NisabaError> {
    let string_array =
        array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

    let mut builder = FixedSizeBinaryBuilder::new(16);

    for i in 0..string_array.len() {
        if string_array.is_null(i) {
            builder.append_null();
        } else {
            let input = string_array.value(i);
            let uuid = Uuid::parse_str(
                &input
                    .replace("urn:uuid:", "")
                    .replace("-", "")
                    .replace("{", "")
                    .replace("}", ""),
            )?;

            builder.append_value(uuid.as_bytes())?;
        }
    }

    Ok(Arc::new(builder.finish()))
}
