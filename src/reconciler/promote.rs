use arrow::{
    array::{Array, ArrayRef, FixedSizeBinaryBuilder, RecordBatch, StringArray},
    compute::{CastOptions, cast_with_options, kernels::cast},
    datatypes::{DataType, Field, Schema, TimeUnit},
    error::ArrowError,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{BinaryHeap, HashSet},
    sync::Arc,
};
use uuid::Uuid;

use crate::{error::NisabaError, reconciler::metrics::FieldStats};

#[derive(Debug, Clone)]
/// The `PromotionResult` represents the result of a data type promotion operation,
/// including destination type, confidence level, and optional metadata.
pub struct PromotionResult {
    /// data type to which a value is being promoted or converted.
    pub dest_type: DataType,
    /// the level of certainty or belief in the promotion result
    pub confidence: f32,
    /// Flag indicating whether the corresponding data type can accept NULL values or not
    pub nullable: bool,
    /// maximum length of characters for a data type
    pub character_maximum_length: Option<i32>,
    /// total number of digits that can be stored, including both the digits before and after the decimal point.
    pub numeric_precision: Option<i32>,
    /// number of digits to the right of the decimal point in a number.
    pub numeric_scale: Option<i32>,
    /// precision of a datetime data type
    pub datetime_precision: Option<i32>,
}

pub struct TypeLatticeResolver;

impl TypeLatticeResolver {
    pub fn new() -> Self {
        Self
    }

    pub fn promote(&self, stats: &FieldStats) -> Result<PromotionResult, NisabaError> {
        // Int32 is always widened to In64 so both share a singel rule-set

        let source = match &stats.source {
            DataType::Int32 => &DataType::Int64,
            other => other,
        };

        let nullable = stats.null_ratio() > 0.0;
        let (dest_type, confidence) = self.scout(source, stats)?;
        let (datetime_precision, numeric_precision, numeric_scale, character_maximum_length) =
            Self::meta_for(&dest_type, stats);

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

    fn scout(&self, source: &DataType, stats: &FieldStats) -> Result<(DataType, f32), NisabaError> {
        let mut tallies = self.build_tallies(source, stats)?;

        // Best net score first
        tallies.sort_by(|a, b| {
            b.net()
                .partial_cmp(&a.net())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for tally in &tallies {
            if !tally.majority_for() {
                continue;
            }

            // Scale net score by how safe this cast actually is
            let confidence = cast_safety(source, &tally.dest).score() * tally.net();

            if confidence >= 0.65 {
                return Ok((tally.dest.clone(), confidence));
            }
        }

        // No candidate cleared the bar; stay as-sis with full confidence
        Ok((source.clone(), 1.0))
    }

    fn build_tallies(
        &self,
        source: &DataType,
        stats: &FieldStats,
    ) -> Result<Vec<Tally>, NisabaError> {
        match source {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let candidates: &[DataType] = &[
                    DataType::FixedSizeBinary(16),
                    DataType::Date32,
                    DataType::Time32(TimeUnit::Millisecond),
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    DataType::Int64,
                    DataType::Float64,
                    DataType::Boolean,
                ];

                candidates
                    .par_iter()
                    .map(|dest| {
                        let mut tally = Tally::new(dest.clone());
                        tally.cast(self.uuid_rule(stats, dest)?);
                        tally.cast(self.datetime_rule(stats, dest)?);
                        tally.cast(self.json_rule(stats, dest)?);
                        tally.cast(self.array_rule(stats, dest)?);
                        tally.cast(self.float64_rule(stats, dest)?);
                        tally.cast(self.boolean_rule(stats, dest));

                        Ok(tally)
                    })
                    .collect()
            }

            DataType::Int64 => {
                let candidates: &[DataType] = &[
                    DataType::Timestamp(TimeUnit::Second, None),
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                    DataType::Boolean,
                ];

                candidates
                    .par_iter()
                    .map(|dest| {
                        let mut tally = Tally::new(dest.clone());

                        // Proper integers that are not boolean-integer representatives/maps
                        if let Some(ali) = stats.all_integers
                            && ali
                        {
                            tally.cast(self.span_rule(stats, dest));

                            tally.cast(self.quantile_regularity_rule(stats, dest));

                            tally.cast(self.value_entropy_rule(stats, dest));

                            tally.cast(self.int64_rule(stats, dest)?);

                            tally.cast(self.epoch_validity_rule(stats, dest)?);
                        }

                        tally.cast(self.numeric_down_rule(stats, dest)?);

                        Ok(tally)
                    })
                    .collect()
            }
            DataType::Timestamp(tunit, _) => {
                let mut tally = Tally::new(DataType::Date32);
                tally.cast(self.date32_rule(stats, tunit)?);

                Ok(vec![tally])
            }

            DataType::Int8
            | DataType::Int16
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64 => {
                let candidates: &[DataType] = &[DataType::Boolean, DataType::Int64];

                candidates
                    .par_iter()
                    .map(|dest| {
                        let mut tally = Tally::new(dest.clone());

                        tally.cast(self.numeric_down_rule(stats, dest)?);

                        Ok(tally)
                    })
                    .collect()
            }

            _ => Ok(vec![]),
        }
    }

    // String Source Rules

    /// The function `uuid_rule` checks the sample string values in a column to
    /// be parsed as UUID, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to compute a Vote based on the provided profile.
    /// * `dest`: The target type variant that a cast/promotion is desired
    ///
    /// Returns:
    ///
    /// The function `uuid_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of the UUID format being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn uuid_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        // QUick length guard before touching the array
        match (stats.character_max_length, stats.character_max_length) {
            (Some(mn), Some(mx)) if mn >= 32 && mx <= 45 => {}
            _ => return Ok(Vote::Abstain),
        }

        let all_uuid = match stats.all_uuid {
            Some(alu) => alu,
            None => return Ok(Vote::Abstain),
        };

        if !all_uuid {
            return Ok(Vote::Abstain);
        }

        Ok(match dest {
            DataType::FixedSizeBinary(16) => Vote::For,
            // A UUID column is technically valid Utf8, but we prefer binary;
            // do not penalise Utf8 so it can still win if safety blocks binary
            DataType::Utf8 => Vote::Abstain,
            // UUID evidence rules out every numeric and temporal type
            _ => Vote::Against,
        })
    }

    /// The function `array_rule` checks the sample string values in a column to
    /// be parsed as list/array, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to compute a Vote of a certain data type based on the provided profile.
    /// * `dest`: The target type variant that a cast/promotion is desired
    ///
    /// Returns:
    ///
    /// The function `uuid_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of the list/array format being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn array_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        let all_array = match stats.all_array {
            Some(ala) => ala,
            None => return Ok(Vote::Abstain),
        };
        if !all_array {
            return Ok(Vote::Abstain);
        }

        Ok(match dest {
            DataType::List(_) => Vote::For,
            DataType::Int64
            | DataType::Float64
            | DataType::Boolean
            | DataType::Date32
            | DataType::FixedSizeBinary(16) => Vote::Against,
            _ => Vote::Abstain,
        })
    }

    /// The function `json_rule` checks the sample string values in a column to
    /// be parsed as JSON, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to compute a Vote of a certain data type based on the provided profile.
    /// * `dest`: The target type variant that a cast/promotion is desired
    ///
    /// Returns:
    ///
    /// The function `json_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of the JSON format being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn json_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        let all_json = match stats.all_json {
            Some(alj) => alj,
            None => return Ok(Vote::Abstain),
        };

        if !all_json {
            return Ok(Vote::Abstain);
        }

        Ok(match dest {
            DataType::Utf8 => Vote::For,
            DataType::Int64
            | DataType::Float64
            | DataType::Boolean
            | DataType::Date32
            | DataType::FixedSizeBinary(16) => Vote::Against,
            _ => Vote::Abstain,
        })
    }

    /// The function `datetime_rule` checks the sample string values in a column to
    /// be parsed as specific date and time formats, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to be detectected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    /// The function `datetime_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of the datetime format being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn datetime_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        let matched = match dest {
            DataType::Date32 => match stats.all_date32 {
                Some((_, flag)) => flag,
                None => return Ok(Vote::Abstain),
            },

            DataType::Time32(TimeUnit::Millisecond) => match stats.all_time {
                Some((_, flag)) => flag,
                None => return Ok(Vote::Abstain),
            },

            DataType::Timestamp(_, _) => match stats.all_timestamp {
                Some((_, flag)) => flag,
                None => return Ok(Vote::Abstain),
            },
            _ => return Ok(Vote::Abstain),
        };

        Ok(if matched { Vote::For } else { Vote::Against })
    }

    /// The function `float64_rule` checks the sample string values in a column to
    /// be parsed as Float64/Int64, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to be detectected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    /// The function `float64_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of Int64/Float64 being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn float64_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        let all_integers = match stats.all_integers {
            Some(ali) => ali,
            None => return Ok(Vote::Abstain),
        };

        Ok(match dest {
            DataType::Int64 => {
                if all_integers {
                    Vote::For
                } else {
                    // Decimals rule out Int64
                    Vote::Against
                }
            }

            DataType::Float64 => {
                if !all_integers {
                    // Boolean/Int64 rule out Float64
                    Vote::For
                } else {
                    // All finite floats support Float64
                    Vote::Against
                }
            }

            DataType::Utf8 => Vote::Against,
            _ => Vote::Abstain,
        })
    }

    fn boolean_rule(&self, stats: &FieldStats, dest: &DataType) -> Vote {
        let all_bool = match stats.all_utf8_bools {
            Some(alb) => alb,
            None => return Vote::Abstain,
        };

        match dest {
            DataType::Boolean => {
                if all_bool {
                    Vote::For
                } else {
                    Vote::Against
                }
            }

            _ => Vote::Abstain,
        }
    }

    // Integer Source Rules

    /// The function `numeric_down_rule` checks the sample numeric values in a column to
    /// be parsed as Boolean/Int64/Float64, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to be detectected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    /// The function `datetime_rule` returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of Boolean/Int64/Float64 being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn numeric_down_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        let (all_bool, all_integers) = match (stats.all_int_bools, stats.all_integers) {
            (Some(alb), Some(ali)) => (alb, ali),
            _ => return Ok(Vote::Abstain),
        };

        Ok(match dest {
            DataType::Boolean => {
                if all_bool {
                    Vote::For
                } else {
                    Vote::Against
                }
            }
            DataType::Int64 => {
                if all_integers {
                    Vote::For
                } else {
                    // Decimals rule out Int64
                    Vote::Against
                }
            }

            DataType::Float64 => {
                if !all_bool || !all_integers {
                    // Boolean/Int64 rule out Float64
                    Vote::For
                } else {
                    // All finite floats support Float64
                    Vote::Against
                }
            }

            DataType::Utf8 => Vote::Against,
            _ => Vote::Abstain,
        })
    }

    /// The function `span_rule` checks the sample integer values in a column if they
    /// exist within the desired range differences, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to calculate the likelihood of a certain data type based on the provided
    ///   profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to be detected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    /// The function returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of epoch(time) being correct, or an `Err`
    /// containing a `NisabaError` if there was an issue during the execution
    fn span_rule(&self, stats: &FieldStats, dest: &DataType) -> Vote {
        // Destructing min and max to find out
        let (min, max) = match (stats.min_val, stats.max_val) {
            (Some(min), Some(max)) if min < max => (min, max),
            _ => return Vote::Abstain,
        };

        // Span Sanity
        // Logic: time spans cluster around human scales
        // counters grow arbitrarily
        // Span should be plausible at the scale of the target type.
        // Checked with a log-ratio so the test is scale-invariant across units.
        let span = (max - min).abs().max(1.0);

        let expected = match dest {
            DataType::Timestamp(TimeUnit::Second, _) => 1., // 1 secs
            DataType::Timestamp(TimeUnit::Millisecond, _) => 1_000_000.,
            DataType::Timestamp(TimeUnit::Microsecond, _) => 1_000_000_000.,
            DataType::Timestamp(TimeUnit::Nanosecond, _) => 1_000_000_000_000.,

            _ => return Vote::Abstain,
        };

        // Define broad human-plausible bounds
        // 1 second to 200 years
        const MIN_SECS: f64 = 1.0;
        const MAX_SECS: f64 = 200.0 * 365.25 * 86400.0;

        let ratio_min = ((span / expected) / MIN_SECS).ln();
        let ratio_max = ((span / expected) / MAX_SECS).ln();

        let ln10 = std::f64::consts::LN_10;

        // Too small or too large beyond tolerance → reject
        if ratio_min < -3.0 * ln10 || ratio_max > 3.0 * ln10 {
            Vote::Against
        }
        // Clearly within human range → positive signal
        else if span >= MIN_SECS * expected && span <= MAX_SECS * expected {
            Vote::For
        } else {
            Vote::Abstain
        }
    }

    /// The function calculates the coefficient of variation for quantile spacing consistency and
    /// returns a Vote based on the data type for an integer column.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to compute a Vote of a certain data type based on the provided profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to detected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    /// The function returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of epoch(time) being correct in terms of regularity,
    ///  or an `Err` containing a `NisabaError` if there was an issue during the execution
    fn quantile_regularity_rule(&self, stats: &FieldStats, dest: &DataType) -> Vote {
        let q = match &stats.quantiles_f64 {
            Some(q) => q,
            None => return Vote::Abstain,
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

        if deltas.is_empty() {
            return Vote::Abstain;
        }

        let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let var = deltas.iter().map(|d| (*d - mean).powi(2)).sum::<f32>() / deltas.len() as f32;

        let cv = var.sqrt() / mean.max(1.0);

        match dest {
            DataType::Timestamp(_, _) => {
                // HARD rule threshold based on timestamps observed
                if cv > 0.12 && cv < 0.9 {
                    Vote::For
                } else {
                    Vote::Against
                }
            }
            _ => Vote::Abstain,
        }
    }

    /// The function computes a Vote on entropy based on the destination data type.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to computes a Vote for a certain data type based on the provided profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to detected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    ///
    /// The function returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of (`Date32`,
    /// `Date64`, `Timestamp`) being correct in terms of entropy,
    ///  or an `Err` containing a `NisabaError` if there was an issue during the execution
    fn value_entropy_rule(&self, stats: &FieldStats, dest: &DataType) -> Vote {
        // Global value entropy is low for temporal sequences (locally structured)
        // and high for random IDs or hashes.
        match dest {
            DataType::Timestamp(_, _) => {
                if stats.entropy > 0.90 {
                    Vote::For
                } else {
                    Vote::Against
                }
            }
            _ => Vote::Abstain,
        }
    }

    fn int64_rule(&self, stats: &FieldStats, dest: &DataType) -> Result<Vote, NisabaError> {
        if stats.sample_size <= 1 {
            return Ok(Vote::Abstain);
        }

        let DataType::Timestamp(unit, _) = dest else {
            return Ok(Vote::Abstain);
        };

        let Some(band_counts) = stats.timestamp_bands.as_ref().and_then(|bc| bc.get(unit)) else {
            return Ok(Vote::Abstain);
        };

        const OFFSET: usize = 5;

        // Step 3 & 4: Check for contiguity
        let u_bands: HashSet<u8> = (5u8..11)
            .filter(|v| band_counts[(*v) as usize - OFFSET] > 0)
            .collect();

        let mut u_bands: Vec<u8> = u_bands.into_iter().collect();
        u_bands.sort_unstable();

        // The first band has 1 day: we can ignore it for contiguity check
        u_bands.pop_if(|x| *x == 5);

        let diffs = u_bands.windows(2).map(|w| w[1] - w[0]).all(|v| v == 1);

        if !diffs && u_bands.len() > 1 {
            return Ok(Vote::Against);
        }

        // Step 5: Dominant band check using array
        let mut max_heap: BinaryHeap<usize> = band_counts.iter().map(|v| *v).collect();
        let max_count = [max_heap.pop(), max_heap.pop()]
            .iter()
            .flatten()
            .sum::<usize>();

        let fraction_dominant = max_count as f64 / band_counts.iter().sum::<usize>() as f64;

        if fraction_dominant >= 0.5 {
            Ok(Vote::For)
        } else {
            Ok(Vote::Abstain)
        }
    }

    // Timestamp Source Rule

    /// The function checks the timestamp values in a column to
    /// exhibit a day granularity behavior by casting timestamp values down, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to computes a Vote for a certain data type based on the provided profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to detected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    ///
    /// The function returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of Date32 type being correct in terms of delta_scale,
    ///  or an `Err` containing a `NisabaError` if there was an issue during the execution
    ///
    fn date32_rule(&self, stats: &FieldStats, tunit: &TimeUnit) -> Result<Vote, NisabaError> {
        let Some(aligned) = stats
            .all_timestamp_date32
            .as_ref()
            .and_then(|bc| bc.get(tunit))
        else {
            return Ok(Vote::Abstain);
        };

        Ok(if *aligned { Vote::For } else { Vote::Against })
    }

    /// The function checks the sample integer values in a column with the assumption of second unit to
    /// confirm if they are distinguishable as Millisecond, Microsecond and Nanosecond unit, returning a Vote.
    ///
    /// Arguments:
    ///
    /// * `stats`: The `stats` parameter is a reference to actual data and statistics of a column
    ///   used to computes a Vote for a certain data type based on the provided profile.
    /// * `dest`: The `dest` parameter represents the desired data type that needs to detected
    ///   based on the input string and column statistics.
    ///
    /// Returns:
    ///
    ///
    /// The function returns a `Result<Vote, NisabaError>`. The possible return
    /// values are `Ok` with a Vote for/against of a sub-second unit timestamp
    /// type being correct in terms of delta_scale,
    ///  or an `Err` containing a `NisabaError` if there was an issue during the execution
    ///
    fn epoch_validity_rule(
        &self,
        stats: &FieldStats,
        dest: &DataType,
    ) -> Result<Vote, NisabaError> {
        match (stats.min_val, stats.max_val) {
            (Some(min), Some(max)) if min < max => {}
            _ => return Ok(Vote::Abstain),
        }

        let (p10, p90) = match (stats.p10, stats.p90) {
            (Some(p1), Some(p9)) => (p1, p9),
            _ => return Ok(Vote::Abstain),
        };

        // Realistic date range expressed in seconds (1900–2100)
        const SECS_MIN: i64 = -2_208_988_800;
        const SECS_MAX: i64 = 4_102_444_800;
        const MIN_SPAN: i64 = 60; // must span at least 1 minute

        match dest {
            DataType::Timestamp(unit, _) => {
                let scale: i64 = match unit {
                    TimeUnit::Second => 1,
                    TimeUnit::Millisecond => 1_000,
                    TimeUnit::Microsecond => 1_000_000,
                    TimeUnit::Nanosecond => 1_000_000_000,
                };

                // Magnitude check: median must sit inside the scaled realism window
                let scaled_min = SECS_MIN.saturating_mul(scale);
                let scaled_max = SECS_MAX.saturating_mul(scale);

                if !(scaled_min..scaled_max).contains(&(stats.median as i64)) {
                    return Ok(Vote::Abstain);
                }

                // Span/realism checks

                let p10 = p10.div_euclid(scale as f64) as i64;
                let p90 = p90.div_euclid(scale as f64) as i64;

                if p10 < SECS_MIN || p90 > SECS_MAX {
                    return Ok(Vote::Against);
                }

                // Check if p90-p10 span is permissible i.e more than a minute(60secs)
                if (p90 - p10) < MIN_SPAN {
                    return Ok(Vote::Abstain);
                }

                // Remainder-variance test: values with near-zero sub-second variance
                // are likely seconds naively widened to a finer unit (fake precision)
                // ---- Intrinsic resolution test ----
                //
                // Detect declared precision much finer than actual data resolution.
                //
                // IMPORTANT:
                // Coarse resolution (e.g., minute-sampled ms data)
                // does NOT imply fake precision.
                //
                // We only vote Against if intrinsic resolution
                // is strictly coarser than the declared unit.
                if let Some(delta_gcd) = stats.delta_gcd {
                    if delta_gcd > scale {
                        // Data does not vary at declared resolution.
                        // e.g. declared nanoseconds but intrinsic resolution is milliseconds.
                        return Ok(Vote::Against);
                    }
                }

                Ok(Vote::For)
            }
            _ => Ok(Vote::Abstain),
        }
    }

    fn meta_for(
        dest: &DataType,
        stats: &FieldStats,
    ) -> (Option<i32>, Option<i32>, Option<i32>, Option<i32>) {
        match dest {
            DataType::Date32 | DataType::Boolean | DataType::Time32(_) => (None, None, None, None),
            DataType::Timestamp(_, _) => (Some(6), None, None, None),
            _ if dest.is_numeric() => (None, None, None, None),
            _ => (
                stats.datetime_precision,
                stats.numeric_precision,
                stats.numeric_scale,
                stats.character_max_length,
            ),
        }
    }
}

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
/// * `from`: The `from` parameter represents the data type that you want to cast from.
/// * `to`: The `to` parameter represents the data type that you want to cast to.
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
        (DataType::Float64, DataType::Int64) => CastSafety::FallibleLossy,

        (
            DataType::Decimal32(_, _),
            DataType::Decimal64(_, _) | DataType::Decimal128(_, _) | DataType::Decimal256(_, _),
        ) => CastSafety::Safe,

        //Timestamp conversions
        (DataType::Int64, DataType::Timestamp { .. }) => CastSafety::FallibleLossy,
        (DataType::Timestamp { .. }, DataType::Int64) => CastSafety::Safe,
        (DataType::Timestamp(u1, ..), DataType::Timestamp(u2, ..)) if u1 == u2 => CastSafety::Safe,
        (DataType::Timestamp { .. }, DataType::Timestamp { .. }) => CastSafety::FallibleLossy,
        (DataType::Timestamp { .. }, DataType::Date32) => CastSafety::FallibleLossy,

        // Date conversions
        (DataType::Int32, DataType::Date32) => CastSafety::FallibleLossy,
        (DataType::Date32, DataType::Int32) => CastSafety::Safe,

        // String/JSON conversions
        (DataType::Utf8, DataType::List { .. }) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Struct { .. }) => CastSafety::FallibleLossy,
        (DataType::Binary, DataType::Utf8) => CastSafety::Unsafe,
        (DataType::Utf8, DataType::Binary) => CastSafety::Safe,

        (DataType::FixedSizeBinary(16), DataType::Utf8) => CastSafety::Safe,
        (DataType::Utf8, DataType::FixedSizeBinary(16)) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Timestamp(_, _)) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Date32) => CastSafety::FallibleLossy,
        (DataType::Utf8, DataType::Float64 | DataType::Int64) => CastSafety::FallibleLossy,
        (DataType::Binary, DataType::FixedSizeBinary(16)) => CastSafety::FallibleLossy,

        // Boolean conversions
        (DataType::Utf8, DataType::Boolean) => CastSafety::FallibleLossy,
        (DataType::Float16 | DataType::Float32 | DataType::Float64, DataType::Boolean) => {
            CastSafety::FallibleLossy
        }
        (
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
            DataType::Boolean,
        ) => CastSafety::FallibleLossy,

        _ => CastSafety::Unsafe,
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
pub fn cast_column(
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

    let array = batch.column(index.0);

    let cast_array = match dest {
        DataType::FixedSizeBinary(16) => {
            // Handling for UUID assuming 16-byte fixed size
            utf8_to_uuid(array)?
        }
        DataType::Date32 | DataType::Timestamp(_, _) | DataType::Int64 => {
            // Arrow inbuilt casting
            let cast_options = CastOptions {
                safe: false,
                format_options: Default::default(),
            };

            cast_with_options(array, dest, &cast_options)?
        }

        DataType::Boolean => {
            // Handling boolean parsing from utf8
            match array.data_type() {
                DataType::Utf8 => {
                    // Replace 1.0 and 0.0 to Arrow accepted values (1/0)
                    replace_float_bool(array)?;

                    // Arrow inbuilt casting
                    let cast_options = CastOptions {
                        safe: false,
                        format_options: Default::default(),
                    };
                    cast_with_options(array, dest, &cast_options)?
                }
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64 => {
                    // Handle boolean parsing/casting from numeric
                    numeric_to_bool(array)?
                }

                _ => Err(ArrowError::CastError(format!(
                    "Unsupported cast from {} to {:?}",
                    array.data_type(),
                    dest,
                )))?,
            }
        }

        _ => Err(ArrowError::CastError(format!(
            "Unsupported cast from UTF8 to {:?}",
            dest
        )))?,
    };

    // Create new schema with updated field type
    let mut fields: Vec<Field> = schema.fields.iter().map(|f| (**f).clone()).collect();
    fields[index.0] = Field::new(column_name, dest.clone(), fields[index.0].is_nullable());

    // Create new columns array with the cast column
    let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();

    columns[index.0] = cast_array;

    *batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;

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

pub fn replace_float_bool(array: &ArrayRef) -> Result<ArrayRef, NisabaError> {
    let string_array =
        array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(ArrowError::CastError(
                "Failed to cast to StringArray".into(),
            ))?;

    let arr = StringArray::from(
        string_array
            .iter()
            .map(|v| v.map(|v| v.replace("1.0", "1").replace("0.0", "0")))
            .collect::<Vec<Option<String>>>(),
    );

    Ok(Arc::new(arr))
}

fn numeric_to_bool(array: &ArrayRef) -> Result<ArrayRef, NisabaError> {
    let arr = cast(array, &DataType::Boolean)?;

    Ok(arr)
}

/// Rule voting
/// A rule inspects `ColumnStats` and casts of the three votes for a specific candidate type
///
/// `Abstain` - the rule has no opinion if the observation is evidence enough
/// `Against` - the observation is negative evidence the column is that type
/// `For` - the observation is positive evidence the column is that type
///
/// The critical attribute : a rule must only vote on the type(s) within its remit.
/// For instance, A UUID rule has nothing to say about Float64 and must abstain rather than
/// silently polluting the Float64 tally with UUID evidence
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Vote {
    Abstain,
    Against,
    For,
}

#[derive(Debug)]
/// Accumulated voted for a single candidate type
struct Tally {
    dest: DataType,
    for_: u32,
    against: u32,
}

impl Tally {
    fn new(dest: DataType) -> Self {
        Self {
            dest,
            for_: 0,
            against: 0,
        }
    }

    fn cast(&mut self, vote: Vote) {
        match vote {
            Vote::Abstain => {}
            Vote::Against => self.against += 1,
            Vote::For => self.for_ += 1,
        }
    }

    /// Net score in [-1, 1]: (for - against) / (for + against)
    /// Returns 0.0 when no opinionated rule has fired
    fn net(&self) -> f32 {
        let total = self.for_ + self.against;
        if total == 0 {
            return 0.0;
        }

        (self.for_ as f32 - self.against as f32) / total as f32
    }

    // True when at least one rule voted for and a strict majority did so
    fn majority_for(&self) -> bool {
        self.for_ > 0 && self.for_ > self.against
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    use arrow::datatypes::DataType;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use crate::reconciler::promote::TypeLatticeResolver;

    // Happy Path tests
    // Int64 -> Timestamp inference is problematic and will need more refinement

    #[test]
    fn test_type_promotion() {
        let file = File::open("./assets/promote_data.parquet").unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .with_limit(1000);

        let batch = builder
            .build()
            .unwrap()
            .next()
            .expect("No record batch in Parquet file")
            .unwrap();

        let schema = batch.schema();

        let resolver = TypeLatticeResolver::new();

        for (field, column) in schema.fields().iter().zip(batch.columns()) {
            let field_name = field.name();

            let is_supported = matches!(
                column.data_type(),
                DataType::LargeUtf8
                    | DataType::Utf8
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Timestamp(_, _)
                    | DataType::Float64
            );

            if !is_supported {
                continue;
            }

            let stats = FieldStats::calculate(&column, "promote_data", field_name).unwrap();
            let resolved_result = resolver.promote(&stats).unwrap();

            match field_name.as_str() {
                "uuid_str" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::FixedSizeBinary(16)
                    ));
                }

                "bool_utf8" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Boolean));
                }

                "int64_str" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Int64));
                }

                "int_64" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Int64))
                }

                "float64_str" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Float64))
                }

                "time32_str" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Time32(_)))
                }

                "date32_str" => {
                    assert!(matches!(resolved_result.dest_type, DataType::Date32));
                }

                "timestamp_str" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::Timestamp(_, _)
                    ))
                }

                "timestamp_sec" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::Timestamp(TimeUnit::Second, _)
                    ))
                }

                "timestamp_ms" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::Timestamp(TimeUnit::Millisecond, _)
                    ))
                }

                "timestamp_us" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::Timestamp(TimeUnit::Microsecond, _)
                    ))
                }

                "timestamp_ns" => {
                    assert!(matches!(
                        resolved_result.dest_type,
                        DataType::Timestamp(TimeUnit::Nanosecond, _)
                    ))
                }
                "arr_str" => {
                    assert!(matches!(resolved_result.dest_type, DataType::List(_)));
                }

                _ => {}
            }
        }
    }
}
