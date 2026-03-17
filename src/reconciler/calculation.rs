use std::collections::HashSet;

use arrow::datatypes::DataType;
use nalgebra::DVector;

use crate::reconciler::{metrics::FieldStats, reconcile::TableAlignmentScore};

pub fn cosine_similarity(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_a = a.norm();
    let norm_b = b.norm();

    dot / (norm_a * norm_b).max(1e-6)
}

/// Jaccard over matched query-field *index sets*, not similarity scores.
/// Ensures two tables are only connected if they matched the same fields,
/// not merely fields with similar scores.
pub fn jaccard_matched_fields(a: &TableAlignmentScore, b: &TableAlignmentScore) -> f32 {
    let set_a: HashSet<usize> = a.per_field_similarities.iter().map(|(i, _)| *i).collect();
    let set_b: HashSet<usize> = b.per_field_similarities.iter().map(|(i, _)| *i).collect();

    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

pub fn field_weight(stat: &FieldStats) -> f32 {
    if stat.sample_size == 0 {
        return 0.001;
    }

    let n = stat.sample_size as f32;

    let validity = 1.0 - (stat.null_count as f32 / n);
    let cardinality = stat
        .cardinality
        .unwrap_or(stat.distinct_count as f64)
        .max(1.0) as f32;

    let info = stat.entropy.max(0.001) * cardinality.ln().max(0.001);
    let degeneracy = if stat.distinct_count <= 1 {
        0.001
    } else {
        (stat.distinct_count as f32 / n).clamp(0.001, 1.0)
    };

    let richness = compute_richness(stat);

    let w = validity * info * richness * degeneracy;

    if w.is_finite() { w.max(0.001) } else { 0.001 }
}

fn compute_richness(stat: &FieldStats) -> f32 {
    match &stat.source {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => {
            let (max, min) = match (stat.max_val, stat.min_val) {
                (Some(mx), Some(mn)) => (mx, mn),
                _ => (0.002, 0.001),
            };

            let r = (max - min) as f32;

            (r / (r + 1.0)).clamp(0.001, 1.0)
        }

        DataType::Utf8 | DataType::LargeUtf8 => {
            let (max, min) = match (stat.character_max_length, stat.character_min_length) {
                (Some(mx), Some(mn)) => (mx as f32, mn as f32),
                _ => (0.002, 0.001),
            };

            let r = max - min;

            (r / (r + 1.0)).clamp(0.001, 1.0)
        }

        DataType::Timestamp(_, _) | DataType::Date32 | DataType::Time32(_) => {
            stat.histogram_entropy.unwrap_or(1.0).clamp(0.001, 1.0) as f32
        }

        _ => 1.0,
    }
}

pub fn l2_norm(x: DVector<f32>) -> DVector<f32> {
    let norm = x.norm();
    if norm < 1e-8 { x } else { x / norm }
}

pub fn skewness(data: &[f64], sample_size: usize, avg: f64) -> Option<f64> {
    let nf = sample_size as f64;

    let (m2, m3) = data.into_iter().fold((0.0_f64, 0.0_f64), |(s2, s3), x| {
        let d = x - avg;
        (s2 + d * d, s3 + d * d * d)
    });

    let variance = m2 / nf;
    if variance == 0.0 {
        return Some(0.0);
    }

    let std_dev = variance.sqrt();

    let skew = (nf / ((nf - 1.0) * (nf - 2.0))) * (m3 / std_dev.powi(3));

    Some(skew)
}

pub fn kurtosis(data: &[f64], sample_size: usize, avg: f64) -> Option<f64> {
    let nf = sample_size as f64;

    let (m2, m4) = data.into_iter().fold((0.0_f64, 0.0_f64), |(s2, s4), x| {
        let d = x - avg;
        let d2 = d * d;

        (s2 + d2, s4 + d2 * d2)
    });

    let variance = m2 / nf;
    if variance == 0.0 {
        return Some(0.0);
    }

    // G2 (unbiased excess kurtosis)
    let kurt = (nf * (nf + 1.0) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0)))
        * (m4 / (variance * variance))
        - 3.0 * (nf - 1.0).powi(2) / ((nf - 2.0) * (nf - 3.0));

    Some(kurt)
}

pub fn histogram_entropy(
    data: &[f64],
    sample_size: usize,
    bins: Option<usize>,
    max: Option<f64>,
    min: Option<f64>,
) -> Option<f64> {
    let (Some(max), Some(min)) = (max, min) else {
        return None;
    };

    if min.partial_cmp(&max).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Equal {
        return Some(0.0);
    }

    let bins = bins.unwrap_or(7);

    let width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];

    for x in data {
        let idx = ((x - min) / width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let entropy = counts
        .iter()
        .filter(|v| **v > 0)
        .map(|v| {
            let p = *v as f64 / sample_size as f64;
            -p * p.ln()
        })
        .sum();

    Some(entropy)
}

pub fn autocorrelation(data: &[f64], sample_size: usize, avg: f64, lag: usize) -> Option<f64> {
    if sample_size < lag {
        return None;
    }

    let variance = data.into_iter().map(|x| (x - avg).powi(2)).sum::<f64>();

    if variance == 0.0 {
        return None;
    }

    let vals = data.iter().collect::<Vec<&f64>>();

    let cov: f64 = vals[..sample_size - lag]
        .iter()
        .zip(&vals[lag..])
        .map(|(a, b)| (**a - avg) * (**b - avg))
        .sum();
    Some(cov / variance)
}
