use arrow::datatypes::{DataType, Field, Schema};
use std::{
    fmt::{self, Write},
    hash::Hash,
    sync::Arc,
};
use uuid::Uuid;

use crate::reconciler::{
    metrics::{ColSignature, FieldStats},
    report::FieldMatch,
    retriever::Storable,
};

#[derive(Debug, Clone)]
pub struct FieldProfile {
    pub field_def: FieldDef,
    pub field_stats: Option<FieldStats>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Global unique Id for the field
    pub id: Uuid,
    /// Id of the table in which the field is member
    pub table_id: Uuid,
    /// Name of the field
    pub name: String,
    /// Arrow DataType of the field
    pub canonical_type: DataType,
    /// Optional floating-point confidence in DataType of the field
    pub type_confidence: Option<f32>,
    // Optional String property for a default value for of a String field
    pub column_default: Option<String>,
    // Boolean flag if the field allows null
    pub is_nullable: bool,
    // Optional integer max length of characters of a String field
    pub char_max_length: Option<i32>,
    // Optional integer precision of a floating-point field
    pub numeric_precision: Option<i32>,
    // Optional integer scale of a floating-point field
    pub numeric_scale: Option<i32>,
    // Optional integer precision of a datetime field
    pub datetime_precision: Option<i32>,
}

impl std::fmt::Display for FieldDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FieldDef: \n- id: {} \n- table id: {} \n- field name: {} \n- data type: {} \n- is nullable: {}",
            self.id, self.table_id, self.name, self.canonical_type, self.is_nullable
        )?;

        if let Some(tc) = self.type_confidence {
            write!(f, "\n- type confidence: {}", tc)?;
        }

        if let Some(tc) = self.type_confidence {
            write!(f, "\n- type confidence: {}", tc)?;
        }

        if let Some(default) = &self.column_default {
            write!(f, "\n- column default: {}", default)?;
        }

        if let Some(max) = self.char_max_length {
            write!(f, "\n- char max length: {}", max)?;
        }

        if let Some(np) = self.numeric_precision {
            write!(f, "\n- numeric precision: {}", np)?;
        }

        if let Some(ns) = self.numeric_scale {
            write!(f, "\n- numeric scale: {}", ns)?;
        }

        if let Some(dp) = self.datetime_precision {
            write!(f, "\n- datetime precision: {}", dp)?;
        }

        Ok(())
    }
}

impl Hash for FieldDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Eq for FieldDef {}

impl FieldProfile {
    /// The `write_field_def_paragraph` function generates a detailed description of a field's
    /// properties such as type, cardinality, nullability, default value, size constraints, numeric
    /// precision, and datetime precision.
    ///
    /// Arguments:
    ///
    /// * `out`: The `out` parameter is a mutable reference to a `String` where the generated paragraph will be written into.
    pub fn write_field_def_paragraph(&self, out: &mut String) {
        let base = 256;
        let extra = self
            .field_def
            .column_default
            .as_ref()
            .map_or(0, |s| s.len() + 70)
            + self.field_stats.as_ref().map_or(0, |_|70) // avg bytes
            + self.field_def.char_max_length.map_or(0, |_| 70)
            + self.field_def.numeric_precision.map_or(0, |_| 60)
            + self.field_def.numeric_scale.map_or(0, |_| 60)
            + self.field_def.datetime_precision.map_or(0, |_| 70);

        out.reserve(base + extra);

        // Identity + context
        // Thoughts are that identity is irrelevant considering names and tables could have little to no significance e.g table1, table2

        // Canonical type
        self.write_type_sentence(out);

        // Numeric<Mean, Median, Max>
        self.write_numeric(out);

        // Cardinality
        self.write_cardinality(out);

        // Monotonicity
        if let Some(st) = &self.field_stats
            && st.is_monotonic
        {
            out.push_str(
                "Values in the field increase monotonically when sorted, suggesting a sequence. ",
            );
        }

        // Nullability
        if self.field_def.is_nullable {
            out.push_str("Null values are allowed. ");
        } else {
            out.push_str("Null values are not allowed. ");
        }

        // Default value
        if let Some(default) = &self.field_def.column_default {
            self.write_str(
                out,
                format_args!("The field has a default value defined as \'{}\'.", default),
            );
        }

        // Size
        if let Some(st) = &self.field_stats
            && st.avg > 0.0
        {
            self.write_str(
                out,
                format_args!(
                    "Typical values have an average size of approximately {:2} bytes. ",
                    st.avg
                ),
            );
        }

        // Max size
        if let Some(max) = self.field_def.char_max_length {
            self.write_str(
                out,
                format_args!(
                    "Values are constrained to a maximum length of {} characters. ",
                    max
                ),
            );
        }

        // Numeric precision
        if let Some(np) = self.field_def.numeric_precision {
            self.write_str(
                out,
                format_args!(
                    "Numeric values are stored with a precision of {} digits. ",
                    np
                ),
            );
        }

        // Numeric scale
        if let Some(ns) = self.field_def.numeric_scale {
            self.write_str(
                out,
                format_args!("Numeric values use a scale of {} decimal places. ", ns),
            );
        }

        // Datetime precision
        if let Some(dp) = self.field_def.datetime_precision {
            self.write_str(
                out,
                format_args!(
                    "Datetime values are stored with a precision of {} digits. ",
                    dp
                ),
            );
        }

        if let Some(st) = &self.field_stats {
            if let Some(kurt) = st.kurtosis {
                if kurt <= -0.1 {
                    out.push_str("The values have a left skew. ");
                } else if kurt >= 0.1 {
                    out.push_str("The values have a right skew. ");
                } else {
                    out.push_str("The values exhibit a normal distribution. ");
                }
            }

            if let Some(skew) = st.skewness {
                if skew <= -0.1 {
                    out.push_str("The values have low outliers. ");
                } else if skew >= 0.1 {
                    out.push_str("The values have high outliers. ");
                } else {
                    out.push_str("The values are normally ditributed. ");
                }
            }
        }

        // Char class
        self.write_char_class(out);
    }

    /// The function `write_type_sentence` writes a sentence describing the type of data based
    /// on certain conditions and data types.
    ///
    /// Arguments:
    ///
    /// * `out`: The `write_type_sentence` function takes a mutable reference to a `String` named `out`
    ///   as a parameter. This function is responsible for constructing a sentence describing the type of
    ///   data based on certain conditions and appending it to the provided `out` string.
    fn write_type_sentence(&self, out: &mut String) {
        let hedge = match self.field_def.type_confidence {
            Some(c) if c > 0.65 => "The field is ",
            _ => "The field may represent ",
        };

        out.push_str(hedge);

        match self.field_def.canonical_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => self.write_str(
                out,
                format_args!(
                    "an integer-valued field of \'{}\' variant. ",
                    self.field_def.canonical_type
                ),
            ),
            DataType::Float16 | DataType::Float32 | DataType::Float64 => self.write_str(
                out,
                format_args!(
                    "a continuous numeric measurement of \'{}\' variant. ",
                    self.field_def.canonical_type
                ),
            ),

            DataType::Date32 => self.write_str(
                out,
                format_args!(
                    "a calendar date field of \'{}\' variant. ",
                    self.field_def.canonical_type
                ),
            ),

            DataType::Time32(p) | DataType::Time64(p) => self.write_str(
                out,
                format_args!(
                    "a time-of-day field of \'{}\' variant and \'{:?}\' resolution. ",
                    self.field_def.canonical_type, p
                ),
            ),

            DataType::Boolean => out.push_str("a boolean field. "),

            DataType::Null => out.push_str("a null field. "),

            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                out.push_str("a binary field. ")
            }

            DataType::FixedSizeBinary(16) => out.push_str("a uuid fied. "),

            DataType::FixedSizeBinary(s) => {
                self.write_str(out, format_args!("a \'{}\' fixed size binary field. ", s))
            }

            DataType::FixedSizeList(_, p) => {
                self.write_str(out, format_args!("a \'{}\' fixed size list field. ", p))
            }

            DataType::Timestamp(p, _) => self.write_str(
                out,
                format_args!(
                    "a timestamp field of \'{:?}\' resolution representing a moment in time. ",
                    p
                ),
            ),

            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_) => out.push_str("a list field. "),

            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => self.write_str(
                out,
                format_args!(
                    "a textual string field of \'{:?}\' variant. ",
                    self.field_def.canonical_type
                ),
            ),

            _ => self.write_str(
                out,
                format_args!(
                    "a structured or complex field of \'{}\' variant. ",
                    self.field_def.canonical_type
                ),
            ),
        }
    }

    /// The function `write_numeric` adds raw numeric signal to the text buffer.
    /// Intention is to limit replica or self alignment of similar columns by type and text generalizations.
    /// There is an information limit when columns are replicas or booleans i.e There's no distinguishing
    /// signal
    ///
    /// Arguments:
    ///
    /// * `out`: The `write_numeric` function takes a mutable reference to a `String` named `out` as
    ///   a parameter.
    fn write_numeric(&self, out: &mut String) {
        let Some(fs) = &self.field_stats else {
            return;
        };

        let min_val = fs.max_val.or(fs.character_min_length.map(|v| v as f64));
        let max_val = fs.max_val.or(fs.character_max_length.map(|v| v as f64));

        if let Some(min) = min_val {
            self.write_str(out, format_args!("Minimum: {}, ", min));
        }

        if let Some(qq) = &fs.quantiles_f64 {
            self.write_str(
                out,
                format_args!(
                    "P10: {}, P25: {}, Median: {}, P75: {}, P90: {}, ",
                    qq[0], qq[2], qq[5], qq[7], qq[10]
                ),
            );
        } else {
            self.write_str(out, format_args!("Median: {}, ", fs.median));
        }

        if let Some(max) = max_val {
            self.write_str(out, format_args!("Maximum: {}. ", max));
        }
    }

    /// The function `write_cardinality` analyzes the cardinality of a field's values and provides a
    /// description based on the uniqueness level.
    ///
    /// Arguments:
    ///
    /// * `out`: The `write_cardinality` function takes a mutable reference to a `String` named `out` as
    ///   a parameter. This function checks the cardinality of a field and appends a message describing
    ///   the cardinality to the `out` string based on the cardinality value.
    fn write_cardinality(&self, out: &mut String) {
        if let Some(st) = &self.field_stats
            && let Some(cardinality) = st.cardinality
        {
            if cardinality > 0.89 {
                out.push_str("Values in the field are highly unique. ");
            } else if cardinality > 0.49 {
                out.push_str("Values in the field are moderately unique. ");
            } else if cardinality > 0.009 {
                out.push_str("Values in the field repeat frequently, exhibiting low-cardinality. ");
            } else {
                out.push_str("Values in the field revolve around two or three values. ");
            }
        }
    }

    /// The function `write_char_class` analyzes the character composition of values and provides
    /// insights based on the predominant characters present.
    ///
    /// Arguments:
    ///
    /// * `out`: The `out` parameter in the `write_char_class` function is a mutable reference to a
    ///   `String` where the messages about the character class will be appended. This function analyzes
    ///   the character class signature and appends a message to the `out` string based on the
    ///   characteristics of the character class.
    fn write_char_class(&self, out: &mut String) {
        const DIGIT: usize = 0;
        const ALPHA: usize = 1;
        const WHITES: usize = 2;
        const PUNCT: usize = 3;
        const SPECIAL: usize = 4;

        let Some(st) = &self.field_stats else {
            return;
        };
        let Some(sign) = &st.char_class_signature else {
            return;
        };

        let ColSignature::CharClass(sig) = sign else {
            return;
        };

        let msg = match () {
            _ if sig[DIGIT] > 0.8 && sig[ALPHA] < 0.1 && sig[SPECIAL] < 0.1 => {
                "Values consist primarily of digits, suggesting numeric identifiers or codes."
            }
            _ if sig[DIGIT] > 0.5 && sig[ALPHA] > 0.3 => {
                "Values contain a mix of digits and letters, suggesting alphanumeric identifiers or codes."
            }
            _ if sig[ALPHA] > 0.8 && sig[WHITES] < 0.1 => {
                "Values are predominantly single alphabetic tokens, indicating labels or categories."
            }
            _ if sig[ALPHA] > 0.7 && sig[WHITES] > 0.1 => {
                "Values are predominantly alphabetic with whitespace, indicating descriptive or free-form text."
            }
            _ if sig[SPECIAL] > 0.5 => {
                "Values are heavily symbol-laden, suggesting encoded, escaped, or structured strings."
            }
            _ if sig[SPECIAL] > 0.3 => {
                "Values contain a significant proportion of symbols, suggesting formatted or encoded strings."
            }
            _ if sig[PUNCT] > 0.3 && sig[DIGIT] > 0.3 => {
                "Values contain punctuation and digits, suggesting dates, versions, or structured numeric formats."
            }
            _ if sig[WHITES] > 0.5 => {
                "Values contain predominantly whitespace, suggesting sparse or poorly formatted data."
            }
            _ if sig[DIGIT] > 0.3 && sig[ALPHA] > 0.3 && sig[SPECIAL] > 0.2 => {
                "Values exhibit high compositional diversity, suggesting hashes, tokens, or complex identifiers."
            }
            _ => "Values exhibit almost undescribable character composition.",
        };
        out.push_str(msg);
    }

    fn write_str(&self, out: &mut String, args: fmt::Arguments<'_>) {
        out.write_fmt(args).unwrap()
    }
}

impl Storable for FieldDef {
    type SearchResult = FieldMatch;

    fn get_id(&self) -> Uuid {
        self.id
    }

    fn table_id(&self) -> &Uuid {
        &self.table_id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn schema(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("table_id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("canonical_type", DataType::Utf8, false),
            Field::new("type_confidence", DataType::Float32, true),
            Field::new("column_default", DataType::Utf8, true),
            Field::new("is_nullable", DataType::Boolean, false),
            Field::new("char_max_length", DataType::Int32, true),
            Field::new("numeric_precision", DataType::Int32, true),
            Field::new("numeric_scale", DataType::Int32, true),
            Field::new("datetime_precision", DataType::Int32, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                true,
            ),
        ]))
    }

    fn result_columns() -> Vec<String> {
        vec![
            "id".to_string(),
            "table_id".to_string(),
            "name".to_string(),
            "canonical_type".to_string(),
            "type_confidence".to_string(),
            "column_default".to_string(),
            "is_nullable".to_string(),
            "char_max_length".to_string(),
            "numeric_precision".to_string(),
            "numeric_scale".to_string(),
            "datetime_precision".to_string(),
            "vector".to_string(),
        ]
    }

    fn vtable_name() -> &'static str {
        "field_def"
    }
}
