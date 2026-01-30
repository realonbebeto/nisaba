use arrow::datatypes::{DataType, Field, Schema};
use std::{
    fmt::{self, Write},
    hash::Hash,
    sync::Arc,
};
use uuid::Uuid;

use crate::analyzer::{inference::FieldMetrics, report::FieldMatch, retriever::Storable};

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Global unique Id for the field
    pub id: Uuid,
    /// Id of silo in which the field is member
    pub silo_id: String,
    /// Name of the schema in which the field is member
    pub table_schema: String,
    /// Name of the table in which the field is member
    pub table_name: String,
    /// Name of the field
    pub name: String,
    /// Arrow DataType of the field
    pub canonical_type: DataType,
    /// Optional floating-point confidence in DataType of the field
    pub type_confidence: Option<f32>,
    /// Optional floating-point measure of the uniquness/distinctness
    pub cardinality: Option<f32>,
    /// Floating-point average of the length of String/Binary field
    pub avg_byte_length: Option<f32>,
    /// Boolean flag if the field is sequential for integer/time/timestamp/date field
    pub is_monotonic: bool,
    // Optional character class signature array of a String field
    pub char_class_signature: Option<[f32; 4]>, // [digit, alpha, whitespace, symbol]
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
            "FieldDef: \n- id: {} \n- silo id: {} \n- table name: {} \n- field name: {} \n- data type: {} \n- is monotonic: {} \n- is nullable: {}",
            self.id,
            self.silo_id,
            self.table_name,
            self.name,
            self.canonical_type,
            self.is_monotonic,
            self.is_nullable
        )?;

        if let Some(tc) = self.type_confidence {
            write!(f, "\n- type confidence: {}", tc)?;
        }

        if let Some(c) = self.cardinality {
            write!(f, "\n- cardinality: {}", c)?;
        }

        if let Some(avg) = self.avg_byte_length {
            write!(f, "\n- avg byte length: {}", avg)?;
        }

        if let Some(tc) = self.type_confidence {
            write!(f, "\n- type confidence: {}", tc)?;
        }

        if let Some(sig) = self.char_class_signature {
            write!(f, "\n- char class signature: {:?}", sig)?;
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

impl FieldDef {
    /// The function `enrich_from_arrow` updates certain fields of a struct based on the provided
    /// `FieldMetrics` if it is not None.
    ///
    /// Arguments:
    ///
    /// * `metrics`: The `enrich_from_arrow` function takes in an optional reference to `FieldMetrics`
    ///   struct as the `metrics` parameter. If the `metrics` parameter is `Some`, the function will
    ///   update certain fields of the struct it is called on based on the values in the `FieldMetrics`
    ///   struct
    pub fn enrich_from_arrow(&mut self, metrics: Option<&FieldMetrics>) {
        if let Some(m) = metrics {
            self.char_class_signature = Some(m.char_class_signature);
            self.is_monotonic = m.monotonicity;
            self.cardinality = Some(m.cardinality);
            self.avg_byte_length = m.avg_byte_length;
        }
    }

    /// The `write_field_def_paragraph` function generates a detailed description of a field's
    /// properties such as type, cardinality, nullability, default value, size constraints, numeric
    /// precision, and datetime precision.
    ///
    /// Arguments:
    ///
    /// * `out`: The `out` parameter is a mutable reference to a `String` where the generated paragraph will be written into.
    pub fn write_field_def_paragraph(&self, out: &mut String) {
        let base = 256;
        let extra = self.column_default.as_ref().map_or(0, |s| s.len() + 40)
            + self.avg_byte_length.map_or(0, |_| 70)
            + self.char_max_length.map_or(0, |_| 70)
            + self.numeric_precision.map_or(0, |_| 60)
            + self.numeric_scale.map_or(0, |_| 60)
            + self.datetime_precision.map_or(0, |_| 70);

        out.reserve(base + extra);

        // Identity + context
        // Thoughts are that identity is irrelevant considering names and tables could have little to no significance e.g table1, table2

        // Canonical type
        self.write_type_sentence(out);

        // Cardinality
        self.write_cardinality(out);

        // Monotonicity
        if self.is_monotonic {
            out.push_str(
                "Values in the field increase monotonically when sorted, suggesting a sequence.",
            );
        }

        // Nullability
        if self.is_nullable {
            out.push_str("The field may contain null values.");
        } else {
            out.push_str("The field does not allow null values");
        }

        // Default value
        if let Some(default) = &self.column_default {
            self.write_str(
                out,
                format_args!("The field has a default value defined as \"{}\".", default),
            );
        }

        // Size
        if let Some(avg) = self.avg_byte_length {
            self.write_str(
                out,
                format_args!(
                    "Typical values have an average size of approximately {:2} bytes",
                    avg
                ),
            );
        }

        // Max size
        if let Some(max) = self.char_max_length {
            self.write_str(
                out,
                format_args!(
                    "Values are constrained to a maximum length of {} characters.",
                    max
                ),
            );
        }

        // Numeric precision
        if let Some(np) = self.numeric_precision {
            self.write_str(
                out,
                format_args!(
                    "Numeric values are stored with a precision of {} digits",
                    np
                ),
            );
        }

        // Numeric scale
        if let Some(ns) = self.numeric_scale {
            self.write_str(
                out,
                format_args!("Numeric values use a scale of {} decimal places", ns),
            );
        }

        // Datetime precision
        if let Some(dp) = self.datetime_precision {
            self.write_str(
                out,
                format_args!(
                    "Datetime values are stored with a precision of {} digits",
                    dp
                ),
            );
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
        let hedge = match self.type_confidence {
            Some(c) if c > 0.91 => "It is ",
            Some(c) if c > 0.71 => "It is very likely ",
            Some(c) if c > 0.49 => "It appears to be ",
            _ => "It may represent ",
        };

        out.push_str(hedge);

        match self.canonical_type {
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
                    "an integer-valued field of \"{}\" variant.",
                    self.canonical_type
                ),
            ),
            DataType::Float16 | DataType::Float32 | DataType::Float64 => self.write_str(
                out,
                format_args!(
                    "a continuous numeric measurement of \"{}\" variant.",
                    self.canonical_type
                ),
            ),

            DataType::Date32 | DataType::Date64 => self.write_str(
                out,
                format_args!(
                    "a calendar date field of \"{}\" variant.",
                    self.canonical_type
                ),
            ),

            DataType::Time32(p) | DataType::Time64(p) => self.write_str(
                out,
                format_args!(
                    "a time-of-day field of \"{}\" variant and \"{:?}\" resolution.",
                    self.canonical_type, p
                ),
            ),

            DataType::Boolean => out.push_str("a boolean field."),

            DataType::Null => out.push_str("a null field."),

            DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
                out.push_str("a binary field.")
            }

            DataType::FixedSizeBinary(16) => out.push_str("a uuid fied."),

            DataType::FixedSizeBinary(s) => {
                self.write_str(out, format_args!("a \"{}\" fixed size binary field.", s))
            }

            DataType::FixedSizeList(_, p) => {
                self.write_str(out, format_args!("a \"{}\" fixed size list field", p))
            }

            DataType::Timestamp(p, _) => self.write_str(
                out,
                format_args!(
                    "a timestamp field of \"{:?}\" resolution representing a moment in time.",
                    p
                ),
            ),

            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_) => out.push_str("a list field"),

            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => self.write_str(
                out,
                format_args!(
                    "a textual string field of \"{:?}\" variant",
                    self.canonical_type
                ),
            ),

            _ => self.write_str(
                out,
                format_args!(
                    "a structured or complex field of \"{}\" variant",
                    self.canonical_type
                ),
            ),
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
        if let Some(cardinality) = self.cardinality {
            if cardinality > 0.95 {
                out.push_str("Values in the field are highly unique");
            } else if cardinality < 0.1 {
                out.push_str("Values in the field repeat frequently, exhibiting low-cardinality.");
            } else {
                out.push_str("Values in the field exhibit moderate diversity");
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
        const SYMBOL: usize = 3;

        if let Some(sig) = self.char_class_signature {
            let msg = match () {
                _ if sig[DIGIT] > 0.8 && sig[ALPHA] < 0.1 => {
                    "Values consist primarily of digits, suggesting numeric identifiers or codes. "
                }
                _ if sig[ALPHA] > 0.7 => {
                    "Values are predominantly alphabetic, indicating descriptive text. "
                }
                _ if sig[SYMBOL] > 0.3 => {
                    "Values contain a significant number if symbols, suggesting encoded or formatted strings."
                }
                _ => "Values exhibit a mixed character composition. ",
            };

            out.push_str(msg);
        }
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

    fn silo_id(&self) -> &str {
        &self.silo_id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn schema(dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("silo_id", DataType::Utf8, false),
            Field::new("table_schema", DataType::Utf8, false),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("canonical_type", DataType::Utf8, false),
            Field::new("type_confidence", DataType::Float32, true),
            Field::new("cardinality", DataType::Float32, true),
            Field::new("avg_byte_length", DataType::Float32, true),
            Field::new("is_monotonic", DataType::Boolean, false),
            Field::new(
                "char_class_signature",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
                true,
            ),
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
            "silo_id".to_string(),
            "table_schema".to_string(),
            "table_name".to_string(),
            "name".to_string(),
            "canonical_type".to_string(),
            "type_confidence".to_string(),
            "cardinality".to_string(),
            "avg_byte_length".to_string(),
            "is_monotonic".to_string(),
            "char_class_signature".to_string(),
            "column_default".to_string(),
            "is_nullable".to_string(),
            "char_max_length".to_string(),
            "numeric_precision".to_string(),
            "numeric_scale".to_string(),
            "datetime_precision".to_string(),
            "_distance".to_string(),
        ]
    }

    fn vtable_name() -> &'static str {
        "field_def"
    }
}
