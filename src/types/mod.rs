//! Store element implementations
//!
//! Overview
//! - [`FieldDef`]: A vital type responsible for holding all metadata about a field store element.
//! - [`Matchable`]: A trait responsible for id and silo_id on store element retrieval from latent store.
//! - [`MatchCandidate`]: A trait responsible for giving access to attributes of search results after retrieval.
//! - [`TableDef`]: A vital type responsible for logical organization of FieldDef and a table store element.
//!

mod field;
mod table;

pub use field::{FieldDef, FieldProfile};
pub use table::{TableDef, TableRep};
