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

pub use field::FieldDef;
pub use table::{TableDef, TableRep};

// /// Trait used to give access to id and silo_id on storage or retrieval.
// pub trait Matchable {
//     type Id: Clone;
//     type Match: MatchCandidate<Id = Self::Id>;

//     fn silo_id(&self) -> &str;
// }

// /// Trait used to give access to details of a TableMatch/FieldMatch
// #[allow(unused)]
// pub trait MatchCandidate {
//     type Id: Clone;
//     type Body: Storable;

//     fn confidence(&self) -> f32;
//     fn schema_id(&self) -> Self::Id;
//     fn schema_silo_id(&self) -> &str;
//     fn body(&self) -> &Self::Body;
// }
