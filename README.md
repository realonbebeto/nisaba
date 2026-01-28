<h1 align="center">nisaba</h1>
<div align="center">
 <strong>
   Data quality, reconciliation, and validation framework across different data store in Rust
 </strong>
</div>

<br />

<div align="center">
  <!-- Crates version -->
  <a href="https://crates.io/crates/nisaba">
    <img src="https://img.shields.io/crates/v/nisaba.svg?style=flat-square"
    alt="Crates.io version" />
  </a>
  <!-- Downloads -->
  <a href="https://crates.io/crates/nisaba">
    <img src="https://img.shields.io/crates/d/nisaba.svg?style=flat-square"
      alt="Download" />
  </a>
  <!-- docs.rs docs -->
  <a href="https://docs.rs/nisaba">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square"
      alt="docs.rs docs" />
  </a>
</div>
<br/>


# nisaba


This library brings a disciplined structure to data reconciliation, providing deterministic conflict resolution, data validation and quality checks, clean merging primitives, and a trustworthy source of truth—so engineers can focus on building systems, not untangling inconsistent or unreliable records. It exposes the properties and attributes of datasets to make differences, inconsistencies, and risks explicit.

In its initial versions, the library focuses on reconciliation, producing deterministic insights and reports that explain how disparate data silos relate to one another and how they can be merged into a single, unified source of truth. This enables data store migrations and integrations to surface gaps when discovered and resolve them efficiently, before trust is lost downstream.


## Naming


In Mesopotamian/Sumeria mythology, [Nisaba](https://en.wikipedia.org/wiki/Nisaba) is the goddess of writing, accounting, and the orderly keeping of records, entrusted with maintaining clarity across ledgers and knowledge archives.


## Core Concepts and Features


- **Reconciliation-first architecture**: Establishes dataset equivalence across systems as the strongest guarantee of correctness.

- **Deterministic reconciliation engine**: Produces order-independent, repeatable results suitable for CI and automated workflows.

- **Cross-store data support**: Unified handling of tabular data across SQL (MySQL, PostgreSQL, SQLite), NoSQL (MongoDB), and file formats (CSV, Excel, Parquet).

- **Store-agnostic internal data model**: Logical representation of data decoupled from physical storage or format.

## Getting Started


To get started, just add to Cargo.toml

```toml
[dependencies]
nisaba = { version = "0.2.0-beta" }
```

## Usage


Prefer using the example and [the generated docs](https://docs.rs/nisaba) or:

```rust,no_run
use nisaba::{SchemaAnalyzerBuilder, StorageBackend, StorageConfig};

#[tokio::main]
async fn main() {
    // analyzer
    let analyzer = SchemaAnalyzerBuilder::default().build();

    let csv_config = StorageConfig::new_file_backend(StorageBackend::Csv, "./assets/csv").unwrap();

    let parquet_config =
        StorageConfig::new_file_backend(StorageBackend::Parquet, "./assets/parquet").unwrap();

    let _result = analyzer
        .analyze(vec![csv_config, parquet_config])
        .await
        .unwrap();
}

```

## How nisaba works


Assume that a data engineer discovers multiple schema/sources with several tables that have been long been ignored and wants to deduce how they are connected and related between themselves and (or) the contemporary data store. The engineer would:

1. Map out the sources and relevant credentials
2. Setup Nisaba StorageConfigs
3. Setup SchemaAnalyzer
4. Run the analyzer with the storage configs
5. Review the Results/Report for reconcialiation hints


## Roadmap


Successive improvements will allow more features in providing quality and validation as documented in the [roadmap](./ROADMAP.md)


## Versioning


As with most Rust crates, this library is versioned according to
[Semantic Versioning](https://semver.org/). [Breaking changes] will only
be made with good reason, and as infrequently as is feasible. Such
changes will generally be made in releases where the major version
number is increased (note [Cargo's caveat for pre-1.x
versions][caveat]), although [limited exceptions may occur][exceptions].
Increases in the minimum supported Rust version (MSRV) are not
considered breaking, but will result in a minor version bump.

See also [the changelog](./CHANGELOG.md) for details about changes in
recent versions.

[Breaking changes]: https://doc.rust-lang.org/cargo/reference/semver.html
[exceptions]: https://rust-lang.github.io/rfcs/1105-api-evolution.html#principles-of-the-policy
[caveat]: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#default-requirements

## License


Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

## Contribution


Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.