# nisaba

This library brings a disciplined structure to data reconciliation, providing deterministic conflict resolution, data validation and quality checks, clean merging primitives, and a trustworthy source of truth—so engineers can focus on building systems, not untangling inconsistent or unreliable records.

## Naming

In Mesopotamian/Sumeria mythology, [Nisaba](https://en.wikipedia.org/wiki/Nisaba) is the goddess of writing, accounting, and the orderly keeping of records, entrusted with maintaining clarity across ledgers and knowledge archives.

## Roadmap

### v0.7 - Validation and Reconciliation Intelligence

- [ ] Schema evolution utilities
- [ ] Data Profiling Helpers
- [ ] AI Reports/Recommendations

### v0.6 - Extensibility + Quality

- [ ] Data Quality Checks
- [ ] Configurable Merge Rules
- [ ] File/struct adapters: CSV, Parquet
- [ ] Conflict Detection + Diff Output

### v0.5 - Foundation

- [x] Deterministic Reconciliation Engines
- [x] Structured Reconciliation Reports
- [x] SQL/NoSQL/File Support (MySQL, PostgreSQL, SQLite/MongoDB/CSV, Excel, Parquet)
- [x] CI/CD

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

* Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.