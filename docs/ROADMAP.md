# Roadmap

## Vision

Build a **data management and trust framework** for describing, reconciling, validating, and transforming both schemaless and schemafull data across heterogeneous data stores.

The framework is designed around a simple belief:

> Reconciliation is the strongest form of validation.

If datasets can be proven equivalent across systems, then quality, correctness, and trust can be layered on top — explicitly and deterministically.

---

## Capability Layers

The framework evolves by progressively unlocking higher-level guarantees:

1. **Reconcile** — Are two datasets equivalent?
2. **Describe** — What is this data supposed to look like?
3. **Validate** — Does the data conform to expectations?
4. **Assure Quality** — Is it reliable enough to trust?
5. **Transform** — How do we safely evolve it?

## v0.4 — Validation & Intelligent Guarantees

**Theme:** _From rules to understanding._

Move from static checks to **signal-driven validation and guidance**.

### Validate
- [ ] Unified validation engine  
  - Schema validation  
  - Quality rule execution  
  - Reconciliation-aware validation
- [ ] Structured validation reports  
  - Shared format with reconciliation outputs

### Quality Intelligence
- [ ] Data profiling primitives  
  - Cardinality, distributions, entropy, sparsity  
  - Type drift and anomaly detection
- [ ] Schema evolution detection  
  - Breaking vs non-breaking changes  
  - Version-aware validation

### Guidance
- [ ] Intelligent recommendations  
  - Suggested quality rules  
  - Suggested reconciliation tolerances  
  - Schema improvement hints

**Outcome:**  
The framework doesn’t just report failures — it explains **risk, impact, and next actions**.

## v0.3 — Description & Quality Foundations

**Theme:** _Make expectations explicit._

Expand from partial description to **first-class data description and quality semantics**.

### Describe
- [ ] Rich schema definitions  
  - Field-level constraints (ranges, formats, enums)  
  - Dataset-level metadata
- [ ] Schema inference & refinement  
  - Infer → review → persist workflows
- [ ] Schema comparison utilities  
  - Backward/forward compatibility checks

### Quality Assurance
- [ ] Core data quality checks  
  - Nullability, uniqueness, ranges, monotonicity  
  - Cross-store invariants
- [ ] Configurable tolerance  
  - Strict vs permissive modes  
  - Threshold-based acceptance

**Outcome:**  
You can define **what the data is** and **what “good” means**, independently of where it lives.


## v0.2 — Reconciliation Core

**Theme:** _Establish truth across systems._

This release focuses on **deterministic reconciliation**, with just enough descriptive metadata to make comparisons meaningful.

### Reconcile
- [x] Deterministic reconciliation engines  
  - Order-independent comparison  
  - Explicit conflict semantics
- [x] Structured reconciliation reports  
- [x] Cross-store reconciliation  
  - SQL: MySQL, PostgreSQL, SQLite  
  - NoSQL: MongoDB  
  - Files: CSV, Excel, Parquet

### Describe
- [x] Minimal schema representation  
  - Logical field types  
  - Nullability and basic constraints
- [x] Store-agnostic internal data model  

**Outcome:**  
You can prove whether two datasets represent the **same truth**, even if they live in different systems.


## What This Framework Is

- A **cross-store data trust kernel**
- A foundation for synchronization, migration, and verification
- A systems-grade alternative to schema-only validation


