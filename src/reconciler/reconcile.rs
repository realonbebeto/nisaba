use futures::future::join_all;
use nalgebra::DVector;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    vec,
};
use uuid::Uuid;

use crate::{
    CsvInferenceEngine, ExcelInferenceEngine, MySQLInferenceEngine, NoSQLInferenceEngine,
    ParquetInferenceEngine, PostgreSQLInferenceEngine, SimilarityConfig, SqliteInferenceEngine,
    error::NisabaError,
    reconciler::{
        SchemaAnalyzer, SchemaAnalyzerBuilder,
        calculation::{cosine_similarity, jaccard_matched_fields},
        datastore::{DatabaseType, FileStoreType, SourceType},
        report::{
            FieldCluster, FieldMatch, FieldResult, LoadedFields, ReconcileReport, TableCluster,
            TableMatch, TableResult,
        },
        retriever::{LatentStore, Storable},
    },
    types::{FieldDef, TableRep},
};

impl SchemaAnalyzer {
    pub fn builder() -> SchemaAnalyzerBuilder {
        SchemaAnalyzerBuilder::default()
    }

    /// The `latent_store` function sets a predefined latent store.
    ///
    /// Arguments:
    ///
    /// * `latent_store`: This parameter is a shared pre-defined latent shore.
    ///
    pub fn latent_store(mut self, latent_store: Arc<LatentStore>) -> Self {
        self.context.persistence = latent_store;
        self
    }
    /// The `analyze` function runs agains the beforehand provided storage location as Sources, infers their
    /// schemas, clusters them based on similarities.
    ///
    /// Returns:
    ///
    /// The `analyze` function returns a `Result` containing either an `Option` of a vector of
    /// `TableCluster` objects or a `NisabaError`. If the clustering process is successful and at least
    /// one cluster is formed, it returns `Ok(Some(clusters))` with the clustered tables. If no clusters
    /// are formed, it returns `Ok(None)`.
    pub async fn analyze(&self) -> Result<ReconcileReport, NisabaError> {
        let table_reps = self.discover_ecosystem().await?;

        // Step 1: ANN table retrieval (Filter by size compatibility done at source)
        let query_candidates = self.assemble_table_candidates(&table_reps).await?;

        let mut clusters: Vec<TableCluster> = Vec::new();

        // Step 2: Load fields for query and all candidates
        // TODO: Streaming of all candidates
        for query_candidate in &query_candidates {
            let query_table_id = query_candidate.0.id;
            let table_ids = query_candidate
                .1
                .iter()
                .map(|v| v.schema.id)
                .collect::<Vec<Uuid>>();

            let compatible_candidates = query_candidate.1.clone();

            let loaded_fields = self
                .find_candidate_fields::<FieldDef>(&query_table_id, &table_ids)
                .await?;

            // Step 3: Bidirectional field alignment
            let table_alignments = align_fields(
                &loaded_fields.query_fields,
                &loaded_fields.candidate_fields,
                &self.context.config.similarity,
            );

            // Step 4: Cluster fields
            let field_clusters = cluster_fields(
                &loaded_fields.query_fields,
                &table_alignments,
                &loaded_fields.candidate_fields,
            );

            // Step 5: Score tables by alignment quality
            let table_scores = tables_alignment_scores(
                &query_candidate.0,
                &loaded_fields.query_fields,
                &loaded_fields.candidate_fields,
                &field_clusters,
                &compatible_candidates,
            );

            // Step 6: cluster tables
            let table_clusters = cluster_tables(
                table_scores,
                field_clusters,
                &self.context.config.similarity,
            );

            // Step 7: Update clusters that have empty field clusters
            let table_clusters = table_clusters.into_iter().map(|mut tc| {
                if tc.field_clusters.is_empty() {
                    let q_f = loaded_fields
                        .query_fields
                        .iter()
                        .enumerate()
                        .map(|(idx, qf)| FieldCluster {
                            cluster_id: idx as u32,
                            confidence: 1.0,
                            fields: vec![FieldResult {
                                id: qf.schema.id,
                                table_id: qf.schema.table_id,
                                field_name: qf.schema.name.clone(),
                                similarity: 1.0,
                            }],
                        })
                        .collect::<Vec<FieldCluster>>();

                    tc.field_clusters.extend(q_f);
                }
                if tc.field_clusters.is_empty() {
                    let t_ids = tc.tables.iter().map(|v| v.id).collect::<Vec<Uuid>>();

                    for t_id in t_ids {
                        if let Some(fms) = loaded_fields.candidate_fields.get(&t_id) {
                            let c_f = fms
                                .into_iter()
                                .enumerate()
                                .map(|(idx, qf)| FieldCluster {
                                    cluster_id: idx as u32,
                                    confidence: 1.0,
                                    fields: vec![FieldResult {
                                        id: qf.schema.id,
                                        table_id: qf.schema.table_id,
                                        field_name: qf.schema.name.clone(),
                                        similarity: 1.0,
                                    }],
                                })
                                .collect::<Vec<FieldCluster>>();
                            tc.field_clusters.extend(c_f);
                        }
                    }
                }

                tc
            });

            clusters.extend(table_clusters);
        }

        let clusters: HashSet<TableCluster> = clusters.into_iter().collect();
        let clusters = clusters
            .into_iter()
            .enumerate()
            .map(|(idx, mut tc)| {
                tc.cluster_id = idx as u32;
                tc
            })
            .collect::<Vec<TableCluster>>();

        let stats = {
            let lockd = self.context.stats.lock().unwrap();
            let stats = lockd.clone();
            stats
        };

        Ok(
            ReconcileReport {
                stats,
                tables: Some(clusters),
            }, // TODO: Optional CleanUp - Delete the lancedb
        )
    }

    async fn assemble_table_candidates(
        &self,
        table_reps: &[TableRep],
    ) -> Result<Vec<(TableRep, Vec<TableMatch>)>, NisabaError> {
        let futures: Vec<_> = table_reps
            .chunk_by(|a, b| a.silo_id == b.silo_id)
            .map(|chunk| async { self.find_candidates_tables::<TableRep>(chunk).await })
            .collect();

        let results = join_all(futures).await;

        results
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
    }

    async fn find_candidates_tables<T: Storable>(
        &self,
        table_reps: &[TableRep],
    ) -> Result<Vec<(TableRep, Vec<TableMatch>)>, NisabaError> {
        let table_handler = self.context.persistence.table_handler::<T>();

        table_handler
            .search_tables(table_reps, T::result_columns())
            .await
    }

    async fn find_candidate_fields<T: Storable>(
        &self,
        query_table_id: &Uuid,
        table_ids: &[Uuid],
    ) -> Result<LoadedFields, NisabaError> {
        let table_handler = self.context.persistence.table_handler::<T>();

        table_handler
            .load_fields(query_table_id, table_ids, T::result_columns())
            .await
    }

    async fn discover_ecosystem(&self) -> Result<Vec<TableRep>, NisabaError> {
        let mut table_reps = Vec::new();

        let table_handler = self.context.persistence.table_handler::<TableRep>();

        table_handler.initialize().await?;

        let field_handler = self.context.persistence.table_handler::<FieldDef>();

        field_handler.initialize().await?;

        for source in &self.sources {
            let reps = match source.metadata.source_type {
                SourceType::FileStore(FileStoreType::Csv) => {
                    let csv_inferer = CsvInferenceEngine::new();

                    csv_inferer
                        .csv_store_infer(
                            source,
                            self.context.stats.clone(),
                            self.context.threads,
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                SourceType::FileStore(FileStoreType::Excel) => {
                    let excel_inferer = ExcelInferenceEngine::new();
                    excel_inferer
                        .excel_store_infer(
                            source,
                            self.context.stats.clone(),
                            self.context.threads,
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                SourceType::Database(DatabaseType::MongoDB) => {
                    let mongo_inferer = NoSQLInferenceEngine::new();
                    mongo_inferer
                        .mongodb_store_infer(
                            source,
                            self.context.stats.clone(),
                            Arc::new(|table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            }),
                        )
                        .await
                }

                SourceType::Database(DatabaseType::MySQL) => {
                    let mysql_inferer = MySQLInferenceEngine::new();

                    mysql_inferer
                        .mysql_store_infer(source, self.context.stats.clone(), |table_defs| async {
                            table_handler.store_tables(table_defs).await?;
                            Ok(())
                        })
                        .await
                }

                SourceType::FileStore(FileStoreType::Parquet) => {
                    let parquet_inferer = ParquetInferenceEngine::new();

                    parquet_inferer
                        .parquet_store_infer(
                            source,
                            self.context.stats.clone(),
                            4,
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                SourceType::Database(DatabaseType::PostgreSQL) => {
                    let postgres_inferer = PostgreSQLInferenceEngine::new();

                    postgres_inferer
                        .postgres_store_infer(
                            source,
                            self.context.stats.clone(),
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }

                SourceType::Database(DatabaseType::SQLite) => {
                    let sqlite_inferer = SqliteInferenceEngine::new();

                    sqlite_inferer
                        .sqlite_store_infer(
                            source,
                            self.context.stats.clone(),
                            |table_defs| async {
                                table_handler.store_tables(table_defs).await?;
                                Ok(())
                            },
                        )
                        .await
                }
            };

            table_reps.extend(reps?);

            {
                let mut stats = self.context.stats.lock().unwrap();
                stats.sources_analyzed += 1;
            }
        }

        table_handler.create_index().await?;

        field_handler.create_index().await?;

        Ok(table_reps)
    }
}

#[derive(Clone, Debug)]
struct FieldAlignment {
    pub query_field_id: Uuid,
    pub candidate_field_id: Uuid,
    pub similarity: f32,
}

#[allow(unused)]
#[derive(Clone, Debug)]
struct TableAlignment {
    // Query table id
    pub table_id: Uuid,
    pub candidate_tid: Uuid,
    pub alignments: Vec<FieldAlignment>,
    pub query_coverage: f32,
    pub candidate_coverage: f32,
    // pub avg_similarity: f32,
}

/// Field cluster across multiple tables
#[derive(Debug, Clone)]
struct FieldRawCluster {
    pub query_field_id: Uuid,
    pub aligned_fields: Vec<FieldResult>,
}

/// Each candidate table session (since we have multiple candidate)
/// has fields. And each field is compared with query fields bi-directionally.
/// Mutually similar query field and candidate field pairs are
fn align_fields(
    query_fields: &[FieldMatch],
    candidate_fields: &HashMap<Uuid, Vec<FieldMatch>>,
    config: &SimilarityConfig,
) -> Vec<TableAlignment> {
    let mut table_alignments = Vec::with_capacity(candidate_fields.len());

    // Process each candidate table fields independently
    for (table_id, c_fields) in candidate_fields {
        let mut alignments = Vec::new();
        let mut aligned_query_fields = HashSet::new();
        let mut aligned_candidate_fields = HashSet::new();

        // Step 1: Query -> Candidate
        // For each query field, find the best matching candidate field
        for query_field in query_fields {
            let mut best_match: Option<(Uuid, f32)> = None;

            for c_field in c_fields {
                let similarity = cosine_similarity(&query_field.embedding, &c_field.embedding);

                if similarity >= config.threshold {
                    if best_match.is_none() || similarity > best_match.unwrap().1 {
                        best_match = Some((c_field.schema.id, similarity));
                    }
                }
            }

            if let Some((c_field_id, similarity)) = best_match {
                aligned_query_fields.insert(query_field.schema.id);
                aligned_candidate_fields.insert(c_field_id);

                alignments.push(FieldAlignment {
                    query_field_id: query_field.schema.id,
                    candidate_field_id: c_field_id,
                    similarity,
                });
            }
        }

        // Step 2: Candidate -> Query
        // For each candidate field, find the best matching query field
        // This validates that the alignment is mutual
        let mut reverse_alignments = HashSet::new();

        for c_field in c_fields {
            let mut best_match: Option<(Uuid, f32)> = None;

            for query_field in query_fields {
                let similarity = cosine_similarity(&query_field.embedding, &c_field.embedding);

                if similarity >= config.threshold {
                    if best_match.is_none() || similarity > best_match.unwrap().1 {
                        best_match = Some((query_field.schema.id, similarity));
                    }
                }
            }

            if let Some((query_field_id, _)) = best_match {
                reverse_alignments.insert((query_field_id, c_field.schema.id));
            }
        }

        // Keep only bidirectional alignments (mutual best matches)
        alignments.retain(|alignment| {
            reverse_alignments.contains(&(alignment.query_field_id, alignment.candidate_field_id))
        });

        // Re-calculate coverage based on bi-directional alignments
        aligned_query_fields.clear();
        aligned_candidate_fields.clear();

        for alignment in &alignments {
            aligned_query_fields.insert(alignment.query_field_id);
            aligned_candidate_fields.insert(alignment.candidate_field_id);
        }

        // Compute metrics
        let query_coverage = if !query_fields.is_empty() {
            aligned_query_fields.len() as f32 / query_fields.len() as f32
        } else {
            0.0
        };

        let candidate_coverage = if !c_fields.is_empty() {
            aligned_candidate_fields.len() as f32 / c_fields.len() as f32
        } else {
            0.0
        };

        // Only include tables that meet coverage thresholds
        // Coverage can be experimented with to give hints where we hit an information limit
        // i.e replica or boolean columns thus replica or self alignment
        if query_coverage >= 0.95 && candidate_coverage >= 0.95 {
            table_alignments.push(TableAlignment {
                table_id: query_fields[0].schema.table_id,
                candidate_tid: *table_id,
                alignments,
                candidate_coverage,
                query_coverage,
            });
        }
    }

    table_alignments
}

fn cluster_fields(
    query_fields: &[FieldMatch],
    table_alignments: &[TableAlignment],
    candidate_fields: &HashMap<Uuid, Vec<FieldMatch>>,
) -> Vec<FieldRawCluster> {
    let mut field_clusters = Vec::with_capacity(table_alignments.len());

    // For each query field, collect all its alignments across tables
    for query_field in query_fields.iter() {
        let mut aligned_fields = Vec::new();

        // Collect all alignments
        let all_field_aligns = table_alignments
            .iter()
            .map(|ta| {
                let fa = ta
                    .alignments
                    .iter()
                    .map(|v| (ta.candidate_tid, v))
                    .collect::<Vec<(Uuid, &FieldAlignment)>>();

                fa
            })
            .collect::<Vec<Vec<(Uuid, &FieldAlignment)>>>()
            .concat();

        // Find query-field alignment match
        let match_aligns = all_field_aligns
            .iter()
            .filter_map(|(cid, fa)| {
                if fa.query_field_id == query_field.schema.id {
                    Some((*cid, *fa))
                } else {
                    None
                }
            })
            .collect::<Vec<(Uuid, &FieldAlignment)>>();

        if !match_aligns.is_empty() {
            for (cid, fa) in match_aligns {
                if let Some(candidate_metas) = candidate_fields.get(&cid) {
                    if let Some(cm) = candidate_metas
                        .iter()
                        .find(|v| v.schema.id == fa.candidate_field_id)
                    {
                        aligned_fields.push(FieldResult {
                            id: cm.schema.id,
                            table_id: cm.schema.table_id,
                            field_name: cm.schema.name.clone(),
                            similarity: fa.similarity,
                        });
                    }
                }
            }

            aligned_fields.push(FieldResult {
                id: query_field.schema.id,
                table_id: query_field.schema.table_id,
                field_name: query_field.schema.name.clone(),
                similarity: 1.0,
            });
        }

        field_clusters.push(FieldRawCluster {
            query_field_id: query_field.schema.id,
            aligned_fields,
        });
    }

    field_clusters
}

/// Alignment scores for a candidate table
#[derive(Debug, Clone)]
pub struct TableAlignmentScore {
    pub table_id: Uuid,
    pub table_name: String,
    pub silo_id: String,

    /// Query coverage: matched_query / total_query
    pub query_coverage: f32,

    /// Combined final score
    pub final_score: f32,

    /// Per-query-field similarities: (query_field_index, best_similarity).
    /// Only field entries where a match was found above threshold are included.
    /// This is the source data for similarity vectors and Jaccard overlap
    pub per_field_similarities: Vec<(usize, f32)>,
}
fn tables_alignment_scores(
    query_table: &TableRep,
    query_fields: &[FieldMatch],
    candidate_fields: &HashMap<Uuid, Vec<FieldMatch>>,
    field_clusters: &[FieldRawCluster],
    table_matches: &[TableMatch],
) -> Vec<TableAlignmentScore> {
    let mut table_scores = Vec::new();

    for table_match in table_matches {
        let table_id = table_match.schema.id;

        // Get fields for a matched/candidate table
        let Some(c_fields) = candidate_fields.get(&table_id) else {
            continue;
        };

        let mut matched_query_field_ids = HashSet::new();
        let mut matched_candidate_field_ids = HashSet::new();
        let mut total_similarity = 0.0f32;
        let mut match_count = 0;
        let mut per_field_similarities = Vec::new();

        // Go through each group of fields related/matched to query_field_id
        // and find those that are member to a table (field results for a specific table)
        for cluster in field_clusters {
            let fieldc_results: Vec<_> = cluster
                .aligned_fields
                .iter()
                .filter(|a| a.table_id == table_id)
                .collect();

            if fieldc_results.is_empty() {
                continue;
            }
            // The candidate table has matcching FieldRawCluster(group of fields)
            // Thefore a query_field_id associated to FieldRawCluster has been found
            matched_query_field_ids.insert(cluster.query_field_id);

            // Track which candidate fields matched
            //
            if let Some(best) = fieldc_results.iter().max_by(|a, b| {
                a.similarity
                    .partial_cmp(&b.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                matched_candidate_field_ids.insert(best.id);
                total_similarity += best.similarity;
                match_count += 1;

                // Mapping query field -> index
                if let Some(idx) = query_fields
                    .iter()
                    .position(|q| q.schema.id == cluster.query_field_id)
                {
                    per_field_similarities.push((idx, best.similarity));
                }
            }
        }

        // Compute coverage metrics
        let query_coverage = if !c_fields.is_empty() {
            matched_query_field_ids.len() as f32 / query_fields.len() as f32
        } else {
            0.0
        };

        let candidate_coverage = if !c_fields.is_empty() {
            matched_candidate_field_ids.len() as f32 / c_fields.len() as f32
        } else {
            0.0
        };

        // Compute size penalty
        let size_penalty = query_fields.len().min(c_fields.len()) as f32
            / query_fields.len().max(c_fields.len()) as f32;

        let alignment_similarity = if match_count > 0 {
            total_similarity / match_count as f32
        } else {
            0.0
        };

        let final_score = alignment_similarity * 0.4
            + query_coverage * 0.3
            + candidate_coverage * 0.15
            + size_penalty * 0.15;

        table_scores.push(TableAlignmentScore {
            table_id,
            table_name: table_match.schema.name.clone(),
            silo_id: table_match.schema.silo_id.clone(),
            query_coverage,
            final_score,
            per_field_similarities,
        });
    }

    let per_sim = query_fields
        .iter()
        .enumerate()
        .map(|(idx, _)| (idx, 1.0))
        .collect::<Vec<(usize, f32)>>();

    table_scores.push(TableAlignmentScore {
        table_id: query_table.id,
        table_name: query_table.name.clone(),
        silo_id: query_table.silo_id.clone(),
        query_coverage: 1.0,
        final_score: 1.0,
        per_field_similarities: per_sim,
    });

    table_scores
}

fn cluster_tables(
    table_scores: Vec<TableAlignmentScore>,
    field_clusters: Vec<FieldRawCluster>,
    config: &SimilarityConfig,
) -> Vec<TableCluster> {
    if table_scores.is_empty() {
        return Vec::new();
    }

    // Exclude tables that matched nothing
    let (scorable, _): (Vec<_>, Vec<_>) = table_scores
        .into_iter()
        .partition(|ts| !ts.per_field_similarities.is_empty());

    // Unified vector space - all vectors share the same dimension
    let n_query_fields = scorable
        .iter()
        .flat_map(|ts| ts.per_field_similarities.iter().map(|(i, _)| i))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let vectors = build_similarity_vectors(&scorable, n_query_fields);

    // Adjacency graph with dual gate (cosine + jaccard)
    let graph = build_similarity_graph(config, &scorable, &vectors);

    // Connected tables
    let tables = connected_tables(&scorable, &graph);

    // Build clusters from tables
    let clusters: Vec<TableCluster> = tables
        .into_iter()
        .enumerate()
        .map(|(cluster_id, group)| build_table_cluster(cluster_id as u32, group, &field_clusters))
        .collect();

    clusters
}

/// Unified similarity vector
/// Every table vector has a standard length (the number of query fields - `n_query_fields`)
/// It is a global dimension computed from the full table set
/// Index i = best similarity of a query field of this table achived
/// against a candidate field in another table or ).0 if unmatched
fn build_similarity_vectors(
    table_scores: &[TableAlignmentScore],
    n_query_fields: usize,
) -> HashMap<Uuid, DVector<f32>> {
    table_scores
        .iter()
        .map(|ts| {
            let mut vector = vec![0.0f32; n_query_fields];
            for &(idx, sim) in &ts.per_field_similarities {
                if idx < n_query_fields && sim > vector[idx] {
                    vector[idx] = sim;
                }
            }

            (ts.table_id, DVector::from_vec(vector))
        })
        .collect()
}

/// Similarity graph with dual gate
/// A edge requires BOTH:
///     cosine similarity and jaccard similarity
/// Cosine alone would connect tables on semantics only (high risk of disjointed matches)
/// so the jaccard ensures tables connect on individual query fields matching
fn build_similarity_graph(
    config: &SimilarityConfig,
    table_scores: &[TableAlignmentScore],
    vectors: &HashMap<Uuid, DVector<f32>>,
) -> HashMap<Uuid, Vec<Uuid>> {
    let mut graph: HashMap<Uuid, Vec<Uuid>> = table_scores
        .iter()
        .map(|ts| (ts.table_id, Vec::new()))
        .collect();

    for i in 0..table_scores.len() {
        for j in (i + 1)..table_scores.len() {
            let a = &table_scores[i];
            let b = &table_scores[j];

            let cos = cosine_similarity(&vectors[&a.table_id], &vectors[&b.table_id]);

            if cos < config.threshold {
                continue;
            }

            let jac = jaccard_matched_fields(a, b);

            if jac < config.threshold {
                continue;
            }

            graph.entry(a.table_id).or_default().push(b.table_id);
            graph.entry(b.table_id).or_default().push(a.table_id);
        }
    }

    graph
}

/// Connected tables vis BFS
/// Flood-fill - order independent,determinstic
/// Transitive similarity is allowed but drift effects should be handled by cosine and jaccard thresholds
fn connected_tables(
    table_scores: &[TableAlignmentScore],
    graph: &HashMap<Uuid, Vec<Uuid>>,
) -> Vec<Vec<TableAlignmentScore>> {
    let score_map: HashMap<Uuid, &TableAlignmentScore> =
        table_scores.iter().map(|ts| (ts.table_id, ts)).collect();

    let mut visited: HashSet<Uuid> = HashSet::new();
    let mut tables: Vec<Vec<TableAlignmentScore>> = Vec::new();

    for ts in table_scores {
        if visited.contains(&ts.table_id) {
            continue;
        }

        let mut table = Vec::new();
        let mut queue = vec![ts.table_id];
        visited.insert(ts.table_id);

        while let Some(current) = queue.pop() {
            table.push((score_map[&current]).clone());

            for neighbor in graph.get(&current).into_iter().flatten() {
                if visited.insert(*neighbor) {
                    queue.push(*neighbor);
                }
            }
        }

        tables.push(table);
    }

    tables
}

fn build_table_cluster(
    cluster_id: u32,
    table_scores: Vec<TableAlignmentScore>,
    field_clusters: &[FieldRawCluster],
) -> TableCluster {
    let n = table_scores.len() as f32;

    let avg_final_score = table_scores.iter().map(|ts| ts.final_score).sum::<f32>() / n;
    let avg_coverage = table_scores.iter().map(|ts| ts.query_coverage).sum::<f32>() / n;

    let tables: Vec<TableResult> = table_scores
        .iter()
        .map(|ts| TableResult {
            id: ts.table_id,
            silo_id: ts.silo_id.clone(),
            table_name: ts.table_name.clone(),
        })
        .collect();

    let field_clusters = extract_and_score_field_clusters(&table_scores, field_clusters);

    TableCluster {
        cluster_id,
        tables,
        field_clusters,
        confidence: avg_final_score,
        coverage_score: avg_coverage,
    }
}

/// Extract field clusters and recomute their confidence
/// A field cluster in 4/4 tables outscores one in 1/4 even at raw similarity
/// Structural centrality is promoted as fields across the whole cluster are more canonical than
/// fields unique to a single table
fn extract_and_score_field_clusters(
    table_scores: &[TableAlignmentScore],
    field_clusters: &[FieldRawCluster],
) -> Vec<FieldCluster> {
    let table_ids: HashSet<Uuid> = table_scores.iter().map(|ts| ts.table_id).collect();
    let n_tables = table_scores.len() as f32;

    field_clusters
        .iter()
        .enumerate()
        .filter_map(|(idx, cluster)| {
            let relevant_fields: Vec<FieldResult> = cluster
                .aligned_fields
                .iter()
                .filter(|af| table_ids.contains(&af.table_id))
                .map(|af| FieldResult {
                    id: af.id,
                    table_id: af.table_id,
                    field_name: af.field_name.clone(),
                    similarity: af.similarity,
                })
                .collect();

            if relevant_fields.is_empty() {
                return None;
            }

            let covered_tables: HashSet<Uuid> =
                relevant_fields.iter().map(|f| f.table_id).collect();

            let table_coverage = covered_tables.len() as f32 / n_tables;

            let avg_sim = cluster
                .aligned_fields
                .iter()
                .filter(|af| table_ids.contains(&af.table_id))
                .map(|af| af.similarity)
                .sum::<f32>()
                / relevant_fields.len() as f32;

            Some(FieldCluster {
                cluster_id: idx as u32,
                fields: relevant_fields,
                confidence: avg_sim * table_coverage,
            })
        })
        .collect()
}
