use std::{collections::HashSet, sync::Arc};

use graphrs::{
    Edge, EdgeDedupeStrategy, Graph, GraphSpecs, MissingNodeStrategy, SelfLoopsFalseStrategy,
    algorithms::community::leiden::{QualityFunction, leiden},
};
use nalgebra::{DMatrix, DVector, SVector};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use uuid::Uuid;

use crate::{AnalyzerConfig, error::NisabaError};

/// The `deterministic_projection` function performs linear projection of a Matrix from one size to another
///
/// Arguments:
///
/// * `input`: The `builder` parameter is a mutable reference of Box of type implementing `ArrayBuilder`, where
///   values are to be appended to.
/// * `d_out`: The `d_out` parameter is of type `usize`, givign how big the resultant vector should be.
/// * `seed`: The `index` parameter is of type `u64`, to set the random number generator deterministic and reproducible.
///
/// Returns:
///
/// The `deterministic_projection` function returns a DVector of d_out size.
pub fn deterministic_projection<const C: usize>(
    input: SVector<f32, C>,
    d_out: usize,
    seed: u64,
) -> DVector<f32> {
    let d_in = input.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Deterministic row-major fill
    let data: Vec<f32> = (0..d_in * d_out).map(|_| normal.sample(&mut rng)).collect();

    // Row-major constructor: d_in rows, d_out columns
    let projection_matrix = DMatrix::from_vec(d_in, d_out, data);

    let y = projection_matrix.transpose() * input;

    let norm = y.norm();

    if norm == 0.0 { y.clone() } else { y / norm }
}

pub struct GraphClusterer {
    graph: Graph<Uuid, ()>,
}

impl GraphClusterer {
    /// The `new` function creates an instance of GraphClusterer with desired settings
    /// for purposes of clustering.
    pub fn new() -> Self {
        let graph = Graph::new(GraphSpecs {
            directed: false,
            edge_dedupe_strategy: EdgeDedupeStrategy::KeepLast,
            missing_node_strategy: MissingNodeStrategy::Create,
            multi_edges: false,
            self_loops: false,
            self_loops_false_strategy: SelfLoopsFalseStrategy::Error,
        });

        Self { graph }
    }

    /// The `add_ann_edges` function adds weighted undirected edges to the graph on the mutable self. The cosine distance from the latent store
    /// is adjusted to cosine similarity for purposes of comparison.
    ///
    /// Arguments:
    ///
    /// * `config`: The `config` parameter is a shared AnalyzerConfig which provides the similarity threshold.
    /// * `source`: The `source` parameter is of type implementing `Storable`, giving the source node details.
    /// * `candidates`: The `candidates` parameter is a slice of types implementing `MatchCandidate` trait, giving the destinatination node details.
    ///
    /// Returns:
    ///
    /// The `add_ann_edges` function returns a Result of unit value when successful and NisabaError on error.
    pub fn add_ann_edges(
        &mut self,
        config: Arc<AnalyzerConfig>,
        source: Uuid,
        candidates: &[(Uuid, f32)],
    ) -> Result<(), NisabaError> {
        for (id, conf) in candidates {
            let adj_score = 1.0 - conf;
            if adj_score >= config.similarity.threshold {
                if let Ok(existing) = self.graph.get_edge(source, *id)
                    && adj_score > existing.weight as f32
                {
                    self.graph
                        .add_edge(Edge::with_weight(source, *id, adj_score.into()))
                        .map_err(NisabaError::Graph)?;
                } else {
                    self.graph
                        .add_edge(Edge::with_weight(source, *id, adj_score.into()))
                        .map_err(NisabaError::Graph)?;
                }
            }
        }

        Ok(())
    }

    /// The `clusters` function runs the leiden community algorithm on the graph to give HashSet
    /// clusters of ids and map the ids to TableDefs/FieldDefs.
    ///
    /// Arguments:
    ///
    /// * `defs`: The `defs` parameter is of type implementing ClusterDef where the Id is Uuid.
    /// * `build_cluster`: The `build_cluster` parameter is of type implementing `Fn`, a function
    ///   that runs to group FieldDefs/TableDefs.
    ///
    /// Returns:
    ///
    /// The `clusters` function returns a Result of Vec of C determined by build_cluster function
    /// when successful and NisabaError on error.
    pub fn clusters(&self) -> Result<Vec<HashSet<Uuid>>, NisabaError> {
        // Community clusters
        let clusters = leiden(&self.graph, true, QualityFunction::CPM, None, None, None)
            .map_err(NisabaError::Graph)?;

        Ok(clusters)
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::Vector1;
    use std::{sync::Arc, thread};
    use uuid::Uuid;

    use crate::{
        AnalyzerConfig,
        analyzer::calculation::{GraphClusterer, deterministic_projection},
    };

    fn create_test_graph() -> GraphClusterer {
        let all_ids = [
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
        ];

        let sources = [
            all_ids[0], all_ids[0], all_ids[2], all_ids[3], all_ids[4], all_ids[2],
        ];
        let targets = [
            all_ids[1], all_ids[2], all_ids[5], all_ids[4], all_ids[5], all_ids[1],
        ];
        let weights = [0.02_f32, 0.1, 0.1, 0.7, 0.3, 0.4, 0.8];

        let config = Arc::new(AnalyzerConfig::default());

        let mut clusterer = GraphClusterer::new();

        for ((source, candidate), weight) in sources.into_iter().zip(targets).zip(weights) {
            clusterer
                .add_ann_edges(config.clone(), source, &[(candidate, weight)])
                .unwrap();
        }

        clusterer
    }

    #[test]
    fn test_deterministic_projection() {
        let input = Vector1::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let result1 = deterministic_projection(input, 384, 42);

        let result2 = thread::spawn(move || deterministic_projection(input, 384, 42))
            .join()
            .expect("Thread paniced");

        assert_eq!(result1.len(), result2.len());

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_graph_clustering() {
        let graph = create_test_graph();

        let result = graph.clusters().unwrap();

        dbg!(&result);

        assert!(!result.is_empty());
    }
}
