use std::sync::Arc;

use graphrs::{
    Edge, EdgeDedupeStrategy, Graph, GraphSpecs, MissingNodeStrategy, SelfLoopsFalseStrategy,
    algorithms::community::leiden::{QualityFunction, leiden},
};
use nalgebra::{DMatrix, DVector, SVector};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use uuid::Uuid;

use crate::{
    AnalyzerConfig,
    analyzer::{
        report::{ClusterDef, ClusterItem},
        retriever::Storable,
    },
    error::NisabaError,
    types::MatchCandidate,
};

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

    pub fn add_ann_edges<T, C>(
        &mut self,
        config: Arc<AnalyzerConfig>,
        source: &T,
        candidates: &[C],
    ) -> Result<(), NisabaError>
    where
        T: Storable,
        T::SearchResult: MatchCandidate<Body = T>,
        C: MatchCandidate<Body = T>,
    {
        for c in candidates {
            let adj_score = 1.0 - c.confidence();
            if adj_score >= config.similarity_threshold {
                if let Ok(existing) = self.graph.get_edge(source.get_id(), c.body().get_id())
                    && adj_score > existing.weight as f32
                {
                    self.graph
                        .add_edge(Edge::with_weight(
                            source.get_id(),
                            c.body().get_id(),
                            adj_score.into(),
                        ))
                        .map_err(NisabaError::Graph)?;
                } else {
                    self.graph
                        .add_edge(Edge::with_weight(
                            source.get_id(),
                            c.body().get_id(),
                            adj_score.into(),
                        ))
                        .map_err(NisabaError::Graph)?;
                }
            }
        }

        Ok(())
    }

    pub fn clusters<D, R, C>(
        &self,
        defs: &[D],
        build_cluster: impl Fn(u32, Vec<R>) -> C,
    ) -> Result<Vec<C>, NisabaError>
    where
        D: ClusterDef<Id = Uuid>,
        R: ClusterItem<Def = D>,
    {
        // Community clusters
        let clusters = leiden(&self.graph, true, QualityFunction::CPM, None, None, None)
            .map_err(NisabaError::Graph)?;

        let communities: Vec<C> = clusters
            .into_iter()
            .enumerate()
            .map(|(cid, vals)| {
                let items = defs
                    .iter()
                    .filter(|d| vals.contains(&d.id()))
                    .cloned()
                    .collect::<Vec<D>>();

                let items = items
                    .into_iter()
                    .map(|it| R::from_def(it.id(), &it))
                    .collect::<Vec<R>>();

                build_cluster(cid as u32, items)
            })
            .collect();

        Ok(communities)
    }
}

#[cfg(test)]
mod tests {
    // use arrow::{
    //     array::{FixedSizeBinaryArray, Float32Array, RecordBatch},
    //     datatypes::{DataType, Field, Schema},
    // };
    use nalgebra::Vector1;
    use std::{
        // collections::{HashMap, HashSet},
        // sync::Arc,
        thread,
    };
    // use uuid::Uuid;

    use crate::analyzer::calculation::{
        // ArrowGraph,
        deterministic_projection,
        // leiden_communities
    };

    // fn create_test_graph(all_ids: &[Uuid; 6]) -> ArrowGraph {
    //     let schema = Arc::new(Schema::new(vec![
    //         Field::new("source", DataType::FixedSizeBinary(16), false),
    //         Field::new("target", DataType::FixedSizeBinary(16), false),
    //         Field::new("weight", DataType::Float32, true),
    //     ]));

    //     let sources = [
    //         all_ids[0], all_ids[1], all_ids[0], all_ids[2], all_ids[3], all_ids[4], all_ids[2],
    //     ];
    //     let targets = [
    //         all_ids[1], all_ids[0], all_ids[2], all_ids[5], all_ids[4], all_ids[5], all_ids[1],
    //     ];
    //     let weights = [1.0_f32, 0.9, 0.9, 0.3, 0.7, 0.6, 0.2];

    //     let edges = RecordBatch::try_new(
    //         schema,
    //         vec![
    //             Arc::new(
    //                 FixedSizeBinaryArray::try_from_iter(sources.iter().map(|v| v.into_bytes()))
    //                     .expect("Failed to create `source` fixedsizebinaryarray"),
    //             ),
    //             Arc::new(
    //                 FixedSizeBinaryArray::try_from_iter(targets.iter().map(|v| v.into_bytes()))
    //                     .expect("Failed to create `target` fixedsizebinaryarray"),
    //             ),
    //             Arc::new(Float32Array::from(
    //                 weights.into_iter().collect::<Vec<f32>>(),
    //             )),
    //         ],
    //     )
    //     .expect("Failed to create record batch");

    //     ArrowGraph::from_edges(edges).expect("Failed to create ArrowGraph instance")
    // }

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
}
