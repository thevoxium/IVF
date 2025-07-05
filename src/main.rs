use kentro::KMeans;
use ndarray::{Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;

struct Ivf {
    is_trained: bool,
    kmeans: KMeans,
    clusters: HashMap<usize, Vec<usize>>,
    data: Array2<f32>,
}

impl Ivf {
    fn new(num_clusters: usize, num_iterations: usize) -> Self {
        let kmeans = KMeans::new(num_clusters)
            .with_euclidean(true)
            .with_iterations(num_iterations);

        Ivf {
            kmeans,
            clusters: HashMap::new(),
            data: Array2::zeros((0, 0)),
            is_trained: false,
        }
    }

    fn build(&mut self, vectors: &Array2<f32>) -> Result<(), String> {
        println!("Building IVF index with {} vectors...", vectors.nrows());
        self.data = vectors.clone();
        let cluster_assignments = self
            .kmeans
            .train(self.data.view(), None)
            .map_err(|e| format!("K-means training failed: {}", e))?;

        for (cluster_id, vector_ids) in cluster_assignments.iter().enumerate() {
            self.clusters.insert(cluster_id, vector_ids.clone());
        }

        self.is_trained = true;
        println!("Index built with {} clusters", self.clusters.len());
        Ok(())
    }

    fn search(
        &self,
        query: &ArrayView1<f32>,
        top_k: usize,
        num_probes: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        if !self.is_trained {
            return Err("Index not trained yet".to_string());
        }

        let query_2d = Array2::from_shape_vec((1, query.len()), query.to_vec())
            .map_err(|e| format!("Failed to reshape query: {}", e))?;

        let nearest_clusters = self
            .kmeans
            .assign(query_2d.view(), num_probes)
            .map_err(|e| format!("Cluster assignment failed: {}", e))?;

        let cluster_ids = &nearest_clusters[0];

        let mut candidates = Vec::new();

        for &cluster_id in cluster_ids {
            if let Some(vector_ids) = self.clusters.get(&cluster_id) {
                for &vector_id in vector_ids {
                    let vector = self.data.row(vector_id);
                    let distance = distance(query, &vector);
                    candidates.push((vector_id, distance));
                }
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates.into_iter().take(top_k).collect())
    }
}

fn distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn generate_random_vectors(count: usize, dim: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..count * dim)
        .map(|_| rng.gen_range(-30.0..=30.0))
        .collect();

    Array2::from_shape_vec((count, dim), data).unwrap()
}

fn main() -> Result<(), String> {
    println!("Simple IVF with Kentro Demo");

    let vectors = generate_random_vectors(100000, 128);
    println!(
        "Generated {} vectors with {} dimensions",
        vectors.nrows(),
        vectors.ncols()
    );

    let mut ivf = Ivf::new(10, 50);
    ivf.build(&vectors)?;

    let query_data = generate_random_vectors(1, 128);
    let query = query_data.row(0);

    println!("\nSearching for top 5 similar vectors...");
    let results = ivf.search(&query, 5, 10)?;
    println!("Results (vector_id, distance):");
    for (vector_id, distance) in results {
        println!("  Vector {}: distance = {:.4}", vector_id, distance);
    }

    Ok(())
}
