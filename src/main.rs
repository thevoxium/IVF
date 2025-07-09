use candle_core::{Device, Result as CandleResult, Tensor};
use ndarray::{Array2, ArrayView1};
use ndarray_npy::read_npy;
use ordered_float::NotNan;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

struct Ivf {
    is_trained: bool,
    centroids: Option<Tensor>,
    centroids_array: Option<Array2<f32>>,
    data_array: Option<Array2<f32>>,
    clusters: Vec<Vec<usize>>,
    data: Tensor,
    device: Device,
}

impl Ivf {
    fn new(device: Device) -> Self {
        Ivf {
            is_trained: false,
            centroids: None,
            centroids_array: None,
            data_array: None,
            clusters: Vec::new(),
            data: Tensor::zeros((0, 0), candle_core::DType::F32, &device).unwrap(),
            device,
        }
    }

    fn build(
        &mut self,
        vectors: &Array2<f32>,
        num_clusters: usize,
        sample_size: usize,
        max_iterations: usize,
    ) -> anyhow::Result<()> {
        let data_vec: Vec<f32> = vectors.iter().cloned().collect();
        self.data = Tensor::from_vec(data_vec, (vectors.nrows(), vectors.ncols()), &self.device)?;

        let sample_data = if vectors.nrows() <= sample_size {
            self.data.clone()
        } else {
            self.sample_vectors(&self.data, sample_size)?
        };
        self.centroids = Some(self.train_kmeans(&sample_data, num_clusters, max_iterations)?);

        let assignments = self.assign_vectors_to_clusters_batched(&self.data)?;

        self.clusters = vec![Vec::new(); num_clusters];

        for (vector_idx, cluster_id) in assignments.iter().enumerate() {
            self.clusters[*cluster_id].push(vector_idx);
        }

        self.cache_arrays_for_search()?;

        self.is_trained = true;

        Ok(())
    }

    fn cache_arrays_for_search(&mut self) -> anyhow::Result<()> {
        let (rows, cols) = self.data.dims2()?;
        let data_vec = self.data.to_vec2::<f32>()?;
        let flat: Vec<f32> = data_vec.into_iter().flatten().collect();
        self.data_array = Some(Array2::from_shape_vec((rows, cols), flat)?);

        if let Some(ref centroids) = self.centroids {
            let centroids_vec = centroids.to_vec2::<f32>()?;
            let centroids_flat: Vec<f32> = centroids_vec.into_iter().flatten().collect();
            let (num_centroids, dim) = centroids.dims2()?;
            self.centroids_array = Some(Array2::from_shape_vec(
                (num_centroids, dim),
                centroids_flat,
            )?);
        }
        Ok(())
    }

    fn train_kmeans(
        &self,
        data: &Tensor,
        num_clusters: usize,
        max_iterations: usize,
    ) -> anyhow::Result<Tensor> {
        let (_num_points, _dim) = data.dims2()?;

        let mut centroids = self.initialize_centroids_fast(data, num_clusters)?;

        for iteration in 0..max_iterations {
            let assignments = self.assign_points_to_centroids_fast(data, &centroids)?;
            let new_centroids = self.update_centroids_fast(data, &assignments, num_clusters)?;
            let diff = (&new_centroids - &centroids)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?;
            centroids = new_centroids;

            println!("K-means iteration {}: change = {:.6}", iteration, diff);

            if diff < 1e-6 {
                break;
            }
        }

        Ok(centroids)
    }

    fn initialize_centroids_fast(
        &self,
        data: &Tensor,
        num_clusters: usize,
    ) -> anyhow::Result<Tensor> {
        let (num_points, dim) = data.dims2()?;
        let mut rng = thread_rng();

        let mut indices: Vec<usize> = (0..num_points).collect();
        indices.shuffle(&mut rng);
        indices.truncate(num_clusters);

        let mut centroids_data = Vec::new();
        for &idx in &indices {
            let centroid = data.get(idx)?.to_vec1::<f32>()?;
            centroids_data.extend(centroid);
        }

        Ok(Tensor::from_vec(
            centroids_data,
            (num_clusters, dim),
            &self.device,
        )?)
    }

    fn assign_points_to_centroids_fast(
        &self,
        data: &Tensor,
        centroids: &Tensor,
    ) -> anyhow::Result<Vec<usize>> {
        let centroids_t = centroids.t()?;
        let dot_products = data.matmul(&centroids_t)?;
        let distances = (1.0 - dot_products)?;

        let assignments = distances.argmin_keepdim(1)?;
        let assignments_vec = assignments.squeeze(1)?.to_vec1::<u32>()?;

        Ok(assignments_vec.into_iter().map(|x| x as usize).collect())
    }

    fn update_centroids_fast(
        &self,
        data: &Tensor,
        assignments: &[usize],
        num_clusters: usize,
    ) -> anyhow::Result<Tensor> {
        let (_, dim) = data.dims2()?;
        let mut new_centroids_data = Vec::with_capacity(num_clusters * dim);

        for cluster_id in 0..num_clusters {
            let cluster_indices: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(idx, _)| idx)
                .collect();

            if cluster_indices.is_empty() {
                if let Some(ref centroids) = self.centroids {
                    let old_centroid = centroids.get(cluster_id)?.to_vec1::<f32>()?;
                    new_centroids_data.extend(old_centroid);
                } else {
                    new_centroids_data.extend(vec![0.0; dim]);
                }
            } else {
                let first_point = data.get(cluster_indices[0])?.to_vec1::<f32>()?;
                let mut sum_vec = first_point;

                for &idx in &cluster_indices[1..] {
                    let point = data.get(idx)?.to_vec1::<f32>()?;
                    for (s, p) in sum_vec.iter_mut().zip(point.iter()) {
                        *s += p;
                    }
                }

                let count = cluster_indices.len() as f32;
                let mut mean_vec: Vec<f32> = sum_vec.into_iter().map(|x| x / count).collect();

                let norm: f32 = mean_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut mean_vec {
                        *x /= norm;
                    }
                }

                new_centroids_data.extend(mean_vec);
            }
        }

        Ok(Tensor::from_vec(
            new_centroids_data,
            (num_clusters, dim),
            &self.device,
        )?)
    }

    fn sample_vectors(&self, data: &Tensor, sample_size: usize) -> anyhow::Result<Tensor> {
        let (num_points, dim) = data.dims2()?;
        let mut rng = thread_rng();

        let mut indices: Vec<usize> = (0..num_points).collect();
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);

        let mut sampled = Vec::with_capacity(sample_size * dim);
        for &idx in &indices {
            let row = data.get(idx)?.to_vec1::<f32>()?;
            sampled.extend(row);
        }

        Ok(Tensor::from_vec(sampled, (sample_size, dim), &self.device)?)
    }

    fn assign_vectors_to_clusters_batched(&self, data: &Tensor) -> anyhow::Result<Vec<usize>> {
        let centroids = self.centroids.as_ref().unwrap();
        let (_num_vectors, _) = data.dims2()?;

        let centroids_t = centroids.t()?;
        let dot_products = data.matmul(&centroids_t)?;
        let distances = (1.0 - dot_products)?;

        let assignments = distances.argmin_keepdim(1)?;
        let assignments_vec = assignments.squeeze(1)?.to_vec1::<u32>()?;

        Ok(assignments_vec.into_iter().map(|x| x as usize).collect())
    }

    fn search(
        &self,
        query: &ArrayView1<f32>,
        top_k: usize,
        num_probes: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if !self.is_trained {
            return Err(anyhow::anyhow!("Index not trained yet"));
        }

        let centroids_array = self.centroids_array.as_ref().unwrap();
        let data_array = self.data_array.as_ref().unwrap();

        let mut centroid_distances: Vec<(usize, f32)> = centroids_array
            .outer_iter()
            .enumerate()
            .map(|(cluster_id, centroid_row)| {
                let distance = 1.0 - query.dot(&centroid_row);
                (cluster_id, distance)
            })
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let selected_clusters: Vec<usize> = centroid_distances
            .into_iter()
            .take(num_probes)
            .map(|(idx, _)| idx)
            .collect();

        let mut heap: BinaryHeap<Reverse<(NotNan<f32>, usize)>> = BinaryHeap::new();

        for cluster_id in selected_clusters {
            if let Some(point_indices) = self.clusters.get(cluster_id) {
                for &point_idx in point_indices {
                    let vector_row = data_array.row(point_idx);
                    let distance = 1.0 - query.dot(&vector_row);
                    let distance_ordered = NotNan::new(distance).unwrap();

                    if heap.len() < top_k {
                        heap.push(Reverse((distance_ordered, point_idx)));
                    } else if let Some(Reverse((worst_distance, _))) = heap.peek() {
                        if distance_ordered < *worst_distance {
                            heap.pop();
                            heap.push(Reverse((distance_ordered, point_idx)));
                        }
                    }
                }
            }
        }

        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|Reverse((distance, doc_id))| (doc_id, distance.into_inner()))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }
}

fn normalize_dataset(data: &mut Array2<f32>) {
    data.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let norm = (row.dot(&row)).sqrt();
            if norm > 0.0 {
                row.mapv_inplace(|x| x / norm);
            }
        });
}

fn load_npy_file(path: &str) -> anyhow::Result<Array2<f32>> {
    let array: Array2<f32> = read_npy(path)?;
    println!("Loaded array with shape: {:?}", array.shape(),);
    Ok(array)
}

fn main() -> anyhow::Result<()> {
    let total_start = Instant::now();

    let device = Device::Cpu;

    let embeddings_full = load_npy_file("index_ivf/embeddings_chunk_0.npy")?;
    let mut embeddings = embeddings_full;

    let mut queries = load_npy_file("index_ivf/query_file.npy")?;
    let num_queries = queries.nrows();

    normalize_dataset(&mut embeddings);
    normalize_dataset(&mut queries);

    let num_centroids = 1000;
    let num_probes = 5;
    let sample_size = 50000;
    let max_iterations = 25;

    let mut ivf = Ivf::new(device);
    ivf.build(&embeddings, num_centroids, sample_size, max_iterations)?;

    let top_k = 5;
    let mut total_query_time = 0.0;
    let mut successful_queries = 0;

    for i in 0..num_queries.min(100) {
        let query = queries.row(i);
        let query_start = Instant::now();

        match ivf.search(&query, top_k, num_probes) {
            Ok(results) => {
                let query_time = query_start.elapsed();
                total_query_time += query_time.as_secs_f64();
                successful_queries += 1;

                if i < 5 {
                    println!(
                        "Query {}: found {} results in {:.4} ms",
                        i,
                        results.len(),
                        query_time.as_secs_f64() * 1000.0
                    );

                    for (rank, (doc_id, dist)) in results.iter().take(5).enumerate() {
                        let similarity = 1.0 - dist;
                        println!(
                            "  {}. Document {}: similarity = {:.6}",
                            rank + 1,
                            doc_id,
                            similarity
                        );
                    }
                }
            }
            Err(e) => {
                println!("Query {} failed: {}", i, e);
            }
        }

        let current_avg = (total_query_time / (i + 1) as f64) * 1000.0;
        println!(
            "Processed {}/{} queries... Current avg: {:.2}ms",
            i + 1,
            num_queries.min(100),
            current_avg
        );
    }

    if successful_queries > 0 {
        let avg_latency_ms = (total_query_time / successful_queries as f64) * 1000.0;
        println!(
            "Total runtime: {:.2} seconds",
            total_start.elapsed().as_secs_f64()
        );
        println!("Total queries processed: {}", successful_queries);
        println!("Average latency per query: {:.4} ms", avg_latency_ms);
        println!(
            "Queries per second: {:.2}",
            successful_queries as f64 / total_query_time
        );
    }

    Ok(())
}
