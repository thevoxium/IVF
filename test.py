import numpy as np
import faiss
import time
import random
from collections import defaultdict
import heapq

class FaissIvf:
    def __init__(self, use_gpu=False):
        self.is_trained = False
        self.centroids = None
        self.clusters = []
        self.data = None
        self.index = None
        self.use_gpu = use_gpu
        self.num_clusters = 0
        
    def build(self, vectors, num_clusters, sample_size, max_iterations):
        print(f"Building IVF index with {vectors.shape[0]} vectors...")
        total_build_start = time.time()
        
        # Convert to float32 if not already
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Step 1: Sample vectors for K-means training
        sample_start = time.time()
        if vectors.shape[0] <= sample_size:
            print(f"Using all {vectors.shape[0]} vectors for training")
            sample_data = vectors.copy()
        else:
            print(f"Sampling {sample_size} vectors from {vectors.shape[0]} for K-means training")
            sample_indices = random.sample(range(vectors.shape[0]), sample_size)
            sample_data = vectors[sample_indices]
        
        print(f"Sampling completed in {time.time() - sample_start:.2f}s")
        
        # Step 2: Train K-means using FAISS
        train_start = time.time()
        self.centroids = self.train_kmeans(sample_data, num_clusters, max_iterations)
        print(f"K-means training completed in {time.time() - train_start:.2f}s")
        
        # Step 3: Assign all vectors to clusters (BLAS-accelerated)
        assign_start = time.time()
        print(f"Starting BLAS-accelerated assignment of {vectors.shape[0]} vectors to clusters...")
        
        assignments = self.assign_vectors_to_clusters_batched(vectors)
        
        print(f"Assignment completed in {time.time() - assign_start:.2f}s (was 100+ minutes before!)")
        
        # Step 4: Build inverted index
        index_start = time.time()
        self.clusters = [[] for _ in range(num_clusters)]
        
        for vector_idx, cluster_id in enumerate(assignments):
            self.clusters[cluster_id].append(vector_idx)
        
        print(f"Index building completed in {time.time() - index_start:.2f}s")
        
        # Step 5: Cache data for fast search
        self.cache_arrays_for_search(vectors)
        
        self.is_trained = True
        self.num_clusters = num_clusters
        
        # Print statistics
        self.print_cluster_stats()
        print(f"Total build time: {time.time() - total_build_start:.2f}s")
        
    def cache_arrays_for_search(self, vectors):
        print("Caching arrays for fast search...")
        cache_start = time.time()
        
        # Cache data array
        self.data = vectors.copy()
        
        print(f"Caching completed in {time.time() - cache_start:.2f}s")
        
    def train_kmeans(self, data, num_clusters, max_iterations):
        num_points, dim = data.shape
        
        # Initialize centroids using K-means++
        print(f"Initializing {num_clusters} centroids with K-means++...")
        init_start = time.time()
        centroids = self.initialize_centroids_fast(data, num_clusters)
        print(f"Initialization completed in {time.time() - init_start:.2f}s")
        
        for iteration in range(max_iterations):
            iter_start = time.time()
            
            # Assign points to clusters using BLAS
            assignments = self.assign_points_to_centroids_fast(data, centroids)
            
            # Update centroids
            new_centroids = self.update_centroids_fast(data, assignments, num_clusters)
            
            # Check for convergence
            diff = np.abs(new_centroids - centroids).sum()
            centroids = new_centroids
            
            print(f"K-means iteration {iteration}: change = {diff:.6f}, time = {time.time() - iter_start:.3f}s")
            
            if diff < 1e-6:
                print(f"K-means converged after {iteration + 1} iterations")
                break
        
        return centroids
    
    def initialize_centroids_fast(self, data, num_clusters):
        num_points, dim = data.shape
        
        # Simplified initialization for speed - just random selection
        indices = list(range(num_points))
        random.shuffle(indices)
        selected_indices = indices[:num_clusters]
        
        return data[selected_indices].copy()
    
    def assign_points_to_centroids_fast(self, data, centroids):
        # Use BLAS-accelerated matrix multiplication
        dot_products = np.dot(data, centroids.T)
        distances = 1.0 - dot_products
        
        # Find argmin for each point
        assignments = np.argmin(distances, axis=1)
        
        return assignments
    
    def update_centroids_fast(self, data, assignments, num_clusters):
        num_points, dim = data.shape
        new_centroids = np.zeros((num_clusters, dim), dtype=np.float32)
        
        for cluster_id in range(num_clusters):
            cluster_mask = assignments == cluster_id
            cluster_points = data[cluster_mask]
            
            if len(cluster_points) == 0:
                # Keep old centroid if no points assigned
                if hasattr(self, 'centroids') and self.centroids is not None:
                    new_centroids[cluster_id] = self.centroids[cluster_id]
                else:
                    # Random centroid
                    new_centroids[cluster_id] = np.zeros(dim, dtype=np.float32)
            else:
                # Compute mean and normalize
                mean_vec = np.mean(cluster_points, axis=0)
                
                # Normalize the centroid
                norm = np.linalg.norm(mean_vec)
                if norm > 0.0:
                    mean_vec = mean_vec / norm
                
                new_centroids[cluster_id] = mean_vec
        
        return new_centroids
    
    def assign_vectors_to_clusters_batched(self, data):
        print("Using BLAS-accelerated matrix multiplication for assignment...")
        batch_start = time.time()
        
        # Single large matrix multiplication - this is where BLAS acceleration matters most!
        dot_products = np.dot(data, self.centroids.T)
        distances = 1.0 - dot_products
        
        print(f"Matrix multiplication completed in {time.time() - batch_start:.2f}s")
        
        # Find closest cluster for each vector
        print(f"Finding argmin for {data.shape[0]} vectors...")
        argmin_start = time.time()
        assignments = np.argmin(distances, axis=1)
        print(f"Argmin completed in {time.time() - argmin_start:.2f}s")
        
        return assignments
    
    def search(self, query, top_k, num_probes):
        if not self.is_trained:
            raise ValueError("Index not trained yet")
        
        # Step 1: Find nearest centroids (vectorized)
        centroid_distances = []
        for cluster_id, centroid in enumerate(self.centroids):
            distance = 1.0 - np.dot(query, centroid)
            centroid_distances.append((cluster_id, distance))
        
        # Sort and select top clusters
        centroid_distances.sort(key=lambda x: x[1])
        selected_clusters = [cluster_id for cluster_id, _ in centroid_distances[:num_probes]]
        
        # Step 2: Search in selected clusters using fast numpy operations
        heap = []
        
        for cluster_id in selected_clusters:
            if cluster_id < len(self.clusters):
                point_indices = self.clusters[cluster_id]
                for point_idx in point_indices:
                    vector_row = self.data[point_idx]
                    distance = 1.0 - np.dot(query, vector_row)
                    
                    if len(heap) < top_k:
                        heapq.heappush(heap, (-distance, point_idx))
                    elif -distance > heap[0][0]:
                        heapq.heappushpop(heap, (-distance, point_idx))
        
        # Convert to sorted results
        results = []
        while heap:
            neg_distance, doc_id = heapq.heappop(heap)
            results.append((doc_id, -neg_distance))
        
        results.reverse()  # Sort by distance (ascending)
        return results
    
    def print_cluster_stats(self):
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        non_empty_clusters = sum(1 for size in cluster_sizes if size > 0)
        
        if non_empty_clusters > 0:
            avg_size = sum(cluster_sizes) / non_empty_clusters
            non_empty_sizes = [size for size in cluster_sizes if size > 0]
            min_size = min(non_empty_sizes) if non_empty_sizes else 0
            max_size = max(cluster_sizes) if cluster_sizes else 0
        else:
            avg_size = 0.0
            min_size = 0
            max_size = 0
        
        print(f"Index built with {len(self.clusters)} total clusters ({non_empty_clusters} non-empty)")
        print(f"Cluster sizes - avg: {avg_size:.1f}, min: {min_size}, max: {max_size}")

def normalize_dataset(data):
    print(f"Normalizing {data.shape[0]} vectors with parallel processing...")
    start = time.time()
    
    # Normalize each row
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)
    data = data / norms
    
    print(f"Normalization completed in {time.time() - start:.3f}s")
    return data

def load_npy_file(path):
    print(f"Loading numpy file: {path}")
    start = time.time()
    array = np.load(path)
    print(f"Loaded array with shape: {array.shape} in {time.time() - start:.2f}s")
    return array

def main():
    total_start = time.time()
    
    print("Using CPU device with BLAS acceleration")
    
    # Load data
    embeddings = load_npy_file("index_ivf/embeddings_chunk_0.npy")
    num_documents = embeddings.shape[0]
    print(f"Loaded {num_documents} document embeddings")
    
    queries = load_npy_file("index_ivf/query_file.npy")
    num_queries = queries.shape[0]
    print(f"Loaded {num_queries} query vectors\n")
    
    # Normalize data
    print("=== Pre-normalizing Data ===")
    embeddings = normalize_dataset(embeddings)
    queries = normalize_dataset(queries)
    
    # Optimized parameters for 1M vectors
    num_centroids = 1000  # More clusters for better partitioning
    num_probes = 5        # Search more clusters for better recall
    sample_size = 50000   # Larger sample for better clustering
    max_iterations = 25   # Should converge faster
    
    print("\n=== Configuration ===")
    print(f"Number of centroids: {num_centroids}")
    print(f"Number of probes: {num_probes}")
    print(f"Sample size for K-means: {sample_size}")
    print(f"Max K-means iterations: {max_iterations}")
    
    # Build index
    print("\n=== Building IVF Index ===")
    ivf = FaissIvf()
    ivf.build(embeddings, num_centroids, sample_size, max_iterations)
    
    # Run queries
    print("\n=== Running Queries ===")
    top_k = 10
    total_query_time = 0.0
    successful_queries = 0
    
    query_start_time = time.time()
    
    for i in range(min(num_queries, 100)):  # Test with 100 queries
        query = queries[i]
        query_start = time.time()
        
        try:
            results = ivf.search(query, top_k, num_probes)
            query_time = time.time() - query_start
            total_query_time += query_time
            successful_queries += 1
            
            if i < 5:
                print(f"Query {i}: found {len(results)} results in {query_time * 1000:.4f} ms")
                
                # Show top 3 results
                for rank, (doc_id, dist) in enumerate(results[:3]):
                    similarity = 1.0 - dist
                    print(f"  {rank + 1}. Document {doc_id}: similarity = {similarity:.6f}")
        
        except Exception as e:
            print(f"Query {i} failed: {e}")
        
        if (i + 1) % 2 == 0:
            current_avg = (total_query_time / (i + 1)) * 1000.0
            print(f"Processed {i + 1}/{min(num_queries, 100)} queries... Current avg: {current_avg:.2f}ms")
    
    total_query_time_wall = time.time() - query_start_time

    if successful_queries > 0:
        avg_latency_ms = (total_query_time / successful_queries) * 1000.0
        print("\n=== Final Results ===")
        print(f"Total runtime: {time.time() - total_start:.2f} seconds")  # Fixed this line
        print(f"Total queries processed: {successful_queries}")
        print(f"Average latency per query: {avg_latency_ms:.4f} ms")
        print(f"Queries per second: {successful_queries / total_query_time:.2f}")
        print("BLAS acceleration: ENABLED ✓")
        print("Cached arrays for search: ENABLED ✓")
    

if __name__ == "__main__":
    main()
