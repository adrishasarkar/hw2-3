#include "common.h"
#include "assert.h"
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

using namespace std;

#define NUM_THREADS 128

double bin_size; // bin_size = 2 * cutoff
int bin_Dim;  // number of stacked bins in each direction.
int num_bins; // = bin_Dim * bin_Dim
int* d_part_ids_by_bin;
int* d_bin_ids_prefix_sum;

// Put any static global variables here that you will use throughout the simulation.
int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //  very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    atomicAdd(&particle.ax, coef * dx);
    atomicAdd(&particle.ay, coef * dy);
}

/*
 * CUDA kernel that initializes acceleration components (ax, ay) of all particles to zero
 * This must be called before computing forces in each time step
 * 
 * Parameters:
 * particles: Array of particles in device memory
 * num_parts: Total number of particles
 * 
 * Note: No atomic operations needed here because:
 * 1. Each thread writes to its own unique particle
 * 2. No thread reads or writes to another thread's particle
 * 3. Each acceleration component is written only once
 */
__global__ void reset_forces_gpu(particle_t* particles, int num_parts) {
    // Calculate unique thread ID from block and thread indices
    // Each thread will handle resetting one particle's forces
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Return if this thread's ID exceeds the number of particles
    // (This happens because we round up the number of blocks)
    if (tid >= num_parts)
        return;

    // Reset both acceleration components to zero
    // This particle will accumulate forces from interactions in compute_forces_gpu
    particles[tid].ax = particles[tid].ay = 0;
}

// Returns the bin_id given the particle - GPU version
__device__ int get_bin_id_for_particle_gpu(particle_t* part, int bin_Dim, double bin_size){
    int bin_x = (int)(part -> x / bin_size);
    int bin_y = (int)(part -> y / bin_size);
    return bin_x + bin_y * bin_Dim;
}

__global__ void compute_forces_gpu(particle_t* d_parts, int num_parts, int bin_Dim, int* d_part_ids_by_bin, int* d_bin_ids_prefix_sum, double bin_size) {
    // Get thread (particle) ID based on global thread indexing
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    // Declare shared memory for particles in the current block
    extern __shared__ particle_t shared_parts[];
    
    // Get the actual particle ID from the binned array
    int part1_id = d_part_ids_by_bin[tid];
    particle_t& my_particle = d_parts[part1_id];
    
    // Calculate bin coordinates directly to match your existing approach
    int bin_x_base = (int)(my_particle.x / bin_size);
    int bin_y_base = (int)(my_particle.y / bin_size);
    
    // Local accumulators to reduce atomic operations
    double ax_local = 0.0;
    double ay_local = 0.0;
    
    // Examine all neighboring bins (including the particle's own bin)
    for(int bin_dy = -1; bin_dy <= 1; bin_dy++){
        for(int bin_dx = -1; bin_dx <= 1; bin_dx++){
            int bin_x = bin_x_base + bin_dx;
            int bin_y = bin_y_base + bin_dy;
            
            // Skip bins that are out of bounds
            if(bin_x < 0 || bin_x >= bin_Dim || bin_y < 0 || bin_y >= bin_Dim)
                continue;
            
            int adj_bin_id = bin_x + bin_y * bin_Dim;
            
            // Get range of particles in this bin using the prefix sum
            int bin_start = d_bin_ids_prefix_sum[adj_bin_id];
            int bin_end = d_bin_ids_prefix_sum[adj_bin_id+1];
            int n_parts_in_adj_bin = bin_end - bin_start;
            
            // Skip empty bins
            if (n_parts_in_adj_bin == 0)
                continue;
            
            // Process particles in chunks that fit in shared memory
            for (int chunk_start = 0; chunk_start < n_parts_in_adj_bin; chunk_start += blockDim.x) {
                int chunk_size = min(blockDim.x, n_parts_in_adj_bin - chunk_start);
                
                // Collaboratively load particles into shared memory
                if (threadIdx.x < chunk_size) {
                    int particle_index = bin_start + chunk_start + threadIdx.x;
                    int part_idx = d_part_ids_by_bin[particle_index];
                    shared_parts[threadIdx.x] = d_parts[part_idx];
                }
                
                // Ensure all threads have loaded their particles before proceeding
                __syncthreads();
                
                // Process all particles in this chunk
                for (int j = 0; j < chunk_size; j++) {
                    // Get particle ID for self-interaction check
                    int neighbor_idx = bin_start + chunk_start + j;
                    int neighbor_part_id = d_part_ids_by_bin[neighbor_idx];
                    
                    // Skip self-interaction
                    if (neighbor_part_id == part1_id)
                        continue;
                        
                    // Access particle from shared memory
                    particle_t& neighbor = shared_parts[j];
                    
                    // Apply force calculation - identical to original algorithm
                    double dx = neighbor.x - my_particle.x;
                    double dy = neighbor.y - my_particle.y;
                    double r2 = dx * dx + dy * dy;
                    
                    if (r2 > cutoff * cutoff)
                        continue;
                    
                    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
                    double r = sqrt(r2);
                    
                    // Same force calculation as original
                    double coef = (1 - cutoff / r) / r2 / mass;
                    ax_local += coef * dx;
                    ay_local += coef * dy;
                }
                
                // Ensure all threads are done with shared memory before next iteration
                __syncthreads();
            }
        }
    }
    
    // Update the particle's acceleration with locally accumulated values
    atomicAdd(&d_parts[part1_id].ax, ax_local);
    atomicAdd(&d_parts[part1_id].ay, ay_local);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //  bounce from walls
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

/*
 * CUDA kernel that counts how many particles belong to each bin
 * Each thread processes one particle and atomically increments the count for that particle's bin
 * 
 * Parameters:
 * d_parts: Array of particles in device memory
 * num_parts: Total number of particles
 * d_bin_ids_prefix_sum: Array storing bin counts (will become prefix sum later)
 *                       d_bin_ids_prefix_sum[i+1] will store count of particles in bin i
 * bin_Dim: Number of bins in each row/column (grid is bin_Dim x bin_Dim)
 * bin_size: Physical size of each bin (= 2 * cutoff)
 */
__global__ void count_parts_in_bins(particle_t* d_parts, int num_parts, int* d_bin_ids_prefix_sum, int bin_Dim, double bin_size){
    // Calculate unique thread ID from block and thread indices
    // threadIdx.x: thread index within the block (0 to NUM_THREADS-1)
    // blockIdx.x: block index (0 to blks-1)
    // blockDim.x: number of threads per block (= NUM_THREADS)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Return if this thread's ID exceeds the number of particles
    // (This happens because we round up the number of blocks)
    if (tid >= num_parts)
        return;

    // Each thread processes the particle with index = thread ID
    int& part_id = tid;

    // Calculate which bin this particle belongs to based on its position
    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);

    // Atomically increment the count for this bin
    // We use bin_id+1 because bin_ids_prefix_sum[0] needs to stay 0 for prefix sum
    // atomicAdd is needed because multiple threads might update the same bin count simultaneously
    atomicAdd(&d_bin_ids_prefix_sum[bin_id+1], 1);
}

/*
 * CUDA kernel that organizes particles into their respective bins
 * Each thread handles one particle and places its ID into the appropriate bin's section
 * 
 * Parameters:
 * d_parts: Array of particles in device memory
 * d_part_ids_by_bin: Output array where particle IDs will be stored, sorted by bin
 * d_bin_ids_prefix_sum: Array containing prefix sums of bin counts
 *                       d_bin_ids_prefix_sum[i] is the starting index for bin i's particles
 * num_parts: Total number of particles
 * bin_Dim: Number of bins in each row/column (grid is bin_Dim x bin_Dim)
 * bin_size: Physical size of each bin (= 2 * cutoff)
 */
__global__ void populate_bins(particle_t* d_parts, int* d_part_ids_by_bin, int* d_bin_ids_prefix_sum, int num_parts, int bin_Dim, double bin_size){
    // Calculate unique thread ID from block and thread indices
    // Each thread will handle one particle
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Return if this thread's ID exceeds the number of particles
    if (tid >= num_parts)
        return;

    // Each thread processes the particle with index = thread ID
    int& part_id = tid;

    // Calculate which bin this particle belongs to based on its position
    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);

    // Atomically claim a position in this bin's section of the array
    // atomicAdd returns the old value, which is our insertion position
    // We use bin_id+1 to maintain consistency with the prefix sum array
    int offset = atomicAdd(&d_bin_ids_prefix_sum[bin_id+1], 1);

    // Store the particle ID at the claimed position in the sorted array
    // This effectively groups particle IDs by their bin
    d_part_ids_by_bin[offset] = part_id;
}

void rebin_particles(particle_t* d_parts, int num_parts){
    // Step 1: Reset the prefix sum array to zeros
    // We need num_bins + 1 elements because prefix sum needs an extra element
    // Each bin will count its particles in bin_ids_prefix_sum[bin_id + 1]
    cudaMemset(d_bin_ids_prefix_sum, 0, (num_bins+1)*sizeof(int));

    // Step 2: Count how many particles fall into each bin
    // Each thread handles one particle and atomically increments the count for that particle's bin
    // After this, bin_ids_prefix_sum[i+1] contains the count of particles in bin i
    count_parts_in_bins<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_ids_prefix_sum, bin_Dim, bin_size);

    // Step 3: Compute exclusive prefix sum of bin counts
    // Wrap raw device pointers in thrust device pointers
    thrust::device_ptr<int> dev_ptr_begin(d_bin_ids_prefix_sum);
    thrust::device_ptr<int> dev_ptr_end(d_bin_ids_prefix_sum + num_bins + 1);
    thrust::exclusive_scan(thrust::device, dev_ptr_begin, dev_ptr_end, dev_ptr_begin);

    // Step 4: Populate the bins array with particle IDs
    // Each thread handles one particle:
    // 1. Finds which bin the particle belongs to
    // 2. Atomically claims a position in that bin's section of the array
    // 3. Writes its particle ID to that position
    // After this, d_part_ids_by_bin contains particle IDs sorted by bin
    populate_bins<<<blks, NUM_THREADS>>>(d_parts, d_part_ids_by_bin, d_bin_ids_prefix_sum, num_parts, bin_Dim, bin_size);
}

/*
 * Allocates GPU memory for binning data structuresf
 * 
 * Parameters:
 * num_parts: Total number of particles (used for particle ID array size)
 * num_bins: Total number of bins (used for prefix sum array size)
 * 
 * Note: This function allocates:
 * 1. d_part_ids_by_bin: Array to store particle IDs grouped by bin
 * 2. d_bin_ids_prefix_sum: Array for bin counts and prefix sums
 */
void allocate_gpu_memory(int num_parts, int num_bins) {
    cudaMalloc((void**)&d_part_ids_by_bin, num_parts * sizeof(int));
    cudaMalloc((void**)&d_bin_ids_prefix_sum, (num_bins+1) * sizeof(int));
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here
    bin_size = 2 * cutoff;
    bin_Dim = ceil(size / bin_size);
    num_bins = bin_Dim * bin_Dim;
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    
    // Allocate GPU memory for binning data structures
    allocate_gpu_memory(num_parts, num_bins);
}

void simulate_one_step(particle_t* d_parts, int num_parts, double size) {
    // Rebin particles
    rebin_particles(d_parts, num_parts);
    
    // Reset forces
    reset_forces_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts);
    
    // Compute forces with shared memory
    size_t shared_mem_size = NUM_THREADS * sizeof(particle_t);
    compute_forces_gpu<<<blks, NUM_THREADS, shared_mem_size>>>(d_parts, num_parts, bin_Dim, d_part_ids_by_bin, d_bin_ids_prefix_sum, bin_size);
    
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
}