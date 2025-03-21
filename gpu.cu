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
int* d_bin_positions; // Added for safe bin population

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

__global__ void reset_forces_gpu(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
}

// Returns the bin_id given the particle - GPU version (safer version)
__device__ int get_bin_id_for_particle_gpu(particle_t* part, int bin_Dim, double bin_size){
    // Handle boundary conditions properly
    int bin_x = min(bin_Dim - 1, max(0, (int)(part->x / bin_size)));
    int bin_y = min(bin_Dim - 1, max(0, (int)(part->y / bin_size)));
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
    int bin_x_base = min(bin_Dim - 1, max(0, (int)(my_particle.x / bin_size)));
    int bin_y_base = min(bin_Dim - 1, max(0, (int)(my_particle.y / bin_size)));
    
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

__global__ void count_parts_in_bins(particle_t* d_parts, int num_parts, int* d_bin_ids_prefix_sum, int bin_Dim, double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_parts)
        return;

    int& part_id = tid;

    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);

    // Safety check for bin_id (shouldn't be needed with the improved get_bin_id function)
    if (bin_id >= 0 && bin_id < bin_Dim * bin_Dim) {
        atomicAdd(&d_bin_ids_prefix_sum[bin_id+1], 1);
    }
}

__global__ void populate_bins(particle_t* d_parts, int* d_part_ids_by_bin, int* d_bin_positions, int* d_prefix_sum, int num_parts, int bin_Dim, double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_parts)
        return;

    int& part_id = tid;

    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);

    // Use temporary array for atomic operations to avoid disturbing prefix sum
    int offset = atomicAdd(&d_bin_positions[bin_id], 1);
    
    // Add the starting position of this bin from the prefix sum
    offset += d_prefix_sum[bin_id];
    
    // Store the particle ID at the claimed position
    d_part_ids_by_bin[offset] = part_id;
}

void rebin_particles(particle_t* d_parts, int num_parts){
    // Reset the prefix sum array to zeros
    cudaMemset(d_bin_ids_prefix_sum, 0, (num_bins+1)*sizeof(int));

    // Count particles per bin
    count_parts_in_bins<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_ids_prefix_sum, bin_Dim, bin_size);
    cudaDeviceSynchronize();  // Add synchronization

    // Compute exclusive prefix sum of bin counts
    thrust::device_ptr<int> dev_ptr_begin(d_bin_ids_prefix_sum);
    thrust::device_ptr<int> dev_ptr_end(d_bin_ids_prefix_sum + num_bins + 1);
    thrust::exclusive_scan(thrust::device, dev_ptr_begin, dev_ptr_end, dev_ptr_begin);
    
    // Reset bin positions counter array
    cudaMemset(d_bin_positions, 0, num_bins*sizeof(int));
    
    // Populate bins - now passing the prefix sum array as a parameter
    populate_bins<<<blks, NUM_THREADS>>>(d_parts, d_part_ids_by_bin, d_bin_positions, d_bin_ids_prefix_sum, num_parts, bin_Dim, bin_size);
    cudaDeviceSynchronize();  // Add synchronization
}

void allocate_gpu_memory(int num_parts, int num_bins) {
    cudaMalloc((void**)&d_part_ids_by_bin, num_parts * sizeof(int));
    cudaMalloc((void**)&d_bin_ids_prefix_sum, (num_bins+1) * sizeof(int));
    // Add allocation for bin positions
    cudaMalloc((void**)&d_bin_positions, num_bins * sizeof(int));
}

void init_simulation(particle_t* parts, int num_parts, double size) {
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
    cudaDeviceSynchronize();  // Add synchronization
    
    // Compute forces with shared memory
    size_t shared_mem_size = NUM_THREADS * sizeof(particle_t);
    compute_forces_gpu<<<blks, NUM_THREADS, shared_mem_size>>>(d_parts, num_parts, bin_Dim, d_part_ids_by_bin, d_bin_ids_prefix_sum, bin_size);
    cudaDeviceSynchronize();  // Add synchronization
    
    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
    cudaDeviceSynchronize();  // Add synchronization
}

// Add cleanup function to free GPU memory
void cleanup_simulation() {
    cudaFree(d_part_ids_by_bin);
    cudaFree(d_bin_ids_prefix_sum);
    cudaFree(d_bin_positions);
}