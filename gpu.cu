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
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Returns the bin_id given the particle - GPU version
__device__ int get_bin_id_for_particle_gpu(particle_t* part, int bin_Dim, double bin_size){
    int bin_x = min(bin_Dim - 1, max(0, (int)(part->x / bin_size)));
    int bin_y = min(bin_Dim - 1, max(0, (int)(part->y / bin_size)));
    return bin_x + bin_y * bin_Dim;
}

__global__ void count_parts_in_bins(particle_t* d_parts, int num_parts, int* d_bin_ids_prefix_sum, int bin_Dim, double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_parts)
        return;

    int bin_id = get_bin_id_for_particle_gpu(&d_parts[tid], bin_Dim, bin_size);
    atomicAdd(&d_bin_ids_prefix_sum[bin_id+1], 1);
}

__global__ void populate_bins(particle_t* d_parts, int* d_part_ids_by_bin, int* d_bin_counts, int* d_bin_prefix_sum, int num_parts, int bin_Dim, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_parts)
        return;

    int bin_id = get_bin_id_for_particle_gpu(&d_parts[tid], bin_Dim, bin_size);
    int offset = atomicAdd(&d_bin_counts[bin_id], 1);
    d_part_ids_by_bin[d_bin_prefix_sum[bin_id] + offset] = tid;
}

__global__ void compute_forces_gpu(particle_t* d_parts, int num_parts, int bin_Dim, int* d_part_ids_by_bin, int* d_bin_ids_prefix_sum, double bin_size) {
    // Get thread (particle) ID based on global thread indexing
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    
    // Declare shared memory for particles in the current block
    extern __shared__ particle_t shared_parts[];
    
    // Get the actual particle ID from the binned array
    int part1_id = tid;
    particle_t& my_particle = d_parts[part1_id];
    
    // Initialize forces to zero
    my_particle.ax = 0;
    my_particle.ay = 0;
    
    // Calculate bin coordinates for current particle
    int bin_x_base = min(bin_Dim - 1, max(0, (int)(my_particle.x / bin_size)));
    int bin_y_base = min(bin_Dim - 1, max(0, (int)(my_particle.y / bin_size)));
    
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
            
            // Process all particles in this bin
            for (int j = bin_start; j < bin_end; j++) {
                int neighbor_id = d_part_ids_by_bin[j];
                apply_force_gpu(my_particle, d_parts[neighbor_id]);
            }
        }
    }
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

void rebin_particles(particle_t* d_parts, int num_parts){
    // Step 1: Reset the prefix sum array to zeros
    cudaMemset(d_bin_ids_prefix_sum, 0, (num_bins+1)*sizeof(int));

    // Step 2: Count particles per bin
    count_parts_in_bins<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_ids_prefix_sum, bin_Dim, bin_size);
    cudaDeviceSynchronize();

    // Step 3: Compute exclusive prefix sum
    thrust::device_ptr<int> dev_ptr_begin(d_bin_ids_prefix_sum);
    thrust::device_ptr<int> dev_ptr_end(d_bin_ids_prefix_sum + num_bins + 1);
    thrust::exclusive_scan(thrust::device, dev_ptr_begin, dev_ptr_end, dev_ptr_begin);
    
    // Step 4: Create temporary array for counting during population
    int* d_bin_counts;
    cudaMalloc((void**)&d_bin_counts, num_bins * sizeof(int));
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));
    
    // Step 5: Populate the bins array with particle IDs
    populate_bins<<<blks, NUM_THREADS>>>(d_parts, d_part_ids_by_bin, d_bin_counts, d_bin_ids_prefix_sum, num_parts, bin_Dim, bin_size);
    cudaDeviceSynchronize();
    
    // Free temporary array
    cudaFree(d_bin_counts);
}

void allocate_gpu_memory(int num_parts, int num_bins) {
    cudaMalloc((void**)&d_part_ids_by_bin, num_parts * sizeof(int));
    cudaMalloc((void**)&d_bin_ids_prefix_sum, (num_bins+1) * sizeof(int));
}

void free_gpu_memory() {
    cudaFree(d_part_ids_by_bin);
    cudaFree(d_bin_ids_prefix_sum);
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
    // First rebinning (sorting particles by spatial proximity)
    rebin_particles(d_parts, num_parts);
    
    // Compute forces - now optimized with spatial binning
    size_t shared_mem_size = NUM_THREADS * sizeof(particle_t);
    compute_forces_gpu<<<blks, NUM_THREADS, shared_mem_size>>>(d_parts, num_parts, bin_Dim, d_part_ids_by_bin, d_bin_ids_prefix_sum, bin_size);
    cudaDeviceSynchronize();
    
    // Move particles - unchanged from reference
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
    cudaDeviceSynchronize();
}