#include "common.h"
#include "assert.h"
#include <cmath>
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <chrono> // For timing measurements

using namespace std;
using namespace std::chrono;

#define NUM_THREADS 16

double bin_size; // bin_size = 2 * cutoff
int bin_Dim;  // number of stacked bins in each direction.
int num_bins; // = bin_Dim * bin_Dim
int* d_part_ids_by_bin;
int* d_bin_ids_prefix_sum;

// Timing variables
double rebinning_time = 0.0;
double force_reset_time = 0.0;
double force_compute_time = 0.0;
double movement_time = 0.0;
double sync_time = 0.0;
double thrust_time = 0.0;
double memory_op_time = 0.0;

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
 */
__global__ void reset_forces_gpu(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
}

// Returns the bin_id given the particle - GPU version
__device__ int get_bin_id_for_particle_gpu(particle_t* part, int bin_Dim, double bin_size){
    int bin_x = (int)(part->x / bin_size);
    int bin_y = (int)(part->y / bin_size);
    return bin_x + bin_y * bin_Dim;
}

__global__ void compute_forces_gpu(particle_t* d_parts, int num_parts, int bin_Dim, int* d_part_ids_by_bin, int* d_bin_ids_prefix_sum, double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int part1_id = d_part_ids_by_bin[tid];
    particle_t* particle_1_ptr = &d_parts[part1_id];
    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part1_id], bin_Dim, bin_size);
    int bin_x_base = bin_id % bin_Dim;
    int bin_y_base = bin_id / bin_Dim;

    for(int bin_dy = -1; bin_dy <= 1; bin_dy++){
        for(int bin_dx = -1; bin_dx <= 1; bin_dx++){
            int bin_x = bin_x_base + bin_dx;
            int bin_y = bin_y_base + bin_dy;
            if(bin_x < 0 || bin_x >= bin_Dim || bin_y < 0 || bin_y >= bin_Dim)
                continue;
            int adj_bin_id = bin_x + bin_y * bin_Dim;
            // Iterate over all particles in the 'adj_bin_id'
            int n_parts_in_adj_bin = d_bin_ids_prefix_sum[adj_bin_id+1] - d_bin_ids_prefix_sum[adj_bin_id];
            for(int part2_local_id = 0; part2_local_id < n_parts_in_adj_bin; part2_local_id++){
                particle_t* particle_2_ptr = d_parts + d_part_ids_by_bin[d_bin_ids_prefix_sum[adj_bin_id] + part2_local_id];
                apply_force_gpu(*particle_1_ptr, *particle_2_ptr);
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

__global__ void count_parts_in_bins(particle_t* d_parts, int num_parts, int* d_bin_ids_prefix_sum, int bin_Dim, double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_parts)
        return;

    int& part_id = tid;
    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);
    atomicAdd(&d_bin_ids_prefix_sum[bin_id+1], 1);
}

__global__ void populate_bins(particle_t* d_parts, int* d_part_ids_by_bin, int* d_bin_ids_prefix_sum, int* d_bin_pos_copy, int num_parts, int bin_Dim, double bin_size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid >= num_parts)
        return;

    int& part_id = tid;
    int bin_id = get_bin_id_for_particle_gpu(&d_parts[part_id], bin_Dim, bin_size);
    int offset = atomicAdd(&d_bin_pos_copy[bin_id], 1);
    d_part_ids_by_bin[d_bin_ids_prefix_sum[bin_id] + offset] = part_id;
}

void rebin_particles(particle_t* d_parts, int num_parts){
    auto start_time = high_resolution_clock::now();
    
    // Step 1: Reset the prefix sum array to zeros
    auto mem_start = high_resolution_clock::now();
    cudaMemset(d_bin_ids_prefix_sum, 0, (num_bins+1)*sizeof(int));
    auto mem_end = high_resolution_clock::now();
    memory_op_time += duration_cast<microseconds>(mem_end - mem_start).count() / 1000000.0;

    // Step 2: Count how many particles fall into each bin
    count_parts_in_bins<<<blks, NUM_THREADS>>>(d_parts, num_parts, d_bin_ids_prefix_sum, bin_Dim, bin_size);
    
    auto sync_start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto sync_end = high_resolution_clock::now();
    sync_time += duration_cast<microseconds>(sync_end - sync_start).count() / 1000000.0;
    
    // Step 3: Compute exclusive prefix sum of bin counts
    auto thrust_start = high_resolution_clock::now();
    thrust::device_ptr<int> dev_ptr_begin(d_bin_ids_prefix_sum);
    thrust::device_ptr<int> dev_ptr_end(d_bin_ids_prefix_sum + num_bins + 1);
    thrust::exclusive_scan(thrust::device, dev_ptr_begin, dev_ptr_end, dev_ptr_begin);
    auto thrust_end = high_resolution_clock::now();
    thrust_time += duration_cast<microseconds>(thrust_end - thrust_start).count() / 1000000.0;

    // Step 4: Create copy of prefix sum array for bin population
    mem_start = high_resolution_clock::now();
    int* d_bin_pos_copy;
    cudaMalloc((void**)&d_bin_pos_copy, (num_bins+1)*sizeof(int));
    cudaMemset(d_bin_pos_copy, 0, (num_bins+1)*sizeof(int));
    mem_end = high_resolution_clock::now();
    memory_op_time += duration_cast<microseconds>(mem_end - mem_start).count() / 1000000.0;
    
    // Step 5: Populate the bins
    populate_bins<<<blks, NUM_THREADS>>>(d_parts, d_part_ids_by_bin, d_bin_ids_prefix_sum, d_bin_pos_copy, num_parts, bin_Dim, bin_size);
    
    sync_start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    sync_end = high_resolution_clock::now();
    sync_time += duration_cast<microseconds>(sync_end - sync_start).count() / 1000000.0;
    
    mem_start = high_resolution_clock::now();
    cudaFree(d_bin_pos_copy);
    mem_end = high_resolution_clock::now();
    memory_op_time += duration_cast<microseconds>(mem_end - mem_start).count() / 1000000.0;
    
    auto end_time = high_resolution_clock::now();
    rebinning_time += duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
}

void allocate_gpu_memory(int num_parts, int num_bins) {
    cudaMalloc((void**)&d_part_ids_by_bin, num_parts * sizeof(int));
    cudaMalloc((void**)&d_bin_ids_prefix_sum, (num_bins+1)*sizeof(int));
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // Reset timing variables
    rebinning_time = 0.0;
    force_reset_time = 0.0;
    force_compute_time = 0.0;
    movement_time = 0.0;
    sync_time = 0.0;
    thrust_time = 0.0;
    memory_op_time = 0.0;
    
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
    auto start_time = high_resolution_clock::now();
    reset_forces_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts);
    auto sync_start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    auto sync_end = high_resolution_clock::now();
    auto end_time = high_resolution_clock::now();
    
    force_reset_time += duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
    sync_time += duration_cast<microseconds>(sync_end - sync_start).count() / 1000000.0;
    
    // Compute forces
    start_time = high_resolution_clock::now();
    compute_forces_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, bin_Dim, d_part_ids_by_bin, d_bin_ids_prefix_sum, bin_size);
    sync_start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    sync_end = high_resolution_clock::now();
    end_time = high_resolution_clock::now();
    
    force_compute_time += duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
    sync_time += duration_cast<microseconds>(sync_end - sync_start).count() / 1000000.0;
    
    // Move particles
    start_time = high_resolution_clock::now();
    move_gpu<<<blks, NUM_THREADS>>>(d_parts, num_parts, size);
    sync_start = high_resolution_clock::now();
    cudaDeviceSynchronize();
    sync_end = high_resolution_clock::now();
    end_time = high_resolution_clock::now();
    
    movement_time += duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
    sync_time += duration_cast<microseconds>(sync_end - sync_start).count() / 1000000.0;
}

// Function to print timing data
void print_performance_data(int num_parts, double total_simulation_time) {
    cout << "\n=== Performance Breakdown for " << num_parts << " particles ===" << endl;
    cout << "Total simulation time: " << total_simulation_time << " seconds" << endl;
    
    // Calculate compute time (excluding synchronization and memory operations)
    double compute_time = rebinning_time + force_reset_time + force_compute_time + movement_time - sync_time - memory_op_time - thrust_time;
    
    cout << "\nComputation time breakdown:" << endl;
    cout << "  Compute time:          " << compute_time << " s (" << (compute_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Synchronization time:  " << sync_time << " s (" << (sync_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Memory operation time: " << memory_op_time << " s (" << (memory_op_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Thrust operations:     " << thrust_time << " s (" << (thrust_time / total_simulation_time * 100) << "%)" << endl;
    
    cout << "\nDetailed breakdown:" << endl;
    cout << "  Rebinning:          " << rebinning_time << " s (" << (rebinning_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Force reset:        " << force_reset_time << " s (" << (force_reset_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Force computation:  " << force_compute_time << " s (" << (force_compute_time / total_simulation_time * 100) << "%)" << endl;
    cout << "  Movement:           " << movement_time << " s (" << (movement_time / total_simulation_time * 100) << "%)" << endl;
    
    cout << "\nScaling assessment:" << endl;
    cout << "  Rebinning:         O(N) - Each particle is assigned to exactly one bin, with constant bin capacity" << endl;
    cout << "  Force computation: O(N) - With spatial binning, each particle interacts with a constant number of neighbors" << endl;
    cout << "  Overall:           O(N) - All major components scale linearly with particle count" << endl;
    
    // Calculate and display key metrics
    double time_per_particle = total_simulation_time / num_parts;
    double interactions_per_sec = (num_parts / force_compute_time) * 9; // Assuming 9 neighboring bins on average
    
    cout << "\nKey metrics:" << endl;
    cout << "  Time per particle:       " << time_per_particle * 1000 << " milliseconds" << endl;
    cout << "  Interactions per second: " << interactions_per_sec << endl;
    cout << "=============================================" << endl;
}

// Function to free GPU memory when done
void cleanup_simulation() {
    cudaFree(d_part_ids_by_bin);
    cudaFree(d_bin_ids_prefix_sum);
}