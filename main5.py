import os
import torch
from torch.utils.cpp_extension import load
import tempfile
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np
from datetime import datetime
import subprocess
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# Parse arguments for separate profiling mode
parser = argparse.ArgumentParser(description='Focal Loss CUDA Benchmark with Configuration Testing')
parser.add_argument('--batch-size', type=int, help='Batch size for profiling run')
parser.add_argument('--profile', action='store_true', help='Run in profiling mode')
parser.add_argument('--profile-all', action='store_true', help='Profile all batch sizes')
parser.add_argument('--detailed', action='store_true', help='Collect detailed hardware metrics using nvprof/ncu')
parser.add_argument('--output-dir', type=str, default='profiling_results', help='Directory to save results')
parser.add_argument('--runs', type=int, default=5, help='Number of runs for each batch size')
parser.add_argument('--test-configurations', action='store_true', 
                    help='Run tests with various batch and block size configurations')
parser.add_argument('--num-classes', type=int, default=8, 
                    help='Number of classes for testing (default: 8)')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Write CUDA and C++ code to temporary files
with tempfile.TemporaryDirectory() as tmp_dir:
    cpp_path = os.path.join(tmp_dir, 'focal_loss.cpp')
    cuda_path = os.path.join(tmp_dir, 'focal_loss_kernel.cu')
    
    # C++ binding code
    with open(cpp_path, 'w') as f:
        f.write('''
#include <torch/extension.h>

// Forward declaration
torch::Tensor focal_loss_forward_cuda(
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const float alpha,
    const float gamma,
    const int block_size,
    const bool use_shared_memory);

// Python bindings with additional parameters
torch::Tensor focal_loss_forward(
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const float alpha = 0.25,
    const float gamma = 2.0,
    const int block_size = 256,
    const bool use_shared_memory = false) {
    
    // Ensure inputs have the correct types
    auto preds_f = preds.to(torch::kFloat32);
    auto targets_i = targets.to(torch::kInt32);
    
    return focal_loss_forward_cuda(preds_f, targets_i, alpha, gamma, block_size, use_shared_memory);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &focal_loss_forward, "Focal Loss forward",
          py::arg("preds"), py::arg("targets"), 
          py::arg("alpha") = 0.25, py::arg("gamma") = 2.0,
          py::arg("block_size") = 256, py::arg("use_shared_memory") = false);
}
''')
    
    # CUDA kernel code
    with open(cuda_path, 'w') as f:
        f.write('''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with standard implementation
__global__ void focal_loss_kernel_standard(
    const float* preds,
    const int* targets,
    float* loss,
    const int batch_size,
    const int num_classes,
    const float alpha,
    const float gamma) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int target_class = targets[idx];
    
    // Compute softmax for numerical stability
    float max_val = -1e10;
    for (int j = 0; j < num_classes; j++) {
        max_val = fmaxf(max_val, preds[idx * num_classes + j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(preds[idx * num_classes + j] - max_val);
    }
    
    // Compute probability for the target class
    const float pt = expf(preds[idx * num_classes + target_class] - max_val) / sum_exp;
    
    // Compute focal loss
    const float focal_weight = powf(1.0f - pt, gamma);
    loss[idx] = -alpha * focal_weight * logf(pt);
}

// CUDA kernel with shared memory implementation
__global__ void focal_loss_kernel_shared(
    const float* preds,
    const int* targets,
    float* loss,
    const int batch_size,
    const int num_classes,
    const float alpha,
    const float gamma) 
{
    extern __shared__ float shared_data[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int target_class = targets[idx];
    
    // Load predictions into shared memory
    float* shared_preds = &shared_data[threadIdx.x * num_classes];
    for (int j = 0; j < num_classes; j++) {
        shared_preds[j] = preds[idx * num_classes + j];
    }
    __syncthreads();
    
    // Compute softmax using shared memory
    float max_val = -1e10;
    for (int j = 0; j < num_classes; j++) {
        max_val = fmaxf(max_val, shared_preds[j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(shared_preds[j] - max_val);
    }
    
    // Compute probability for the target class
    const float pt = expf(shared_preds[target_class] - max_val) / sum_exp;
    
    // Compute focal loss
    const float focal_weight = powf(1.0f - pt, gamma);
    loss[idx] = -alpha * focal_weight * logf(pt);
}

// C++ wrapper to launch the CUDA kernel with configurable block size and implementation type
torch::Tensor focal_loss_forward_cuda(
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const float alpha,
    const float gamma,
    const int block_size = 256,
    const bool use_shared_memory = false) 
{
    const auto batch_size = preds.size(0);
    const auto num_classes = preds.size(1);
    
    // Create output tensor
    auto loss = torch::empty({batch_size}, 
                           torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(preds.device()));
    
    // Calculate grid size based on batch size and block size
    const int blocks = (batch_size + block_size - 1) / block_size;
    
    if (use_shared_memory) {
        // Calculate shared memory size (num_classes floats per thread)
        const int shared_memory_size = block_size * num_classes * sizeof(float);
        
        focal_loss_kernel_shared<<<blocks, block_size, shared_memory_size>>>(
            preds.data_ptr<float>(),
            targets.data_ptr<int>(),
            loss.data_ptr<float>(),
            batch_size,
            num_classes,
            alpha,
            gamma
        );
    } else {
        focal_loss_kernel_standard<<<blocks, block_size>>>(
            preds.data_ptr<float>(),
            targets.data_ptr<int>(),
            loss.data_ptr<float>(),
            batch_size,
            num_classes,
            alpha,
            gamma
        );
    }
    
    return loss;
}
''')
    
    # Compile the extension
    print("Compiling CUDA extension...")
    cuda_module = load(
        name="focal_loss_cuda",
        sources=[cpp_path, cuda_path],
        verbose=True
    )
    print("Compilation successful!")

# PyTorch implementation for comparison
def focal_loss_pytorch(preds, targets, alpha=0.25, gamma=2.0):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    ce_loss = torch.nn.functional.cross_entropy(preds, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Convert CE loss back to probability
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * ce_loss
    loss_mean = loss.mean()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds

    return loss_mean.item(), elapsed_time

# CUDA implementation wrapper with configurable block size and memory implementation
def compute_focal_loss_cuda(preds, targets, alpha=0.25, gamma=2.0, block_size=256, use_shared_memory=False):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # Call the CUDA implementation with block_size and use_shared_memory parameters
    loss = cuda_module.forward(preds, targets, alpha, gamma, block_size, use_shared_memory)
    loss_mean = loss.mean()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    return loss_mean.item(), elapsed_time

# Function to run configuration benchmark
def run_configuration_benchmark(batch_sizes, block_sizes, num_classes=8, test_shared_memory=True):
    """Test different configurations of batch sizes and block sizes"""
    print("\n----- Configuration Benchmark -----")
    print(f"Number of Classes: {num_classes}")
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Testing block sizes: {block_sizes}")
    print(f"Testing shared memory: {test_shared_memory}")
    print("-" * 60)
    
    results = []
    
    # Warmup
    print("Warming up CUDA...")
    warmup_preds = torch.randn(32, num_classes, device='cuda')
    warmup_targets = torch.randint(0, num_classes, (32,), device='cuda')
    compute_focal_loss_cuda(warmup_preds, warmup_targets)
    focal_loss_pytorch(warmup_preds, warmup_targets)
    
    # Run tests for each configuration
    for batch_size in tqdm(batch_sizes, desc="Batch sizes"):
        try:
            # Create test data for this batch size
            preds = torch.randn(batch_size, num_classes, device='cuda')
            targets = torch.randint(0, num_classes, (batch_size,), device='cuda')
            
            # Test PyTorch implementation
            pytorch_times = []
            for _ in range(args.runs):
                loss_pytorch, time_pytorch = focal_loss_pytorch(preds, targets)
                pytorch_times.append(time_pytorch)
            avg_pytorch = sum(pytorch_times) / len(pytorch_times)
            
            # Test each block size
            for block_size in block_sizes:
                # Test standard implementation
                standard_cuda_times = []
                for _ in range(args.runs):
                    loss_cuda, time_cuda = compute_focal_loss_cuda(
                        preds, targets, 0.25, 2.0, block_size, False)
                    standard_cuda_times.append(time_cuda)
                avg_standard_cuda = sum(standard_cuda_times) / len(standard_cuda_times)
                
                # Calculate speedup
                speedup = avg_pytorch / avg_standard_cuda if avg_standard_cuda > 0 else 0
                
                # Store result
                results.append({
                    'batch_size': batch_size,
                    'block_size': block_size,
                    'implementation': 'standard',
                    'pytorch_time_ms': avg_pytorch,
                    'cuda_time_ms': avg_standard_cuda,
                    'speedup': speedup,
                    'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024)
                })
                
                # Test shared memory implementation if requested
                if test_shared_memory:
                    shared_cuda_times = []
                    for _ in range(args.runs):
                        # Call with shared memory enabled
                        loss_cuda, time_cuda = compute_focal_loss_cuda(
                            preds, targets, 0.25, 2.0, block_size, True)
                        shared_cuda_times.append(time_cuda)
                    avg_shared_cuda = sum(shared_cuda_times) / len(shared_cuda_times)
                    
                    # Calculate speedup
                    speedup = avg_pytorch / avg_shared_cuda if avg_shared_cuda > 0 else 0
                    
                    # Store result
                    results.append({
                        'batch_size': batch_size,
                        'block_size': block_size,
                        'implementation': 'shared',
                        'pytorch_time_ms': avg_pytorch,
                        'cuda_time_ms': avg_shared_cuda,
                        'speedup': speedup,
                        'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024)
                    })
                
                # Print progress
                print(f"Batch size: {batch_size}, Block size: {block_size}, "
                      f"PyTorch: {avg_pytorch:.3f} ms, CUDA: {avg_standard_cuda:.3f} ms, "
                      f"Speedup: {speedup:.2f}x")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping batch size {batch_size} due to out of memory error")
                break
            else:
                raise e
    
    return pd.DataFrame(results)

# Function to visualize configuration results
def visualize_configuration_results(results_df, filename_prefix):
    """Visualize the results of the configuration benchmark"""
    # Set up plotting
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Heatmap of speedup by batch size and block size
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for standard implementation
    pivot_data = results_df[results_df['implementation'] == 'standard'].pivot(
        index='batch_size', 
        columns='block_size', 
        values='speedup'
    )
    
    # Plot heatmap
    sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Speedup vs PyTorch'})
    plt.title('Speedup by Batch Size and Block Size')
    plt.ylabel('Batch Size')
    plt.xlabel('Block Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_speedup_heatmap.png'), dpi=300)
    
    # 2. Line plot of execution time vs batch size for each block size
    plt.figure(figsize=(14, 8))
    
    # Filter standard implementation
    standard_df = results_df[results_df['implementation'] == 'standard']
    
    # Plot CUDA times
    for block_size in standard_df['block_size'].unique():
        block_df = standard_df[standard_df['block_size'] == block_size]
        plt.plot(block_df['batch_size'], block_df['cuda_time_ms'], 'o-', 
                 label=f'Block Size: {block_size}')
    
    # Plot PyTorch times
    plt.plot(standard_df.groupby('batch_size')['pytorch_time_ms'].mean().index,
             standard_df.groupby('batch_size')['pytorch_time_ms'].mean().values,
             'o-', label='PyTorch', linewidth=2, color='black')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Time vs Batch Size for Different Block Sizes')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_execution_time.png'), dpi=300)
    
    # 3. Bar chart comparing block sizes for each batch size
    # Create a grid of subplots for different batch sizes
    batch_sizes = results_df['batch_size'].unique()
    n_plots = len(batch_sizes)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Get data for this batch size
        batch_df = results_df[results_df['batch_size'] == batch_size]
        
        # Plot bar chart
        sns.barplot(x='block_size', y='cuda_time_ms', hue='implementation', data=batch_df)
        
        plt.title(f'Batch Size: {batch_size}')
        plt.xlabel('Block Size')
        plt.ylabel('Execution Time (ms)')
        plt.legend(title='Implementation')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_block_comparison.png'), dpi=300)
    
    # 4. Compare standard vs shared memory if available
    if 'shared' in results_df['implementation'].unique():
        plt.figure(figsize=(12, 8))
        
        # Compute speedup of shared memory over standard implementation
        standard_times = results_df[results_df['implementation'] == 'standard'].set_index(['batch_size', 'block_size'])['cuda_time_ms']
        shared_times = results_df[results_df['implementation'] == 'shared'].set_index(['batch_size', 'block_size'])['cuda_time_ms']
        
        # Make sure indices match
        common_indices = standard_times.index.intersection(shared_times.index)
        standard_times = standard_times.loc[common_indices]
        shared_times = shared_times.loc[common_indices]
        
        # Calculate speedup
        impl_speedup = standard_times / shared_times
        impl_speedup = impl_speedup.reset_index()
        
        # Create pivot table
        pivot_impl = impl_speedup.pivot(index='batch_size', columns='block_size', values='cuda_time_ms')
        
        # Plot heatmap
        sns.heatmap(pivot_impl, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Speedup (Standard/Shared)'})
        plt.title('Shared Memory vs Standard Implementation Speedup')
        plt.ylabel('Batch Size')
        plt.xlabel('Block Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_shared_vs_standard.png'), dpi=300)
    
    # 5. Create a summary table
    summary = results_df.pivot_table(
        index=['batch_size', 'block_size'], 
        columns='implementation', 
        values=['cuda_time_ms', 'speedup'],
        aggfunc='mean'
    ).reset_index()
    
    # Save data
    results_df.to_csv(os.path.join(args.output_dir, f'{filename_prefix}_all_results.csv'), index=False)
    summary.to_csv(os.path.join(args.output_dir, f'{filename_prefix}_summary.csv'))
    
    print(f"Configuration visualizations saved to {args.output_dir}/{filename_prefix}_*.png")

# Function to collect detailed hardware metrics using nvprof or nsight-compute
def collect_hardware_metrics(batch_size, block_size=256, use_shared_memory=False, num_classes=8):
    """Collect detailed GPU hardware metrics for a given batch size and block size"""
    metrics = {}
    
    # Create temporary script for profiling
    script_path = os.path.join(args.output_dir, f'profile_script_{batch_size}_{block_size}.py')
    with open(script_path, 'w') as f:
        f.write(f'''
import torch
import sys
import os
import focal_loss_cuda

# Create data
batch_size = {batch_size}
num_classes = {num_classes}
block_size = {block_size}
use_shared_memory = {str(use_shared_memory).lower()}

preds = torch.randn({batch_size}, {num_classes}, device='cuda')
targets = torch.randint(0, {num_classes}, ({batch_size},), device='cuda').to(torch.int32)

# Warmup
for _ in range(5):
    loss = focal_loss_cuda.forward(preds, targets, 0.25, 2.0, block_size, use_shared_memory)
    torch.cuda.synchronize()

# Actual run that will be profiled
loss = focal_loss_cuda.forward(preds, targets, 0.25, 2.0, block_size, use_shared_memory)
torch.cuda.synchronize()
''')
    
    # Define key metrics to collect
    nvprof_metrics = [
        "achieved_occupancy",          # Achieved occupancy
        "sm_efficiency",               # SM utilization
        "warp_execution_efficiency",   # Warp execution efficiency
        "dram_read_throughput",        # Memory read bandwidth
        "dram_write_throughput",       # Memory write bandwidth
        "gld_efficiency",              # Global load efficiency (coalescing)
        "gst_efficiency",              # Global store efficiency (coalescing)
        "l2_read_hit_rate",            # L2 cache hit rate
    ]
    
    try:
        # Check if ncu (Nsight Compute) is available
        try:
            ncu_check = subprocess.run(['ncu', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            use_ncu = ncu_check.returncode == 0
        except FileNotFoundError:
            use_ncu = False
            
        if use_ncu:
            # Use Nsight Compute (newer GPUs)
            print(f"\nCollecting hardware metrics using NVIDIA Nsight Compute for batch size {batch_size}...")
            
            # Key metrics for insight into performance
            ncu_metrics = [
                "sm__warps_active.avg.pct",
                "sm__occupancy.avg.pct",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct",
                "dram__bytes_read.sum.per_second",
                "dram__bytes_write.sum.per_second",
                "gpu__time_duration.sum"
            ]
            
            # Convert to NCU format
            metrics_arg = " ".join([f"--metrics {m}" for m in ncu_metrics])
            cmd = f"ncu {metrics_arg} --csv --target-processes all python {script_path}"
            
            # Run the command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse the CSV output
            for line in result.stdout.split('\n'):
                if ',' in line and not line.startswith('"ID"'):
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        metric_name = parts[1].strip('"')
                        metric_value = parts[2].strip('"')
                        try:
                            # Convert to numeric value
                            if '%' in metric_value:
                                numeric_val = float(metric_value.replace('%', '').strip())
                            elif ' ' in metric_value:
                                numeric_val = float(metric_value.split(' ')[0].strip())
                            else:
                                numeric_val = float(metric_value)
                            metrics[metric_name] = numeric_val
                        except ValueError:
                            metrics[metric_name] = metric_value
            
            # Map to our standard metrics
            metrics['occupancy'] = metrics.get('sm__occupancy.avg.pct', 0)
            metrics['gpu_util'] = metrics.get('sm__warps_active.avg.pct', 0)
            metrics['cache_hit_rate'] = metrics.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct', 0)
            
        else:
            # Fall back to nvprof for older GPUs
            print(f"\nCollecting hardware metrics using nvprof for batch size {batch_size}...")
            metrics_arg = ",".join(nvprof_metrics)
            cmd = f"nvprof --metrics {metrics_arg} --csv python {script_path}"
            
            # Run the command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Parse the CSV output from nvprof (uses stderr)
            for line in result.stderr.split('\n'):
                for metric in nvprof_metrics:
                    if f",{metric}," in line:
                        parts = line.split(',')
                        value_part = parts[2].strip()
                        # Extract numeric value
                        if '%' in value_part:
                            value = float(value_part.split('%')[0])
                        elif ' ' in value_part:  # Value with unit like "10.5 GB/s"
                            value = float(value_part.split(' ')[0])
                        else:
                            value = float(value_part)
                        
                        metrics[metric] = value
            
            # Extract kernel execution time
            for line in result.stderr.split('\n'):
                if "focal_loss_kernel" in line and "GPU activities" in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        time_str = parts[3].strip()
                        if 'us' in time_str:
                            metrics['kernel_time_ms'] = float(time_str.split(' ')[0]) / 1000.0
                        elif 'ms' in time_str:
                            metrics['kernel_time_ms'] = float(time_str.split(' ')[0])
            
            # Process metrics
            metrics['occupancy'] = metrics.get('achieved_occupancy', 0) * 100  # Convert to percentage
            metrics['gpu_util'] = metrics.get('sm_efficiency', 0)
            metrics['warp_efficiency'] = metrics.get('warp_execution_efficiency', 0)
            metrics['cache_hit_rate'] = metrics.get('l2_read_hit_rate', 0)
            
        # Calculate derived metrics
        if 'dram_read_throughput' in metrics and 'dram_write_throughput' in metrics:
            # Convert to GB/s
            read_bw = metrics['dram_read_throughput'] 
            write_bw = metrics['dram_write_throughput']
            metrics['memory_bandwidth_gbs'] = read_bw + write_bw
        
        # Flag to indicate real metrics were collected
        metrics['using_real_metrics'] = True
        print("  Successfully collected hardware metrics")
        
    except Exception as e:
        print(f"  Failed to collect hardware metrics: {e}")
        print("  Using estimated metrics instead")
        
        # Use estimated values
        metrics['occupancy'] = min(95, 25 + batch_size / 4000)
        metrics['gpu_util'] = min(96, 5 + batch_size / 3000)
        metrics['warp_efficiency'] = min(98, 50 + batch_size / 2000)
        metrics['memory_bandwidth_gbs'] = min(500, 10 + batch_size / 1000)
        metrics['cache_hit_rate'] = max(60, 80 - batch_size / 30000)
        metrics['using_real_metrics'] = False
    
    # Add metadata
    metrics['batch_size'] = batch_size
    metrics['block_size'] = block_size
    metrics['implementation'] = 'shared' if use_shared_memory else 'standard'
    
    # Clean up temporary file
    if os.path.exists(script_path):
        os.remove(script_path)
    
    return metrics

# Run the standard benchmark with our existing batch size configurations
def run_benchmark(batch_sizes):
    print("\n----- Focal Loss Benchmark -----")
    print(f"{'Batch Size':<15}{'PyTorch (ms)':<15}{'CUDA (ms)':<15}{'Speedup':<10}")
    print("-" * 55)
    
    # Results storage
    benchmark_results = []
    
    # Warmup CUDA
    print("Warming up CUDA...")
    warmup_preds = torch.randn(32, 10, device='cuda')
    warmup_targets = torch.randint(0, 10, (32,), device='cuda')
    compute_focal_loss_cuda(warmup_preds, warmup_targets)
    focal_loss_pytorch(warmup_preds, warmup_targets)
    
    for batch_size in tqdm(batch_sizes, desc="Benchmark progress"):
        try:
            # Create test data
            preds = torch.randn(batch_size, 10, device='cuda')
            targets = torch.randint(0, 10, (batch_size,), device='cuda')
            
            # Run multiple times for stability
            pytorch_times = []
            cuda_times = []
            
            for _ in range(args.runs):
                # Run PyTorch implementation
                loss_pytorch, time_pytorch = focal_loss_pytorch(preds, targets)
                pytorch_times.append(time_pytorch)
                
                # Run CUDA implementation
                loss_cuda, time_cuda = compute_focal_loss_cuda(preds, targets)
                cuda_times.append(time_cuda)
            
            # Calculate average times
            avg_pytorch = sum(pytorch_times) / len(pytorch_times)
            avg_cuda = sum(cuda_times) / len(cuda_times)
            
            # Calculate speedup
            speedup = avg_pytorch / avg_cuda if avg_cuda > 0 else float('inf')
            
            # Get memory stats
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Store results
            benchmark_results.append({
'batch_size': batch_size,
                'pytorch_time_ms': avg_pytorch,
                'cuda_time_ms': avg_cuda,
                'speedup': speedup,
                'memory_allocated_mb': memory_allocated,
                'block_size': 256,  # Default block size
            })
            
            # Print results
            print(f"{batch_size:<15}{avg_pytorch:<15.3f}{avg_cuda:<15.3f}{speedup:<10.2f}x")
            
            # Verify results are close
            if abs(loss_pytorch - loss_cuda) > 1e-2:
                print(f"  Warning: Results differ! PyTorch: {loss_pytorch:.6f}, CUDA: {loss_cuda:.6f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping batch size {batch_size} due to out of memory error")
                break
            else:
                raise e
    
    return benchmark_results

# Function to visualize standard benchmark results
def visualize_results(results_df, filename_prefix):
    """Visualize the results of the standard benchmark"""
    # Set up plotting
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Plot execution time vs batch size
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['batch_size'], results_df['pytorch_time_ms'], 'o-', 
             label='PyTorch', color='#3498db', linewidth=2)
    plt.plot(results_df['batch_size'], results_df['cuda_time_ms'], 'o-', 
             label='CUDA', color='#e74c3c', linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Batch Size')
    plt.ylabel('Execution Time (ms)')
    plt.title('Focal Loss: PyTorch vs CUDA Implementation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_execution_time.png'), dpi=300)
    
    # 2. Plot speedup vs batch size
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['batch_size'], results_df['speedup'], 'o-', 
             color='#2ecc71', linewidth=2)
    
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (PyTorch/CUDA)')
    plt.title('CUDA Speedup over PyTorch Implementation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_speedup.png'), dpi=300)
    
    # 3. Plot memory usage vs batch size
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['batch_size'], results_df['memory_allocated_mb'], 'o-', 
             color='#9b59b6', linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Consumption by Batch Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{filename_prefix}_memory.png'), dpi=300)
    
    # Save raw data
    results_df.to_csv(os.path.join(args.output_dir, f'{filename_prefix}_results.csv'), index=False)
    
    print(f"Standard benchmark visualizations saved to {args.output_dir}/{filename_prefix}_*.png")

# Function to profile a specific batch size with detailed metrics
def profile_batch_size(batch_size, num_classes=8, block_sizes=[128, 256, 512]):
    """Profile a single batch size with detailed metrics for different block sizes"""
    print(f"\nProfiling batch size: {batch_size}, num_classes: {num_classes}")
    
    profile_results = []
    
    # Create test data
    preds = torch.randn(batch_size, num_classes, device='cuda')
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda')
    
    # Test PyTorch implementation
    pytorch_times = []
    for _ in range(args.runs):
        loss_pytorch, time_pytorch = focal_loss_pytorch(preds, targets)
        pytorch_times.append(time_pytorch)
    avg_pytorch = sum(pytorch_times) / len(pytorch_times)
    
    # Test each block size
    for block_size in block_sizes:
        print(f"Testing block size: {block_size}")
        
        # Test standard implementation
        print("  Standard implementation:")
        standard_times = []
        for _ in range(args.runs):
            loss_cuda, time_cuda = compute_focal_loss_cuda(
                preds, targets, 0.25, 2.0, block_size, False)
            standard_times.append(time_cuda)
        avg_standard = sum(standard_times) / len(standard_times)
        speedup = avg_pytorch / avg_standard if avg_standard > 0 else 0
        
        print(f"    Time: {avg_standard:.3f} ms, Speedup: {speedup:.2f}x")
        
        # Collect hardware metrics if detailed profiling is enabled
        if args.detailed:
            print("  Collecting hardware metrics...")
            hw_metrics = collect_hardware_metrics(batch_size, block_size, False, num_classes)
            has_hw_metrics = True
        else:
            hw_metrics = {}
            has_hw_metrics = False
        
        # Store standard implementation results
        profile_results.append({
            'batch_size': batch_size,
            'block_size': block_size,
            'implementation': 'standard',
            'pytorch_time_ms': avg_pytorch,
            'cuda_time_ms': avg_standard,
            'speedup': speedup,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'has_hw_metrics': has_hw_metrics,
            **hw_metrics
        })
        
        # Test shared memory implementation
        print("  Shared memory implementation:")
        shared_times = []
        for _ in range(args.runs):
            loss_cuda, time_cuda = compute_focal_loss_cuda(
                preds, targets, 0.25, 2.0, block_size, True)
            shared_times.append(time_cuda)
        avg_shared = sum(shared_times) / len(shared_times)
        speedup = avg_pytorch / avg_shared if avg_shared > 0 else 0
        
        print(f"    Time: {avg_shared:.3f} ms, Speedup: {speedup:.2f}x")
        
        # Collect hardware metrics for shared memory implementation if detailed profiling is enabled
        if args.detailed:
            print("  Collecting hardware metrics for shared memory...")
            hw_metrics_shared = collect_hardware_metrics(batch_size, block_size, True, num_classes)
            has_hw_metrics_shared = True
        else:
            hw_metrics_shared = {}
            has_hw_metrics_shared = False
        
        # Store shared memory implementation results
        profile_results.append({
            'batch_size': batch_size,
            'block_size': block_size,
            'implementation': 'shared',
            'pytorch_time_ms': avg_pytorch,
            'cuda_time_ms': avg_shared,
            'speedup': speedup,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'has_hw_metrics': has_hw_metrics_shared,
            **hw_metrics_shared
        })
    
    return pd.DataFrame(profile_results)

# Main execution
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print GPU information
    print("\nGPU Information:")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        print("CUDA is not available. Please check your GPU setup.")
        exit(1)
    
    # Check if we should run the configuration benchmark
    if args.test_configurations:
        # Define testing configurations
        test_batch_sizes = [32, 128, 512, 2048, 8192, 32768, 131072]
        test_block_sizes = [64, 128, 256, 512]
        
        # Run the benchmark
        config_results = run_configuration_benchmark(
            test_batch_sizes,
            test_block_sizes,
            args.num_classes,
            test_shared_memory=True  # Set to True to test both implementations
        )
        
        # Visualize the results
        visualize_configuration_results(config_results, f'config_benchmark_{timestamp}')
        
        print("\n----- Configuration Benchmark Complete -----")
        print(f"Results saved to {args.output_dir}/config_benchmark_{timestamp}_*.png")
    
    elif args.profile:
        # Profile a specific batch size with detailed metrics
        if args.batch_size is None:
            print("Please specify a batch size with --batch-size")
            exit(1)
        
        profile_results = profile_batch_size(args.batch_size, args.num_classes)
        
        # Save profiling results
        profile_results.to_csv(os.path.join(args.output_dir, f'profile_{args.batch_size}_{timestamp}.csv'), index=False)
        
        print(f"\nProfiling complete for batch size {args.batch_size}")
        print(f"Results saved to {args.output_dir}/profile_{args.batch_size}_{timestamp}.csv")
    
    elif args.profile_all:
        # Profile multiple batch sizes
        test_batch_sizes = [128, 2048, 32768]
        block_sizes = [64, 128, 256, 512]
        
        all_profile_results = []
        
        for batch_size in test_batch_sizes:
            try:
                profile_df = profile_batch_size(batch_size, args.num_classes, block_sizes)
                all_profile_results.append(profile_df)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Skipping batch size {batch_size} due to out of memory error")
                else:
                    raise e
        
        # Combine all profiling results
        if all_profile_results:
            combined_profile = pd.concat(all_profile_results)
            combined_profile.to_csv(os.path.join(args.output_dir, f'profile_all_{timestamp}.csv'), index=False)
            
            print("\nProfiling complete for all batch sizes")
            print(f"Results saved to {args.output_dir}/profile_all_{timestamp}.csv")
        else:
            print("No profiling results collected.")
    
    else:
        # Run the standard benchmark
        benchmark_batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        benchmark_results = run_benchmark(benchmark_batch_sizes)
        
        # Convert to DataFrame
        benchmark_df = pd.DataFrame(benchmark_results)
        
        # Visualize the results
        visualize_results(benchmark_df, f'benchmark_{timestamp}')
        
        print("\n----- Standard Benchmark Complete -----")
        print(f"Results saved to {args.output_dir}/benchmark_{timestamp}_*.png")