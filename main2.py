import os
import torch
from torch.utils.cpp_extension import load
import tempfile
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

# Parse arguments for separate profiling mode
parser = argparse.ArgumentParser(description='Focal Loss CUDA Benchmark with Profiling')
parser.add_argument('--batch-size', type=int, help='Batch size for profiling run')
parser.add_argument('--profile', action='store_true', help='Run in profiling mode')
parser.add_argument('--profile-all', action='store_true', help='Profile all batch sizes')
args = parser.parse_args()

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
    const float gamma);

// Python bindings
torch::Tensor focal_loss_forward(
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const float alpha = 0.25,
    const float gamma = 2.0) {
    
    // Ensure inputs have the correct types
    auto preds_f = preds.to(torch::kFloat32);
    auto targets_i = targets.to(torch::kInt32);
    
    return focal_loss_forward_cuda(preds_f, targets_i, alpha, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &focal_loss_forward, "Focal Loss forward");
}
''')
    
    # CUDA kernel code
    with open(cuda_path, 'w') as f:
        f.write('''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for focal loss
__global__ void focal_loss_kernel(
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

// C++ wrapper to launch the CUDA kernel
torch::Tensor focal_loss_forward_cuda(
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const float alpha,
    const float gamma) 
{
    const auto batch_size = preds.size(0);
    const auto num_classes = preds.size(1);
    
    // Create output tensor
    auto loss = torch::empty({batch_size}, 
                            torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(preds.device()));
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    focal_loss_kernel<<<blocks, threads>>>(
        preds.data_ptr<float>(),
        targets.data_ptr<int>(),
        loss.data_ptr<float>(),
        batch_size,
        num_classes,
        alpha,
        gamma
    );
    
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

# CUDA implementation wrapper
def compute_focal_loss_cuda(preds, targets, alpha=0.25, gamma=2.0):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # Call the CUDA implementation
    loss = cuda_module.forward(preds, targets, alpha, gamma)
    loss_mean = loss.mean()

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    return loss_mean.item(), elapsed_time

# Function to run a single profiling using PyTorch profiler instead of nvprof
def profile_batch_size(batch_size):
    print(f"\nProfiling batch size: {batch_size}")
    
    # Create test data
    preds = torch.randn(batch_size, 10, device='cuda')
    targets = torch.randint(0, 10, (batch_size,), device='cuda')
    
    # Run both implementations multiple times for more accurate timing
    pytorch_times = []
    cuda_times = []
    
    # Run profiling for PyTorch implementation
    print("Profiling PyTorch implementation...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True) as pytorch_prof:
        with record_function("pytorch_focal_loss"):
            for _ in range(5):  # Reduced iterations for profiling
                loss, time_pytorch = focal_loss_pytorch(preds, targets)
                pytorch_times.append(time_pytorch)
    
    # Run profiling for CUDA implementation
    print("Profiling CUDA implementation...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True) as cuda_prof:
        with record_function("cuda_focal_loss"):
            for _ in range(5):  # Reduced iterations for profiling
                loss, time_cuda = compute_focal_loss_cuda(preds, targets)
                cuda_times.append(time_cuda)
    
    # Calculate average times
    avg_pytorch = sum(pytorch_times) / len(pytorch_times)
    avg_cuda = sum(cuda_times) / len(cuda_times)
    speedup = avg_pytorch / avg_cuda
    
    # Print profiling results
    print("\nPyTorch Profiling Summary:")
    print(pytorch_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\nCUDA Extension Profiling Summary:")
    print(cuda_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Get kernel execution metrics by using GPU Events directly
    # No need for separate process/script which was causing the module import error
    print("\nCollecting additional metrics via PyTorch CUDA events...")
    
    # Collect memory stats
    torch.cuda.reset_peak_memory_stats()
    loss, _ = compute_focal_loss_cuda(preds, targets)
    memory_stats = {
        'allocated': torch.cuda.memory_allocated() / (1024 * 1024),  # MB
        'reserved': torch.cuda.memory_reserved() / (1024 * 1024),    # MB
        'peak_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    }
    
    print(f"Memory allocated: {memory_stats['allocated']:.2f} MB")
    print(f"Memory reserved: {memory_stats['reserved']:.2f} MB")
    print(f"Peak memory allocated: {memory_stats['peak_allocated']:.2f} MB")
    
    # Save profiling results for later analysis
    pytorch_prof.export_chrome_trace(f"profile_pytorch_batch_{batch_size}.json")
    cuda_prof.export_chrome_trace(f"profile_cuda_batch_{batch_size}.json")
    print(f"Detailed profiling traces saved to profile_pytorch_batch_{batch_size}.json and profile_cuda_batch_{batch_size}.json")
    
    # Return performance metrics
    return {
        'batch_size': batch_size, 
        'pytorch_time_ms': avg_pytorch, 
        'cuda_time_ms': avg_cuda, 
        'speedup': speedup,
        'memory_allocated_mb': memory_stats['allocated'],
        'memory_reserved_mb': memory_stats['reserved']
    }

def run_benchmark(batch_sizes=[32, 64, 128, 256, 512, 1024, 2048, 
    4096, 8192, 16384, 32768, 65536, 131072, 262144]):
    print("\n----- Focal Loss Benchmark -----")
    print(f"{'Batch Size':<15}{'PyTorch (ms)':<15}{'CUDA (ms)':<15}{'Speedup':<10}")
    print("-" * 55)
    
    # Results storage for benchmark
    benchmark_results = []
    
    # Warmup run to initialize CUDA
    print("Warming up CUDA...")
    warmup_preds = torch.randn(32, 10, device='cuda')
    warmup_targets = torch.randint(0, 10, (32,), device='cuda')
    compute_focal_loss_cuda(warmup_preds, warmup_targets)
    focal_loss_pytorch(warmup_preds, warmup_targets)
    
    for batch_size in batch_sizes:
        # Create test data
        preds = torch.randn(batch_size, 10, device='cuda')
        targets = torch.randint(0, 10, (batch_size,), device='cuda')
        
        # Run PyTorch implementation
        loss_pytorch, time_pytorch = focal_loss_pytorch(preds, targets)
        
        # Run CUDA implementation
        loss_cuda, time_cuda = compute_focal_loss_cuda(preds, targets)
        
        # Calculate speedup
        speedup = time_pytorch / time_cuda if time_cuda > 0 else float('inf')
        
        # Get memory stats
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
        
        # Store results
        benchmark_results.append({
            'batch_size': batch_size,
            'pytorch_time_ms': time_pytorch,
            'cuda_time_ms': time_cuda,
            'speedup': speedup,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved
        })
        
        # Print results
        print(f"{batch_size:<15}{time_pytorch:<15.3f}{time_cuda:<15.3f}{speedup:<10.2f}x")
        
        # Verify results are close
        if abs(loss_pytorch - loss_cuda) > 1e-2:
            print(f"  Warning: Results differ! PyTorch: {loss_pytorch:.6f}, CUDA: {loss_cuda:.6f}")
    
    print("-" * 55)
    
    return benchmark_results

def visualize_results(results_df, filename_prefix):
    # Create a directory for saving results if it doesn't exist
    os.makedirs('profiling_results', exist_ok=True)
    
    # Create plots with default style (no custom style)
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot 1: Batch Size vs Execution Time
    axes[0].plot(results_df['batch_size'], results_df['pytorch_time_ms'], 'o-', label='PyTorch')
    axes[0].plot(results_df['batch_size'], results_df['cuda_time_ms'], 'o-', label='CUDA')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')  # Using log scale for both axes improves visualization
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Execution Time (ms)')
    axes[0].set_title('Batch Size vs Execution Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Batch Size vs Speedup
    axes[1].plot(results_df['batch_size'], results_df['speedup'], 'o-', color='green')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Speedup (PyTorch/CUDA)')
    axes[1].set_title('Batch Size vs Speedup')
    axes[1].grid(True)
    
    # Plot 3: Batch Size vs Memory Usage
    if 'memory_allocated_mb' in results_df.columns:
        axes[2].plot(results_df['batch_size'], results_df['memory_allocated_mb'], 'o-', label='Allocated', color='purple')
        if 'memory_reserved_mb' in results_df.columns:
            axes[2].plot(results_df['batch_size'], results_df['memory_reserved_mb'], 'o-', label='Reserved', color='orange')
        axes[2].set_xscale('log')
        axes[2].set_xlabel('Batch Size')
        axes[2].set_ylabel('Memory Usage (MB)')
        axes[2].set_title('Batch Size vs Memory Usage')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'profiling_results/{filename_prefix}_plots.png')
    print(f"Plots saved to profiling_results/{filename_prefix}_plots.png")
    
    # Create a table of results
    result_table = results_df.copy()
    result_table['batch_size'] = result_table['batch_size'].astype(str)
    result_table.to_csv(f'profiling_results/{filename_prefix}_data.csv', index=False)
    print(f"Data saved to profiling_results/{filename_prefix}_data.csv")

# Main execution
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs('profiling_results', exist_ok=True)
    
    # Define batch sizes for profiling
    # Reduce the largest batch sizes if running into memory issues
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 
                  8192, 16384, 32768, 65536, 131072, 262144]
    
    # Print GPU information
    print("\nGPU Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Total Memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    if args.profile:
        # Run single profiling for a specific batch size
        if args.batch_size is not None:
            result = profile_batch_size(args.batch_size)
            print(f"\nProfiling Results for Batch Size {args.batch_size}:")
            print(f"PyTorch Time: {result['pytorch_time_ms']:.3f} ms")
            print(f"CUDA Time: {result['cuda_time_ms']:.3f} ms")
            print(f"Speedup: {result['speedup']:.2f}x")
            print(f"Memory Allocated: {result['memory_allocated_mb']:.2f} MB")
            print(f"Memory Reserved: {result['memory_reserved_mb']:.2f} MB")
            
    elif args.profile_all:
        # Run profiling for all batch sizes
        print("\nRunning profiling for all batch sizes...")
        profile_results = []
        
        for batch_size in batch_sizes:
            try:
                result = profile_batch_size(batch_size)
                profile_results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Skipping batch size {batch_size} due to out of memory error")
                    # Add partial results with error flag
                    profile_results.append({
                        'batch_size': batch_size,
                        'pytorch_time_ms': float('nan'),
                        'cuda_time_ms': float('nan'),
                        'speedup': float('nan'),
                        'memory_allocated_mb': float('nan'),
                        'memory_reserved_mb': float('nan'),
                        'error': 'OOM'
                    })
                else:
                    raise e
        
        # Create a DataFrame from the results
        profile_df = pd.DataFrame(profile_results)
        
        # Visualize the results
        visualize_results(profile_df, f'profile_{timestamp}')
        
        print("\n--- Profiling Complete ---")
        print(profile_df)
        
    else:
        # Run the normal benchmark
        benchmark_results = run_benchmark(batch_sizes)
        benchmark_df = pd.DataFrame(benchmark_results)
        
        # Visualize the benchmark results
        visualize_results(benchmark_df, f'benchmark_{timestamp}')