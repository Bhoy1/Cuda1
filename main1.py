#Cuda for Focal Loss



#Pytorch Verision 
#Focal Loss is a loss function used for classification tasks when dealing with imbalanced datasets.
# It is an extension of the Cross Entropy loss and is designed to address the problem of class imbalance
# by down-weighting the loss assigned to well-classified examples. The Focal Loss is defined as:
#FL(pt) = -α(1 - pt)γ * log(pt)
#where:
#pt is the predicted probability of the true class,
#α is a balancing parameter that adjusts the importance of the positive class,
#and γ is a focusing parameter that adjusts the rate at which easy examples are down-weighted.
#The Focal Loss is particularly useful when dealing with datasets where the number of negative examples
#is much larger than the number of positive examples, as it helps the model focus on the hard examples
#that are more informative for learning.
import os
import torch
from torch.utils.cpp_extension import load
import tempfile
import time

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

def run_benchmark(batch_sizes=[32, 64, 128, 256, 512, 1024, 2048, 
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]):
    print("\n----- Focal Loss Benchmark -----")
    print(f"{'Batch Size':<15}{'PyTorch (ms)':<15}{'CUDA (ms)':<15}{'Speedup':<10}")
    print("-" * 55)
    
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
        
        # Print results
        print(f"{batch_size:<15}{time_pytorch:<15.3f}{time_cuda:<15.3f}{speedup:<10.2f}x")
        
        # Verify results are close
        if abs(loss_pytorch - loss_cuda) > 1e-2:
            print(f"  Warning: Results differ! PyTorch: {loss_pytorch:.6f}, CUDA: {loss_cuda:.6f}")
    
    print("-" * 55)

# Run the benchmark
if __name__ == "__main__":
    run_benchmark()