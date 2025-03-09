import os
import torch
import torch.utils.cpp_extension 

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version (PyTorch):", torch.version.cuda)
print("CUDA Home:", torch.utils.cpp_extension.CUDA_HOME)
print("CUDA Libraries:", os.environ.get("PATH"))
