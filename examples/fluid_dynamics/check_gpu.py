#!/usr/bin/env python3
"""
Quick script to check GPU availability and test basic operations.
"""

import torch
import sys

print("=" * 60)
print("GPU Availability Check")
print("=" * 60)

# Basic checks
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ No CUDA GPUs detected!")
    print("\nPossible reasons:")
    print("  - No GPU on this machine")
    print("  - NVIDIA drivers not installed")
    print("  - PyTorch CPU-only version installed")
    print("\nTo install PyTorch with CUDA:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# GPU details
num_gpus = torch.cuda.device_count()
print(f"\n2. Number of GPUs: {num_gpus}")

print("\n3. GPU Details:")
for i in range(num_gpus):
    print(f"\n   GPU {i}:")
    print(f"   Name: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    print(f"   Multi-Processors: {props.multi_processor_count}")

# Current device
print(f"\n4. Current Device: {torch.cuda.current_device()}")
print(f"   Device Name: {torch.cuda.get_device_name()}")

# Quick speed test
print("\n5. Quick Speed Test:")
device = torch.device('cuda')
size = 10000

# CPU test
import time
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)
start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start
print(f"   CPU: {cpu_time:.4f} seconds")

# GPU test
x_gpu = torch.randn(size, size, device=device)
y_gpu = torch.randn(size, size, device=device)
torch.cuda.synchronize()  # Wait for GPU to be ready
start = time.time()
z_gpu = torch.matmul(x_gpu, y_gpu)
torch.cuda.synchronize()  # Wait for computation to finish
gpu_time = time.time() - start
print(f"   GPU: {gpu_time:.4f} seconds")
print(f"   Speedup: {cpu_time / gpu_time:.2f}x")

# Memory info
print("\n6. GPU Memory:")
for i in range(num_gpus):
    print(f"   GPU {i}:")
    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"     Allocated: {mem_allocated:.2f} GB")
    print(f"     Reserved:  {mem_reserved:.2f} GB")

print("\n" + "=" * 60)
print("✅ GPU check complete!")
print("=" * 60)

# Recommendation
if num_gpus > 1:
    print(f"\n🎉 You have {num_gpus} GPUs available!")
    print("   Consider using DistributedDataParallel for multi-GPU training.")
    print("\n   Example:")
    print(f"   python -m torch.distributed.launch --nproc_per_node={num_gpus} train.py")
elif num_gpus == 1:
    print("\n✅ You have 1 GPU available.")
    print("   Your current training code will work fine!")
