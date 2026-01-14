"""
Prüfe CUDA/PyTorch GPUs und freien Festplattenspeicher.
"""

import torch, shutil, os

print("torch.cuda.is_available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
  print("Device name:", torch.cuda.get_device_name(0))
  print("Memory (GB):", torch.cuda.get_device_properties(0).total_memory/1024**3)
print("Free disk (GB):", shutil.disk_usage(".").free/1024**3)
