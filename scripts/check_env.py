"""Verify the environment. This is setup tooling, not assignment work.

Run: python scripts/check_env.py
"""

import platform
import sys

import timm
import torch
import torchvision

MODELS = [
    "resnet50.a1_in1k",
    "convnext_tiny.fb_in1k",
]

print(f"Python       {sys.version.split()[0]} ({platform.system()})")
print(f"PyTorch      {torch.__version__}")
print(f"torchvision  {torchvision.__version__}")
print(f"timm         {timm.__version__}")

if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU          {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    print(f"GPU test     {(torch.ones(2, device='cuda') * 2).sum().item() == 4}")
else:
    print("GPU          NOT AVAILABLE, running on CPU")

for name in MODELS:
    found = bool(timm.list_models(name, pretrained=True))
    print(f"timm model   {name}: {'ok' if found else 'NOT FOUND'}")
