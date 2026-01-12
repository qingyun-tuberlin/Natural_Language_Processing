import torch
import pandas as pd
import sklearn
import transformers
import datasets
import platform
import seaborn as sns

print("="*30)
print("SYSTEM CHECK")
print("="*30)

# System Info
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Python Version:   {platform.python_version()}")

# Library Versions
print(f"Pandas:           {pd.__version__}")
print(f"Skikit-learn:     {sklearn.__version__}")
print(f"Transformers:     {transformers.__version__}")
print(f"Datasets:         {datasets.__version__}")
print(f"Seaborn:          {sns.__version__}")
print(f"PyTorch:          {torch.__version__}")

# Mac GPU Check
if torch.backends.mps.is_available():
    print("MPS (GPU):        Available ✅")
else:
    print("MPS (GPU):        Not Found ❌ (Training will be slow)")

print("="*30)