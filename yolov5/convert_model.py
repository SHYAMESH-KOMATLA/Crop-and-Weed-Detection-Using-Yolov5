import torch
import pathlib
from models.experimental import attempt_load

# Patch PosixPath to WindowsPath for Windows compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Load model
weights_path = 'best.pt'
device = torch.device('cpu')
model = attempt_load(weights_path, device=device)

# Save model in Windows-compatible format
torch.save(model, 'best_windows.pt')
print("✅ Model converted and saved as best_windows.pt")
