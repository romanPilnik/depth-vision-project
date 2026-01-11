import torch
import cv2
import transformers

print(f"PyTorch Version: {torch.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Transformers Version: {transformers.__version__}")

# Check Hardware Acceleration
if torch.cuda.is_available():
    print("Device: CUDA (NVIDIA GPU) is ACTIVE")
elif torch.backends.mps.is_available():
    print("Device: MPS (Apple Silicon GPU) is ACTIVE")
else:
    print("Device: CPU")