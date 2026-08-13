import torch
import easyocr

print("PyTorch CUDA:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

reader = easyocr.Reader(['vi', 'en'], gpu=True)
print("EasyOCR successfully loaded on GPU!")
