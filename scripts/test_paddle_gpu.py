import paddle
paddle.device.set_device('gpu')
print("Paddle set_device('gpu') successful! Compiled with CUDA:", paddle.is_compiled_with_cuda())

from paddleocr import PaddleOCR
ocr = PaddleOCR(device='gpu')
print("PaddleOCR instantiated successfully on GPU!")
