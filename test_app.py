import torch
from app import ImprovedCNN

def test_model_forward():
    model = ImprovedCNN()
    x = torch.randn(1, 3, 32, 32)  # یک ورودی تصادفی
    y = model(x)
    assert y.shape == (1, 10)  # باید ۱۰ کلاس بده
