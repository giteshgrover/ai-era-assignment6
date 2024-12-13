import torch
import pytest
from src.model import MNISTModel

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Test number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"
    
def test_batch_norm_layers():
    model = MNISTModel()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should contain batch normalization layers"
    
def test_dropout_layers():
    model = MNISTModel()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should contain dropout layers"

def test_model_training():
    model = MNISTModel()
    test_input = torch.randn(4, 1, 28, 28)
    test_target = torch.randint(0, 10, (4,))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test if model can perform one training step
    output = model(test_input)
    loss = criterion(output, test_target)
    loss.backward()
    optimizer.step() 