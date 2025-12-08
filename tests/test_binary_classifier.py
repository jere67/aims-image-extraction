import pytest
from unittest.mock import MagicMock, patch
import torch
from torchvision import transforms
from PIL import Image
import os

from binary_classifier import PlotBinaryClassifier, PlotDataset

def test_dataset_loading(tmp_path):
    """Test PlotDataset __getitem__."""
    # Create valid image
    img_path = tmp_path / "dataset_test.png"
    Image.new('RGB', (100, 100), color='red').save(img_path)
    
    dataset = PlotDataset([str(img_path)], [0], transform=transforms.ToTensor())
    
    img_tensor, label = dataset[0]
    assert img_tensor.shape == (3, 100, 100)
    assert label == 0

    # Test error handling (missing file)
    dataset = PlotDataset(["/missing/file.png"], [1])
    img, lbl = dataset[0]
    assert img.shape == (3, 224, 224)
    assert lbl == 0

@patch('binary_classifier.models.efficientnet_b0')
def test_classifier_init(mock_efficientnet):
    """Test model initialization."""
    mock_model = MagicMock()
    mock_model.classifier = [MagicMock(), MagicMock()] # feature extractor + classifier
    mock_model.classifier[1].in_features = 1280 
    mock_efficientnet.return_value = mock_model
    
    clf = PlotBinaryClassifier(device='cpu')
    clf.build_model()
    
    assert clf.model is not None

@patch('binary_classifier.PlotBinaryClassifier.build_model')
def test_classify_image_inference(mock_build, tmp_path):
    """Test classify_image logic (mocking the model output)."""
    img_path = tmp_path / "infer.jpg"
    Image.new('RGB', (224, 224)).save(img_path)
    
    clf = PlotBinaryClassifier(device='cpu')
    clf.model = MagicMock()
    
    # Mock output logits: [Diagram_logit, Plot_logit]
    # Softmax([1.0, 2.0]) -> [0.26, 0.73] -> PLOT
    clf.model.return_value = torch.tensor([[1.0, 2.0]])
    
    result = clf.classify_image(str(img_path))
    
    assert result['is_plot'] is True
    assert result['plot_probability'] > 0.5
    assert "PLOT" in result['decision_reason']

    # Test Diagram case
    # Softmax([2.0, 1.0]) -> [0.73, 0.26] -> DIAGRAM
    clf.model.return_value = torch.tensor([[2.0, 1.0]])
    result = clf.classify_image(str(img_path))
    
    assert result['is_plot'] is False
    assert "DIAGRAM" in result['decision_reason']
