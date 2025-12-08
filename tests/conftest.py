import pytest
from unittest.mock import MagicMock
import sys
import os

# Set dummy env vars to prevent base_classifier from exiting
os.environ['OPENAI_API_KEY'] = 'dummy'
os.environ['API_VERSION'] = 'dummy'
os.environ['OPENAI_API_BASE'] = 'dummy'
os.environ['OPENAI_ORGANIZATION'] = 'dummy'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

mock_fitz = MagicMock()
class MockRect:
    def __init__(self, *args):
        if len(args) == 4:
            self.x0, self.y0, self.x1, self.y1 = args
        else:
            self.x0, self.y0, self.x1, self.y1 = 0, 0, 0, 0
    def __getitem__(self, idx):
        return [self.x0, self.y0, self.x1, self.y1][idx]

mock_fitz.Rect = MockRect
sys.modules['fitz'] = mock_fitz

mock_openai = MagicMock()
sys.modules['openai'] = mock_openai

mock_st = MagicMock()
sys.modules['sentence_transformers'] = mock_st

mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_plotly = MagicMock()
sys.modules['plotly'] = mock_plotly
sys.modules['plotly.graph_objects'] = mock_plotly

mock_kaleido = MagicMock()
sys.modules['kaleido'] = mock_kaleido


@pytest.fixture
def mock_pdf_doc():
    """Mocks a fitz.Document for parser testing."""
    mock_doc = MagicMock()
    mock_page = MagicMock()
    
    # Mock page text blocks for caption extraction
    # Format: (x0, y0, x1, y1, "text", block_no, block_type)
    mock_page.get_text.return_value = [
        (50, 500, 200, 520, "Fig 1. A test caption.", 0, 0),
        (50, 600, 200, 620, "Just some random text.", 1, 0)
    ]
    
    # Mock image extraction
    # (xref, smask, width, height, bpc, colorspace, ...)
    mock_page.get_images.return_value = [(1, 0, 200, 200, 8, 'DeviceRGB', '', '', '')]
    mock_page.get_image_bbox.return_value = MagicMock(x0=50, y0=100, x1=250, y1=300)
    
    mock_doc.load_page.return_value = mock_page
    mock_doc.__len__.return_value = 1
    mock_doc.extract_image.return_value = {
        "image": b"fake_image_bytes",
        "ext": "png"
    }
    
    return mock_doc

@pytest.fixture
def mock_sentence_transformer(mocker):
    """Mocks SentenceTransformer to avoid loading model."""
    mock_cls = mocker.patch('parser.SentenceTransformer')
    mock_instance = mock_cls.return_value
    
    import torch
    mock_instance.encode.return_value = torch.rand(1, 512)
    return mock_instance

@pytest.fixture
def mock_openai_client(mocker):
    """Mocks AzureOpenAI client."""
    mock_client = mocker.patch('gpt_classifiers.base_classifier.client')
    return mock_client

@pytest.fixture
def mock_efficientnet(mocker):
    """Mocks EfficientNet dependencies."""
    mocker.patch('torchvision.models.efficientnet_b0')
    mocker.patch('torch.load')
    return True
