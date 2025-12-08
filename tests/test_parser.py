import pytest
from unittest.mock import MagicMock, patch
import torch
import sys
import os
import parser

def test_get_pdf_doi():
    """Test regex extraction of DOIs."""
    mock_doc = MagicMock()
    
    # Case 1: DOI in metadata
    mock_doc.metadata = {'doi': 'doi:10.1234/example'}
    assert parser.get_pdf_doi(mock_doc) == '10.1234/example'
    
    # Case 2: DOI in text
    mock_doc.metadata = {}
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Some text.\nDOI: 10.5678/test\nMore text."
    mock_doc.load_page.return_value = mock_page
    mock_doc.__len__.return_value = 1
    
    assert parser.get_pdf_doi(mock_doc) == '10.5678/test'

def test_extract_caption_near_image():
    """Test caption extraction logic."""
    img_bbox = MagicMock(x0=50, y0=100, x1=250, y1=300, y1_val=300)
    img_bbox.y1 = 300
    img_bbox.x0 = 50
    img_bbox.x1 = 250
    
    # Blocks: (x0, y0, x1, y1, text, ...)
    # Block 1: Right below image, starts with Fig
    b1 = (50, 310, 200, 350, "Fig 1. Correct Caption", 0)
    # Block 2: Further down
    b2 = (50, 400, 200, 450, "Fig 2. Wrong Caption", 1)
    # Block 3: Way above (should be ignored)
    b3 = (50, 50, 200, 80, "Title text", 2)
    
    blocks = [b1, b2, b3]
    
    caption = parser.extract_caption_near_image(None, img_bbox, blocks)
    assert caption == "Fig 1. Correct Caption"

def test_classifier_logic(mock_sentence_transformer):
    """Test ImageClassifier decision logic."""
    classifier = parser.ImageClassifier()
    
    with patch('torch.nn.functional.cosine_similarity') as mock_sim:
        # Case 1: Keep score > Reject score -> PASS
        mock_sim.side_effect = [
            torch.tensor([0.8]), # Keep
            torch.tensor([0.2]), # Reject
            torch.tensor([0.9])  # Visual (PASS)
        ]
        
        mock_image = MagicMock()
        mock_image.size = (500, 500)
        
        result = classifier.classify_image(mock_image)
        assert result['decision'] is True
        assert result['keep_score'] == 0.8
        
        # Case 2: Reject score > Strong Threshold -> FAIL
        mock_sim.side_effect = [
            torch.tensor([0.3]), # Keep
            torch.tensor([0.9]), # Reject
            torch.tensor([0.1])  # Visual
        ]
        result = classifier.classify_image(mock_image)
        assert result['decision'] is False
        assert "REJECTED" in result['reason']

    # Test Image Quality Checks
    small_image = MagicMock()
    small_image.size = (50, 50) # Too small
    result = classifier.classify_image(small_image)
    assert result['decision'] is False
    assert "too small" in result['reason']
