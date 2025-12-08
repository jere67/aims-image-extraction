import pytest
from unittest.mock import MagicMock, patch
import sys
import pipeline

@patch('pipeline.run_stage1_parser')
@patch('pipeline.run_stage2_gpt_4_1_nano')
@patch('pipeline.run_stage3_gpt_5_nano')
def test_pipeline_flow(mock_stage3, mock_stage2, mock_stage1):
    """Test that main calls stages correctly based on start-stage arg."""
    
    # Setup return values for metrics to avoid TypeError comparisons
    mock_stage1.return_value = {'images_approved': 5, 'images_input': 10, 'duration': 1.0}
    mock_stage2.return_value = {'images_approved': 3, 'images_input': 5, 'api_calls': 1, 'total_tokens': 100, 'cost': 0.01, 'duration': 1.0}
    mock_stage3.return_value = {'images_approved': 1, 'images_input': 3, 'api_calls': 1, 'total_tokens': 100, 'cost': 0.01, 'duration': 1.0}

    # Simulate CLI args
    with patch.object(sys, 'argv', ['pipeline.py', '--start-stage', '1']):
        pipeline.main()
        mock_stage1.assert_called_once()
        mock_stage2.assert_called_once()
        mock_stage3.assert_called_once()

    # Reset
    mock_stage1.reset_mock()
    mock_stage2.reset_mock()
    mock_stage3.reset_mock()
    # Mock mocks are reset, but return_value is preserved? It's safer to re-set or ensure it's still there.
    mock_stage1.return_value = {'images_approved': 5, 'images_input': 10, 'duration': 1.0}

    # Start from Stage 2
    with patch.object(sys, 'argv', ['pipeline.py', '--start-stage', '2']):
        pipeline.main()
        mock_stage1.assert_not_called()
        mock_stage2.assert_called_once()
        mock_stage3.assert_called_once()
