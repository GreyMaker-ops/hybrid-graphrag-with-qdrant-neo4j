import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import json

from graphrag.core.visual_analyzer import VisualAnalyzer

class TestVisualAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_temp_dir = "test_temp_analyzer_frames"
        self.dummy_frame_path = os.path.join(self.test_temp_dir, "dummy_frame.jpg")

        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
        os.makedirs(self.test_temp_dir, exist_ok=True)

        with open(self.dummy_frame_path, 'w') as f:
            f.write("dummy frame content")

        self.visual_analyzer = VisualAnalyzer()

    def tearDown(self):
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)

    def test_analyze_frame_rf_detr_placeholder(self):
        result = self.visual_analyzer.analyze_frame_rf_detr(self.dummy_frame_path)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0) # Expecting mock objects
        for item in result:
            self.assertIn('type', item)
            self.assertIn('description', item)
            self.assertIn('confidence', item)
            self.assertIn('bounding_box', item)

        # Test non-existent file
        result_no_file = self.visual_analyzer.analyze_frame_rf_detr("non_existent.jpg")
        self.assertEqual(result_no_file[0]['type'], 'error')


    def test_describe_scene_blip2_placeholder(self):
        result = self.visual_analyzer.describe_scene_blip2(self.dummy_frame_path)
        self.assertIsInstance(result, dict)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'scene_description')
        self.assertIn('description', result)

        # Test non-existent file
        result_no_file = self.visual_analyzer.describe_scene_blip2("non_existent.jpg")
        self.assertEqual(result_no_file['type'], 'error')

    def test_enhance_analysis_gemini_placeholder(self):
        existing_analysis_mock = {
            "detected_objects": [{'description': 'mock_object'}]
        }
        result = self.visual_analyzer.enhance_analysis_gemini(self.dummy_frame_path, existing_analysis_mock)
        self.assertIsInstance(result, dict)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'enhanced_description')
        self.assertIn('description', result)
        self.assertIn('additional_tags', result)

        # Test non-existent file
        result_no_file = self.visual_analyzer.enhance_analysis_gemini("non_existent.jpg", {})
        self.assertEqual(result_no_file['type'], 'error')


    def test_create_unified_output(self):
        rf_results = [{'type': 'object', 'description': 'cat'}]
        blip_desc = {'type': 'scene_description', 'description': 'A cat.'}
        gemini_enhance = {'type': 'enhanced_description', 'description': 'A fluffy cat.'}

        unified = self.visual_analyzer.create_unified_output(
            self.dummy_frame_path, rf_results, blip_desc, gemini_enhance
        )
        self.assertEqual(unified['frame_path'], self.dummy_frame_path)
        self.assertEqual(unified['detected_objects'], rf_results)
        self.assertEqual(unified['scene_description'], blip_desc)
        self.assertEqual(unified['enhanced_analysis'], gemini_enhance)
        self.assertEqual(len(unified['errors']), 0)

        # Test with error in one of the inputs
        rf_results_error = [{'type': 'error', 'description': 'RF DETR failed'}]
        unified_with_error = self.visual_analyzer.create_unified_output(
            self.dummy_frame_path, rf_results_error, blip_desc, gemini_enhance
        )
        self.assertIn("RF-DETR: RF DETR failed", unified_with_error['errors'])


    @patch('graphrag.core.visual_analyzer.VisualAnalyzer.analyze_frame_rf_detr')
    @patch('graphrag.core.visual_analyzer.VisualAnalyzer.describe_scene_blip2')
    @patch('graphrag.core.visual_analyzer.VisualAnalyzer.enhance_analysis_gemini')
    @patch('graphrag.core.visual_analyzer.VisualAnalyzer.create_unified_output')
    def test_analyze_frame_orchestration(self, mock_create_unified, mock_enhance, mock_describe, mock_rf_detr):
        mock_rf_data = [{'type': 'object', 'description': 'mock_cat'}]
        mock_blip_data = {'type': 'scene_description', 'description': 'A mock scene.'}
        mock_gemini_data = {'type': 'enhanced_description', 'description': 'Enhanced mock.'}

        mock_rf_detr.return_value = mock_rf_data
        mock_describe.return_value = mock_blip_data
        mock_enhance.return_value = mock_gemini_data

        # To check what create_unified_output was called with
        capture_unified_args = {}
        def side_effect_create_unified(*args, **kwargs):
            # args[0] is frame_path, args[1] is rf_results etc.
            capture_unified_args['frame_path'] = args[0]
            capture_unified_args['rf_results'] = args[1]
            capture_unified_args['blip_description'] = args[2]
            capture_unified_args['gemini_enhancements'] = args[3]
            return "unified_output_mock" # Return value for analyze_frame
        mock_create_unified.side_effect = side_effect_create_unified

        result = self.visual_analyzer.analyze_frame(self.dummy_frame_path, use_gemini=True)

        mock_rf_detr.assert_called_once_with(self.dummy_frame_path)
        mock_describe.assert_called_once_with(self.dummy_frame_path)

        # Construct expected existing_analysis for Gemini call
        expected_existing_analysis_for_gemini = {
            "detected_objects": mock_rf_data,
            "scene_description": mock_blip_data
        }
        mock_enhance.assert_called_once_with(self.dummy_frame_path, expected_existing_analysis_for_gemini)

        mock_create_unified.assert_called_once()
        self.assertEqual(capture_unified_args['frame_path'], self.dummy_frame_path)
        self.assertEqual(capture_unified_args['rf_results'], mock_rf_data)
        self.assertEqual(capture_unified_args['blip_description'], mock_blip_data)
        self.assertEqual(capture_unified_args['gemini_enhancements'], mock_gemini_data)
        self.assertEqual(result, "unified_output_mock")

    @patch('graphrag.core.visual_analyzer.VisualAnalyzer.analyze_frame_rf_detr')
    def test_analyze_frame_error_handling_in_sub_call(self, mock_rf_detr):
        mock_rf_detr.side_effect = Exception("RF-DETR major failure")

        # Expected: analyze_frame should catch the exception and report it in errors.
        # Other methods (blip, gemini if enabled) should still be called if they don't depend on prior failing steps.
        # However, our current placeholders don't have such dependencies.

        with patch.object(self.visual_analyzer.logger, 'error') as mock_logger_error:
            result = self.visual_analyzer.analyze_frame(self.dummy_frame_path)

            self.assertIn('errors', result)
            self.assertTrue(any("RF-DETR analysis failed: RF-DETR major failure" in e for e in result['errors']))
            mock_logger_error.assert_any_call(f"Error during RF-DETR analysis for {self.dummy_frame_path}: RF-DETR major failure")

    def test_analyze_frame_no_gemini(self):
        with patch('graphrag.core.visual_analyzer.VisualAnalyzer.enhance_analysis_gemini') as mock_enhance:
            self.visual_analyzer.analyze_frame(self.dummy_frame_path, use_gemini=False)
            mock_enhance.assert_not_called()

if __name__ == '__main__':
    unittest.main()
