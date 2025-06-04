import unittest
from unittest.mock import MagicMock, patch

# Module to be tested
from graphrag.core.trend_predictor import TrendPredictor

class TestTrendPredictor(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.mock_neo4j_conn = MagicMock() # If TrendPredictor uses Neo4j directly
        self.trend_predictor = TrendPredictor(neo4j_conn=self.mock_neo4j_conn)

    def test_initialization(self):
        """Test that TrendPredictor initializes correctly."""
        self.assertIsNotNone(self.trend_predictor)
        # self.assertEqual(self.trend_predictor.neo4j_conn, self.mock_neo4j_conn) # If it stores it

    def test_perform_time_series_analysis(self):
        """Test basic time-series analysis (placeholder logic)."""
        # Test with sufficient data
        temporal_data_sufficient = {
            "cluster_id": "trend1",
            "frequency_per_day": {"d1": 10, "d2": 12, "d3": 14, "d4": 16}
        }
        prediction = self.trend_predictor.perform_time_series_analysis(temporal_data_sufficient)
        self.assertIn("predicted_next_period_frequency", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("method", prediction)
        # Current mock logic: avg of last 3: (12+14+16)/3 = 14
        self.assertAlmostEqual(prediction["predicted_next_period_frequency"], 14.0)
        self.assertEqual(prediction["method"], "average_last_3_periods")
        self.assertEqual(prediction["confidence"], 0.5)


        # Test with minimal data (2 points)
        temporal_data_minimal = {
            "cluster_id": "trend2",
            "frequency_per_day": {"d1": 5, "d2": 7}
        }
        prediction_minimal = self.trend_predictor.perform_time_series_analysis(temporal_data_minimal)
        # Avg of last 2: (5+7)/2 = 6
        self.assertAlmostEqual(prediction_minimal["predicted_next_period_frequency"], 6.0)
        self.assertEqual(prediction_minimal["method"], "average_last_2_periods")
        self.assertEqual(prediction_minimal["confidence"], 0.2) # Confidence is lower for less data

        # Test with insufficient data (1 point)
        temporal_data_insufficient = {"cluster_id": "trend3", "frequency_per_day": {"d1": 5}}
        prediction_insufficient = self.trend_predictor.perform_time_series_analysis(temporal_data_insufficient)
        self.assertEqual(prediction_insufficient["predicted_next_period_frequency"], 5) # Last value
        self.assertEqual(prediction_insufficient["method"], "last_value")
        self.assertEqual(prediction_insufficient["confidence"], 0.2)


        # Test with no data
        temporal_data_empty = {"cluster_id": "trend4", "frequency_per_day": {}}
        prediction_empty = self.trend_predictor.perform_time_series_analysis(temporal_data_empty)
        self.assertEqual(prediction_empty["predicted_next_period_frequency"], 0)
        self.assertEqual(prediction_empty["method"], "insufficient_data")

    def test_get_creator_influence_score(self):
        """Test fetching creator influence score (mocked)."""
        score_A = self.trend_predictor.get_creator_influence_score("creatorA")
        self.assertEqual(score_A, 0.8) # From mock_scores in TrendPredictor

        score_unknown = self.trend_predictor.get_creator_influence_score("unknown_creator_id")
        self.assertEqual(score_unknown, 0.4) # Default score

    def test_apply_creator_influence_weighting(self):
        """Test applying creator influence weighting to a prediction."""
        initial_prediction = {"predicted_next_period_frequency": 10.0, "confidence": 0.6, "method":"some_method"}
        creators = ["creatorA", "creatorC"] # Scores 0.8, 0.9. Avg = 0.85

        weighted_pred = self.trend_predictor.apply_creator_influence_weighting(initial_prediction, creators)

        # avg_influence = 0.85
        # adjustment_factor = 1 + (0.85 - 0.5) * 0.2 = 1 + 0.35 * 0.2 = 1 + 0.07 = 1.07
        # adjusted_val = 10.0 * 1.07 = 10.7
        # adjusted_conf = 0.6 * (1 + (0.85 - 0.5) * 0.1) = 0.6 * (1 + 0.035) = 0.6 * 1.035 = 0.621

        self.assertAlmostEqual(weighted_pred["predicted_next_period_frequency_adj_influence"], 10.7)
        self.assertAlmostEqual(weighted_pred["creator_influence_avg_score"], 0.85)
        self.assertAlmostEqual(weighted_pred["confidence_adj_influence"], 0.621)
        self.assertEqual(weighted_pred["method"], "some_method") # Original method should persist

        # Test with no creators
        weighted_pred_no_creators = self.trend_predictor.apply_creator_influence_weighting(initial_prediction, [])
        self.assertEqual(weighted_pred_no_creators, initial_prediction) # Should return original

    def test_calculate_viral_coefficient(self):
        """Test viral coefficient calculation (placeholder logic)."""
        # K = new / existing_prev
        adoption_data1 = {'cluster_id': 't1', 'new_adopters_this_period': 10, 'existing_adopters_last_period': 5}
        k1 = self.trend_predictor.calculate_viral_coefficient(adoption_data1)
        self.assertAlmostEqual(k1, 2.0) # 10 / 5

        adoption_data2 = {'cluster_id': 't2', 'new_adopters_this_period': 5, 'existing_adopters_last_period': 10}
        k2 = self.trend_predictor.calculate_viral_coefficient(adoption_data2)
        self.assertAlmostEqual(k2, 0.5) # 5 / 10

        adoption_data_zero_prev = {'cluster_id': 't3', 'new_adopters_this_period': 5, 'existing_adopters_last_period': 0}
        k3 = self.trend_predictor.calculate_viral_coefficient(adoption_data_zero_prev)
        self.assertAlmostEqual(k3, 5.0) # Special handling: new adopters from zero base

        adoption_data_zero_all = {'cluster_id': 't4', 'new_adopters_this_period': 0, 'existing_adopters_last_period': 0}
        k4 = self.trend_predictor.calculate_viral_coefficient(adoption_data_zero_all)
        self.assertAlmostEqual(k4, 0.0)


    def test_generate_early_warning(self):
        """Test early warning generation logic."""
        trend_id = "warning_trend"

        # High warning scenario
        pred_data_high = {"predicted_next_period_frequency": 15.0, "predicted_next_period_frequency_adj_influence": 18.0}
        velocity_high = 3.5
        viral_coeff_high = 1.8
        warning_high = self.trend_predictor.generate_early_warning(trend_id, pred_data_high, velocity_high, viral_coeff_high)
        self.assertIsNotNone(warning_high)
        self.assertEqual(warning_high["trend_id"], trend_id)
        self.assertEqual(warning_high["warning_level"], "critical") # Based on current rules
        self.assertTrue(len(warning_high["reasons"]) >= 2)

        # Medium warning (e.g., only velocity is high)
        pred_data_med = {"predicted_next_period_frequency": 8.0}
        velocity_med = 2.5
        viral_coeff_med = 0.8
        warning_med = self.trend_predictor.generate_early_warning(trend_id, pred_data_med, velocity_med, viral_coeff_med)
        self.assertIsNotNone(warning_med)
        self.assertEqual(warning_med["warning_level"], "medium")
        self.assertTrue("Positive velocity" in warning_med["reasons"][0])

        # Low/No warning scenario
        pred_data_low = {"predicted_next_period_frequency": 2.0}
        velocity_low = 0.5
        viral_coeff_low = 0.5
        warning_low = self.trend_predictor.generate_early_warning(trend_id, pred_data_low, velocity_low, viral_coeff_low)
        self.assertIsNone(warning_low) # No reasons triggered

    def test_predict_full_trend_trajectory_orchestration(self):
        """Test the main orchestration method predict_full_trend_trajectory."""
        mock_input_trend_data = {
            "cluster_id_proxy": "full_test_trend_001",
            "velocity": 1.5,
            "adoption_count": 5,
            "example_creators": ["creatorX", "creatorY"], # Scores 0.7, 0.5. Avg = 0.6
            "temporal_frequency": {"d1": 2, "d2": 4, "d3": 6} # Avg last 3 = 4
        }

        # Patch the individual methods to assert they are called and to control their output
        with patch.object(self.trend_predictor, 'perform_time_series_analysis') as mock_tsa:
            tsa_return = {"predicted_next_period_frequency": 4.0, "confidence": 0.5, "method":"mock_tsa"}
            mock_tsa.return_value = tsa_return

            with patch.object(self.trend_predictor, 'apply_creator_influence_weighting') as mock_influence:
                influence_return = {"predicted_next_period_frequency_adj_influence": 4.2, "creator_influence_avg_score": 0.6, "confidence_adj_influence": 0.55}
                influence_return.update(tsa_return) # ensure original fields are there too
                mock_influence.return_value = influence_return

                with patch.object(self.trend_predictor, 'calculate_viral_coefficient') as mock_viral:
                    mock_viral.return_value = 0.8 # K-factor

                    with patch.object(self.trend_predictor, 'generate_early_warning') as mock_warning:
                        warning_return = {"trend_id": "full_test_trend_001", "warning_level": "low", "reasons": ["Mock reason"]}
                        mock_warning.return_value = warning_return

                        full_prediction = self.trend_predictor.predict_full_trend_trajectory(mock_input_trend_data)

                        mock_tsa.assert_called_once()
                        # Check that the input to TSA was correct based on mock_input_trend_data
                        self.assertEqual(mock_tsa.call_args[0][0]['frequency_per_day'],
                                         mock_input_trend_data['temporal_frequency'])

                        mock_influence.assert_called_once_with(tsa_return, mock_input_trend_data['example_creators'])

                        # Check input to viral coefficient (it's derived from mock_input_trend_data)
                        mock_viral.assert_called_once()
                        viral_call_args = mock_viral.call_args[0][0]
                        self.assertEqual(viral_call_args['new_adopters_this_period'], 2) # 5 // 2
                        self.assertEqual(viral_call_args['existing_adopters_last_period'], 2) # 5 // 3 + 1

                        mock_warning.assert_called_once_with(
                            "full_test_trend_001",
                            influence_return,
                            mock_input_trend_data['velocity'],
                            0.8
                        )

                        self.assertEqual(full_prediction["trend_id"], "full_test_trend_001")
                        self.assertEqual(full_prediction["time_series_prediction"], tsa_return)
                        self.assertEqual(full_prediction["creator_influenced_prediction"], influence_return)
                        self.assertEqual(full_prediction["viral_coefficient"], 0.8)
                        self.assertEqual(full_prediction["early_warning_assessment"], warning_return)
                        self.assertIn("original_trend_data_summary", full_prediction)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
