import logging
import numpy as np
from collections import defaultdict

# Assuming TrendDetector output might be an input or its components
# from graphrag.core.trend_detector import TrendType, TrendLifecycleStage # If needed

logger = logging.getLogger(__name__)

class TrendPredictor:
    def __init__(self, neo4j_conn=None):
        """
        Initializes the TrendPredictor.

        Args:
            neo4j_conn: Optional Neo4j connection for fetching creator data or trend history.
        """
        self.neo4j_conn = neo4j_conn
        logger.info("TrendPredictor initialized.")

    def perform_time_series_analysis(self, trend_temporal_data: dict) -> dict:
        """
        Performs time-series analysis on trend data to forecast future trajectory.

        Args:
            trend_temporal_data (dict): Data containing timestamps and frequencies.
                                        Example: {'cluster_id': 'xyz',
                                                  'timestamps': [t1, t2, ...],
                                                  'frequency_per_day': {'day1': f1, 'day2': f2, ...}}

        Returns:
            dict: Prediction results, e.g., {'predicted_next_period_frequency': float, 'confidence': float}
        """
        cluster_id = trend_temporal_data.get('cluster_id', 'unknown_cluster')
        logger.info(f"Performing time-series analysis for trend: {cluster_id}")

        frequency_data = trend_temporal_data.get('frequency_per_day', {})
        if not frequency_data or len(frequency_data) < 2:
            logger.warning(f"Not enough data points for time-series analysis on {cluster_id}.")
            return {"predicted_next_period_frequency": 0, "confidence": 0.0, "method": "insufficient_data"}

        # Placeholder: Simple moving average or linear extrapolation
        # Sort days to ensure order
        sorted_days = sorted(frequency_data.keys())
        recent_frequencies = [frequency_data[day] for day in sorted_days[-3:]] # Take last 3 periods

        if len(recent_frequencies) < 2:
            predicted_freq = recent_frequencies[0] if recent_frequencies else 0
            method = "last_value"
        else:
            # Simple average of the last few points as a naive prediction
            predicted_freq = sum(recent_frequencies) / len(recent_frequencies)
            method = f"average_last_{len(recent_frequencies)}_periods"

        # Confidence is very basic here
        confidence = 0.5 if len(frequency_data) > 3 else 0.2

        logger.debug(f"Time-series prediction for {cluster_id}: {predicted_freq:.2f} (method: {method})")
        return {"predicted_next_period_frequency": predicted_freq, "confidence": confidence, "method": method}

    def get_creator_influence_score(self, creator_id: str) -> float:
        """
        Retrieves or calculates an influence score for a given creator.

        Args:
            creator_id (str): The unique identifier for the creator.

        Returns:
            float: The influence score (e.g., 0.0 to 1.0).
        """
        logger.debug(f"Getting influence score for creator: {creator_id}")
        # Placeholder: Mock scores. In a real system, this might query a database
        # or use pre-calculated scores based on followers, engagement, etc.
        mock_scores = {
            "creatorA": 0.8,
            "creatorB": 0.6,
            "creatorC": 0.9,
            "creatorX": 0.7,
            "creatorY": 0.5,
            "creatorZ": 0.85,
        }
        score = mock_scores.get(creator_id, 0.4) # Default score for unknown creators
        logger.debug(f"Influence score for {creator_id}: {score}")
        return score

    def apply_creator_influence_weighting(self, prediction: dict, involved_creators: list[str]) -> dict:
        """
        Adjusts a trend prediction based on the influence of creators involved.

        Args:
            prediction (dict): The initial prediction from time_series_analysis.
            involved_creators (list[str]): A list of creator IDs associated with the trend.

        Returns:
            dict: The adjusted prediction.
        """
        logger.info(f"Applying creator influence weighting. Initial prediction: {prediction.get('predicted_next_period_frequency')}")
        if not involved_creators:
            logger.debug("No creators provided, returning original prediction.")
            return prediction

        avg_influence = 0
        if involved_creators:
            total_influence = sum(self.get_creator_influence_score(c_id) for c_id in involved_creators)
            avg_influence = total_influence / len(involved_creators)

        # Placeholder: Simple adjustment factor.
        # Example: if avg influence is high, slightly boost prediction or confidence.
        adjustment_factor = 1 + (avg_influence - 0.5) * 0.2 # Max +/- 10% adjustment based on avg influence

        adjusted_prediction_val = prediction.get("predicted_next_period_frequency", 0) * adjustment_factor
        adjusted_confidence = prediction.get("confidence", 0) * (1 + (avg_influence - 0.5) * 0.1) # Adjust confidence less
        adjusted_confidence = min(max(adjusted_confidence, 0.0), 1.0) # Clamp confidence

        logger.debug(f"Adjusted prediction: {adjusted_prediction_val:.2f}, Avg influence: {avg_influence:.2f}")

        prediction_copy = prediction.copy()
        prediction_copy["predicted_next_period_frequency_adj_influence"] = adjusted_prediction_val
        prediction_copy["creator_influence_avg_score"] = avg_influence
        prediction_copy["confidence_adj_influence"] = adjusted_confidence
        return prediction_copy

    def calculate_viral_coefficient(self, trend_adoption_data: dict) -> float:
        """
        Calculates a viral coefficient for a trend.

        Args:
            trend_adoption_data (dict): Data about trend adoption.
                                        Example: {'cluster_id': 'xyz',
                                                  'creator_adoption_count': 5,
                                                  'creators_by_timestamp': {ts1: [c1, c2], ts2: [c3]},
                                                  'initial_creator_count': 1,
                                                  'new_adopters_this_period': 2,
                                                  'total_exposure_potential': 10000 # e.g. sum of followers of adopters
                                                  }

        Returns:
            float: The calculated viral coefficient (K-factor).
        """
        # K = (Number of new adopters) / (Number of existing adopters in previous period)
        # This is a simplified K-factor. More complex versions exist.
        # Requires data on *new* adopters in a period vs. *existing* adopters.

        cluster_id = trend_adoption_data.get('cluster_id', 'unknown_cluster')
        logger.info(f"Calculating viral coefficient for trend: {cluster_id}")

        # Placeholder: This requires more detailed input than available in simple TrendDetector output.
        # Let's assume we get `new_adopters_this_period` and `existing_adopters_last_period`.
        new_adopters = trend_adoption_data.get('new_adopters_this_period', 0)
        existing_adopters_prev = trend_adoption_data.get('existing_adopters_last_period', 0)

        if existing_adopters_prev == 0:
            # If no existing adopters, K is undefined or could be considered infinite if new adopters > 0.
            # For simplicity, if new adopters exist from zero base, assign a high K or handle as special case.
            k_factor = float(new_adopters) if new_adopters > 0 else 0.0
        else:
            k_factor = float(new_adopters) / existing_adopters_prev

        logger.debug(f"Viral coefficient for {cluster_id}: {k_factor:.2f} (new: {new_adopters}, existing_prev: {existing_adopters_prev})")
        return k_factor

    def generate_early_warning(self, trend_id: str, prediction_data: dict, velocity: float, viral_coeff: float) -> dict | None:
        """
        Generates an early warning for emerging trends based on various signals.

        Args:
            trend_id (str): Identifier for the trend.
            prediction_data (dict): Output from time-series analysis (and influence weighting).
            velocity (float): Current velocity of the trend.
            viral_coeff (float): Viral coefficient.

        Returns:
            dict: An early warning message or None if no significant warning.
                  Example: {'trend_id': trend_id, 'warning_level': 'high', 'reasons': [...]}
        """
        logger.info(f"Generating early warning assessment for trend: {trend_id}")

        warning_level = "low"
        reasons = []

        predicted_freq = prediction_data.get('predicted_next_period_frequency_adj_influence',
                                           prediction_data.get('predicted_next_period_frequency', 0))

        # Example rules for early warning (these are arbitrary)
        if velocity > 2.0 and predicted_freq > 5: # Strong positive velocity and decent predicted volume
            warning_level = "medium"
            reasons.append(f"Positive velocity ({velocity:.2f}) and predicted frequency ({predicted_freq:.2f}).")

        if viral_coeff > 1.0: # K > 1 suggests viral growth
            if warning_level == "medium":
                warning_level = "high"
            else:
                warning_level = "medium"
            reasons.append(f"Viral coefficient is {viral_coeff:.2f}, indicating potential for rapid spread.")

        if velocity > 3.0 and viral_coeff > 1.5 and predicted_freq > 10:
            warning_level = "critical"
            reasons.append("Strong combination of high velocity, high virality, and growing predicted frequency.")

        if not reasons:
            logger.debug(f"No significant early warning signals for trend {trend_id}.")
            return None

        warning_output = {
            "trend_id": trend_id,
            "warning_level": warning_level,
            "predicted_next_period_frequency": predicted_freq,
            "current_velocity": velocity,
            "viral_coefficient": viral_coeff,
            "reasons": reasons
        }
        logger.info(f"Early warning for {trend_id}: Level - {warning_level}, Reasons: {reasons}")
        return warning_output

    def predict_full_trend_trajectory(self, trend_data_from_detector: dict) -> dict:
        """
        Orchestrates the prediction process for a single trend identified by TrendDetector.

        Args:
            trend_data_from_detector (dict): A dictionary representing a single trend,
                                             as output by TrendDetector. Expected to contain:
                                             'cluster_id_proxy' (or a real cluster_id),
                                             'temporal_frequency', 'velocity',
                                             'example_creators', etc.

        Returns:
            dict: A dictionary containing all prediction insights for the trend.
        """
        trend_id = trend_data_from_detector.get('cluster_id_proxy',
                                              trend_data_from_detector.get('cluster_id', 'unknown_trend'))
        logger.info(f"--- Predicting full trajectory for trend: {trend_id} ---")

        # 1. Time-series analysis
        temporal_data = {
            "cluster_id": trend_id,
            "frequency_per_day": trend_data_from_detector.get('temporal_frequency', {})
            # Add 'timestamps' if available and needed by time_series_analysis
        }
        ts_prediction = self.perform_time_series_analysis(temporal_data)

        # 2. Creator influence weighting
        creators = trend_data_from_detector.get('example_creators', []) # Use 'creators' if that's the key
        influence_weighted_prediction = self.apply_creator_influence_weighting(ts_prediction, creators)

        # 3. Viral coefficient
        # This needs more detailed input, so we'll use mock inputs for now.
        # In a real scenario, this data would need to be fetched or calculated based on historical adoption.
        mock_adoption_data_for_viral_coeff = {
            'cluster_id': trend_id,
            # Mock: new adopters are half of current total, if current total > 1
            'new_adopters_this_period': trend_data_from_detector.get('adoption_count', 0) // 2
                                       if trend_data_from_detector.get('adoption_count', 0) > 1 else 0,
            # Mock: existing adopters were a third of current total (plus 1 to avoid zero division)
            'existing_adopters_last_period': trend_data_from_detector.get('adoption_count', 0) // 3 + 1
        }
        viral_coeff = self.calculate_viral_coefficient(mock_adoption_data_for_viral_coeff)

        # 4. Early warning system
        current_velocity = trend_data_from_detector.get('velocity', 0.0)
        warning = self.generate_early_warning(trend_id, influence_weighted_prediction, current_velocity, viral_coeff)

        full_prediction_output = {
            "trend_id": trend_id,
            "time_series_prediction": ts_prediction,
            "creator_influenced_prediction": influence_weighted_prediction,
            "viral_coefficient": viral_coeff,
            "current_velocity_from_detector": current_velocity,
            "early_warning_assessment": warning,
            "original_trend_data_summary": { # Keep summary to avoid too much data
                "type": trend_data_from_detector.get("type"),
                "lifecycle_stage": trend_data_from_detector.get("lifecycle_stage"),
                "confidence_score": trend_data_from_detector.get("confidence_score")
            }
        }

        logger.info(f"Full prediction for {trend_id} compiled.")
        return full_prediction_output


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting TrendPredictor standalone example.")

    predictor = TrendPredictor()

    # Mock data similar to what TrendDetector might output for a single trend
    mock_trend_from_detector = {
        "cluster_id_proxy": "trend_abc_123",
        "velocity": 2.5,
        "adoption_count": 10,
        "total_occurrences": 75,
        "example_creators": ["creatorX", "creatorY", "creatorZ"],
        "temporal_frequency": { # Mock frequency per day for ~5 days
            "day_17000": 5,  # Approx "5 days ago" if today is day_17005
            "day_17001": 8,
            "day_17002": 12,
            "day_17003": 15,
            "day_17004": 20, # Current day's observation
        },
        "type": "AESTHETIC",
        "confidence_score": 0.85,
        "lifecycle_stage": "PEAKING"
    }

    logger.info(f"
--- Mock Trend Data from Detector for '{mock_trend_from_detector['cluster_id_proxy']}' ---")
    # import json # Not strictly needed for this print format
    # logger.info(json.dumps(mock_trend_from_detector, indent=2, default=str))


    # Perform full prediction
    final_prediction = predictor.predict_full_trend_trajectory(mock_trend_from_detector)

    logger.info(f"
--- Full Prediction Output for '{final_prediction['trend_id']}' ---")
    logger.info(f"  Trend ID: {final_prediction['trend_id']}")
    logger.info(f"  Current Velocity (from Detector): {final_prediction['current_velocity_from_detector']}")
    logger.info(f"  Time Series Prediction (raw): {final_prediction['time_series_prediction']}")
    logger.info(f"  Creator Influenced Prediction: {final_prediction['creator_influenced_prediction']}")
    logger.info(f"  Viral Coefficient: {final_prediction['viral_coefficient']:.2f}")
    if final_prediction['early_warning_assessment']:
        logger.info(f"  Early Warning Level: {final_prediction['early_warning_assessment']['warning_level']}")
        logger.info(f"  Warning Reasons: {final_prediction['early_warning_assessment']['reasons']}")
    else:
        logger.info("  Early Warning Level: None")
    logger.info(f"  Original Trend Summary: {final_prediction['original_trend_data_summary']}")


    # Example for another trend with different characteristics (e.g., declining)
    mock_declining_trend = {
        "cluster_id_proxy": "trend_declining_001",
        "velocity": -1.5,
        "adoption_count": 30,
        "total_occurrences": 200,
        "example_creators": ["creatorA", "creatorB"],
        "temporal_frequency": {
            "day_16995": 25,
            "day_16996": 20,
            "day_16997": 15,
            "day_16998": 12,
            "day_16999": 10,
        },
        "type": "FORMAT",
        "confidence_score": 0.70,
        "lifecycle_stage": "DECLINING"
    }
    logger.info(f"
--- Mock Trend Data from Detector for '{mock_declining_trend['cluster_id_proxy']}' ---")
    final_declining_prediction = predictor.predict_full_trend_trajectory(mock_declining_trend)
    logger.info(f"
--- Full Prediction Output for '{final_declining_prediction['trend_id']}' ---")
    logger.info(f"  Trend ID: {final_declining_prediction['trend_id']}")
    logger.info(f"  Creator Influenced Prediction: {final_declining_prediction['creator_influenced_prediction']}")
    logger.info(f"  Viral Coefficient: {final_declining_prediction['viral_coefficient']:.2f}")
    if final_declining_prediction['early_warning_assessment']:
        logger.info(f"  Early Warning Level: {final_declining_prediction['early_warning_assessment']['warning_level']}")
    else:
        logger.info("  Early Warning Level: None")
    logger.info(f"  Original Trend Summary: {final_declining_prediction['original_trend_data_summary']}")

    logger.info("
TrendPredictor standalone example finished.")
