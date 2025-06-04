import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import time # For generating timestamps in test data

# Modules to be tested
from graphrag.core.trend_detector import TrendDetector, TrendType, TrendLifecycleStage
# Mocked dependencies
from graphrag.core.visual_analyzer import VisualAnalyzer
from graphrag.connectors.qdrant_connection import QdrantConnection


class TestTrendDetector(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.mock_visual_analyzer = MagicMock(spec=VisualAnalyzer)
        self.mock_qdrant_conn = MagicMock(spec=QdrantConnection)
        self.mock_neo4j_conn = MagicMock() # General mock if needed

        # Configure mock VisualAnalyzer's generate_embedding method if it's expected by TrendDetector
        if hasattr(VisualAnalyzer, 'generate_embedding'):
            self.mock_visual_analyzer.generate_embedding.return_value = np.random.rand(512).astype(np.float32)

        # Configure mock QdrantConnection's methods
        self.mock_qdrant_conn.create_collection.return_value = True
        self.mock_qdrant_conn.upsert_vectors.return_value = True
        self.mock_qdrant_conn.search.return_value = [] # Default search result

        self.trend_detector = TrendDetector(
            visual_analyzer=self.mock_visual_analyzer,
            qdrant_conn=self.mock_qdrant_conn,
            neo4j_conn=self.mock_neo4j_conn
        )
        # Ensure the collection is "created" in the mock if TrendDetector tries to do so
        # or if it's a prerequisite for other operations.
        # The TrendDetector's __init__ currently doesn't create it, but identify_trends_from_data might.
        # For safety, we can simulate its existence:
        self.trend_detector.qdrant_collection_name = "test_visual_trends"
        self.mock_qdrant_conn.collections = {self.trend_detector.qdrant_collection_name: True}


    def test_initialization(self):
        """Test that TrendDetector initializes correctly."""
        self.assertIsNotNone(self.trend_detector.visual_analyzer)
        self.assertIsNotNone(self.trend_detector.qdrant_conn)
        self.assertEqual(self.trend_detector.qdrant_collection_name, "test_visual_trends")

    def test_generate_visual_embeddings_with_va_method(self):
        """Test embedding generation when VisualAnalyzer has generate_embedding."""
        # Ensure VA mock has the method
        self.mock_visual_analyzer.generate_embedding = MagicMock(return_value=np.random.rand(512).astype(np.float32))

        embedding = self.trend_detector._generate_visual_embeddings("dummy_frame.jpg")
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (512,))
        self.mock_visual_analyzer.generate_embedding.assert_called_once_with("dummy_frame.jpg")

    def test_generate_visual_embeddings_fallback(self):
        """Test embedding generation fallback when VisualAnalyzer lacks generate_embedding."""
        # Ensure VA mock does NOT have the method for this test
        if hasattr(self.mock_visual_analyzer, 'generate_embedding'):
            del self.mock_visual_analyzer.generate_embedding
            # Or mock it to raise AttributeError
            # self.mock_visual_analyzer.generate_embedding = MagicMock(side_effect=AttributeError)


        embedding = self.trend_detector._generate_visual_embeddings("dummy_frame.jpg")
        self.assertIsNotNone(embedding)
        self.assertEqual(embedding.shape, (512,)) # Fallback random embedding

    def test_process_frame_for_visual_similarity(self):
        """Test processing a single frame for visual similarity."""
        frame_id = "test_frame_001"
        frame_path = "/path/to/test_frame_001.jpg"
        frame_metadata = {"video_id": "vid1", "timestamp": time.time(), "creator_id": "creatorA"}

        # Mock the _generate_visual_embeddings to control its output for this test
        dummy_embedding = np.random.rand(512).astype(np.float32)
        with patch.object(self.trend_detector, '_generate_visual_embeddings', return_value=dummy_embedding) as mock_gen_emb:
            result = self.trend_detector.process_frame_for_visual_similarity(frame_id, frame_path, frame_metadata)

            mock_gen_emb.assert_called_once_with(frame_path)
            self.mock_qdrant_conn.upsert_vectors.assert_called_once()
            args, kwargs = self.mock_qdrant_conn.upsert_vectors.call_args
            self.assertEqual(kwargs['collection_name'], self.trend_detector.qdrant_collection_name)
            self.assertEqual(kwargs['ids'][0], frame_id)
            self.assertEqual(kwargs['vectors'][0], dummy_embedding.tolist())
            self.assertEqual(kwargs['metadata'][0]['original_frame_id'], frame_id)

            self.assertIsNotNone(result)
            self.assertTrue(result['embedding_stored'])
            self.assertEqual(result['frame_id'], frame_id)

    def test_process_frame_for_visual_similarity_no_embedding(self):
        """Test frame processing when embedding generation fails."""
        with patch.object(self.trend_detector, '_generate_visual_embeddings', return_value=None) as mock_gen_emb:
            result = self.trend_detector.process_frame_for_visual_similarity(
                "frame002", "/path/frame.jpg", {}
            )
            self.assertIsNone(result)
            self.mock_qdrant_conn.upsert_vectors.assert_not_called()


    def test_detect_temporal_patterns(self):
        """Test temporal pattern detection (uses mock data internally)."""
        # This method currently uses hardcoded mock data, so we test its structure.
        # In a real scenario, we'd mock Neo4j calls.
        pattern_data = self.trend_detector.detect_temporal_patterns("cluster_A")
        self.assertIn("cluster_id", pattern_data)
        self.assertIn("timestamps", pattern_data)
        self.assertIn("frequency_per_day", pattern_data)
        self.assertTrue(len(pattern_data["timestamps"]) > 0) # Based on current mock

    def test_calculate_trend_velocity(self):
        """Test trend velocity calculation."""
        # Test with enough data
        freq_data_increasing = {"day1": 2, "day2": 5, "day3": 10}
        velocity = self.trend_detector.calculate_trend_velocity({"frequency_per_day": freq_data_increasing})
        self.assertEqual(velocity, 5.0) # 10 - 5

        # Test with decreasing data
        freq_data_decreasing = {"day1": 10, "day2": 6, "day3": 3}
        velocity = self.trend_detector.calculate_trend_velocity({"frequency_per_day": freq_data_decreasing})
        self.assertEqual(velocity, -3.0) # 3 - 6

        # Test with insufficient data
        velocity_insufficient = self.trend_detector.calculate_trend_velocity({"frequency_per_day": {"day1": 1}})
        self.assertEqual(velocity_insufficient, 0.0)

        velocity_empty = self.trend_detector.calculate_trend_velocity({"frequency_per_day": {}})
        self.assertEqual(velocity_empty, 0.0)

    def test_track_cross_creator_adoption(self):
        """Test cross-creator adoption tracking (uses mock data internally)."""
        # This method also uses hardcoded mock data.
        adoption_data = self.trend_detector.track_cross_creator_adoption("cluster_B")
        self.assertIn("cluster_id", adoption_data)
        self.assertIn("creator_adoption_count", adoption_data)
        self.assertIn("creators", adoption_data)
        self.assertTrue(adoption_data["creator_adoption_count"] > 0) # Based on current mock

    def test_classify_trend(self):
        """Test trend classification based on associated data."""
        data_ingredient = {"tags": ["food", "recipe"], "objects": ["bowl"]}
        trend_type = self.trend_detector.classify_trend("cluster_C", data_ingredient)
        self.assertEqual(trend_type, TrendType.INGREDIENT)

        data_format = {"tags": ["dance", "filter"]}
        trend_type = self.trend_detector.classify_trend("cluster_D", data_format)
        self.assertEqual(trend_type, TrendType.FORMAT)

        data_unknown = {"tags": ["random"], "objects": ["thing"]}
        trend_type = self.trend_detector.classify_trend("cluster_E", data_unknown)
        self.assertEqual(trend_type, TrendType.UNKNOWN)

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        # High confidence scenario
        trend_data_high = {"total_occurrences": 150, "adoption_count": 12, "velocity": 3.0}
        score_high = self.trend_detector.calculate_confidence_score(trend_data_high)
        self.assertTrue(0.0 <= score_high <= 1.0)
        # Based on current scoring: (0.4*1 + 0.4*1 + 0.2*min(3/5,1)) = 0.8 + 0.2*0.6 = 0.8 + 0.12 = 0.92
        self.assertAlmostEqual(score_high, 0.92, places=2)


        # Low confidence scenario
        trend_data_low = {"total_occurrences": 5, "adoption_count": 1, "velocity": 0.1}
        score_low = self.trend_detector.calculate_confidence_score(trend_data_low)
        self.assertTrue(0.0 <= score_low <= 1.0)
        # (0.4*0.05 + 0.4*0.1 + 0.2*min(0.1/5,1)) = 0.02 + 0.04 + 0.2*0.02 = 0.06 + 0.004 = 0.064
        self.assertAlmostEqual(score_low, 0.064, places=3)

    def test_detect_trend_lifecycle_stage(self):
        """Test trend lifecycle stage detection."""
        stage = self.trend_detector.detect_trend_lifecycle_stage({"velocity": 2.0, "total_occurrences": 30})
        self.assertEqual(stage, TrendLifecycleStage.EMERGING)

        stage = self.trend_detector.detect_trend_lifecycle_stage({"velocity": 1.0, "total_occurrences": 100})
        self.assertEqual(stage, TrendLifecycleStage.PEAKING) # velocity > 0.5, occurrences >= 50

        stage = self.trend_detector.detect_trend_lifecycle_stage({"velocity": -1.0, "total_occurrences": 150})
        self.assertEqual(stage, TrendLifecycleStage.DECLINING)

        stage = self.trend_detector.detect_trend_lifecycle_stage({"velocity": 0.1, "total_occurrences": 100})
        self.assertEqual(stage, TrendLifecycleStage.STABLE)

    def test_identify_trends_from_data_orchestration(self):
        """Test the main orchestration method identify_trends_from_data.
        This will be a high-level test ensuring components are called.
        The internal logic of clustering is heavily mocked in TrendDetector itself.
        """
        num_frames = 5
        processed_frames_data = []
        base_ts = int(time.time()) - (10 * 24 * 60 * 60)
        for i in range(num_frames):
            processed_frames_data.append({
                "frame_id": f"frame_{i}",
                "frame_path": f"/test/frame_{i}.jpg",
                "metadata": {
                    "timestamp": base_ts + i * 3600,
                    "video_id": f"vid_{i%2}",
                    "creator_id": f"creator_{i%3}"
                }
            })

        # Mock _generate_visual_embeddings as it's called internally by process_frame_for_visual_similarity
        dummy_embedding = np.random.rand(512).astype(np.float32)
        with patch.object(self.trend_detector, '_generate_visual_embeddings', return_value=dummy_embedding):
            # Patch the sub-methods that are complex or rely on external data for this orchestration test
            with patch.object(self.trend_detector, 'detect_temporal_patterns') as mock_detect_temp:
                mock_detect_temp.return_value = {"cluster_id": "mock_cluster", "timestamps": [1,2,3], "frequency_per_day": {"d1":3}}
                with patch.object(self.trend_detector, 'calculate_trend_velocity') as mock_calc_velo:
                    mock_calc_velo.return_value = 1.0
                    with patch.object(self.trend_detector, 'track_cross_creator_adoption') as mock_track_adopt:
                        mock_track_adopt.return_value = {"cluster_id": "mock_cluster", "creator_adoption_count": 2, "creators": ["cA", "cB"]}
                        with patch.object(self.trend_detector, 'classify_trend') as mock_classify:
                            mock_classify.return_value = TrendType.AESTHETIC
                            with patch.object(self.trend_detector, 'calculate_confidence_score') as mock_calc_conf:
                                mock_calc_conf.return_value = 0.75
                                with patch.object(self.trend_detector, 'detect_trend_lifecycle_stage') as mock_detect_life:
                                    mock_detect_life.return_value = TrendLifecycleStage.EMERGING

                                    identified_trends = self.trend_detector.identify_trends_from_data(processed_frames_data)

                                    # Check that process_frame_for_visual_similarity was called for each frame
                                    self.assertEqual(self.mock_qdrant_conn.upsert_vectors.call_count, num_frames)

                                    # Check if trends were identified (mocked clustering logic is very simple)
                                    # The number of trends depends on the mock clustering logic in identify_trends_from_data
                                    # Current mock logic in TrendDetector.identify_trends_from_data might create up to 3 trends.
                                    self.assertTrue(len(identified_trends) <= 3) # Based on current mock

                                    if identified_trends:
                                        # Check if the analysis pipeline methods were called for each mock trend
                                        num_mock_trends = len(identified_trends)
                                        self.assertEqual(mock_detect_temp.call_count, num_mock_trends)
                                        self.assertEqual(mock_calc_velo.call_count, num_mock_trends)
                                        self.assertEqual(mock_track_adopt.call_count, num_mock_trends)
                                        self.assertEqual(mock_classify.call_count, num_mock_trends)
                                        self.assertEqual(mock_calc_conf.call_count, num_mock_trends)
                                        self.assertEqual(mock_detect_life.call_count, num_mock_trends)

                                        # Check structure of one trend
                                        trend = identified_trends[0]
                                        self.assertIn("cluster_id_proxy", trend)
                                        self.assertEqual(trend["type"], TrendType.AESTHETIC)
                                        self.assertEqual(trend["lifecycle_stage"], TrendLifecycleStage.EMERGING.value)
                                        self.assertEqual(trend["confidence_score"], 0.75)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
