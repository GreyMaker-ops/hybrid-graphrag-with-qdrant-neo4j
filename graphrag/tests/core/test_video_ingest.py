import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import shutil
import json
import time # for timestamp checks
# Attempt to import cv2, but it's okay if it's not found as we'll mock it.
try:
    import cv2
except ImportError:
    cv2 = MagicMock() # If cv2 is not installed, mock it globally for type hinting

from graphrag.core.video_ingest import VideoIngestor
from graphrag.connectors.neo4j_connection import Neo4jConnection # For type hinting mock

# Mock Neo4j connection for VideoIngestor if a real one isn't available/desired for tests
class MockNeo4jConnectionForTests:
    def __init__(self):
        self.run_query = MagicMock()
        self.close = MagicMock()

class TestVideoIngestor(unittest.TestCase):
    def setUp(self):
        self.test_temp_dir = "test_temp_frames_ingestor"
        self.dummy_video_path = os.path.join(self.test_temp_dir, "test_video.mp4")

        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)
        os.makedirs(self.test_temp_dir, exist_ok=True)

        # Create a dummy video file (empty)
        with open(self.dummy_video_path, 'w') as f:
            f.write("dummy video content")

        self.mock_neo4j_conn = MockNeo4jConnectionForTests()
        self.video_ingestor = VideoIngestor(
            neo4j_conn=self.mock_neo4j_conn,
            temporary_frame_storage_path=self.test_temp_dir,
            segment_duration=2 # 2s segments for easier testing
        )

    def tearDown(self):
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)

    def _create_mock_cv2_video_capture(self, video_path, frame_count=30, fps=30.0, width=1920, height=1080):
        mock_cap = MagicMock(spec=cv2.VideoCapture)
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop_id: {
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_POS_MSEC: (mock_cap.frame_read_count * (1000.0 / fps)) if fps > 0 else 0 # Simulate time progression
        }.get(prop_id, 0)

        mock_cap.frame_read_count = 0
        def mock_read(*args, **kwargs):
            if mock_cap.frame_read_count < frame_count:
                mock_cap.frame_read_count += 1
                # Return a mock frame (e.g., a small numpy array or just a MagicMock)
                return True, MagicMock() # Simulate a successful frame read
            else:
                return False, None # Simulate end of video

        mock_cap.read.side_effect = mock_read
        mock_cap.release = MagicMock()
        return mock_cap

    @patch('os.path.getsize', return_value=1024*1024) # 1MB
    @patch('os.path.getmtime', return_value=time.time())
    @patch('os.path.getctime', return_value=time.time() - 3600) # Created 1 hour ago
    def test_extract_metadata_basic(self, mock_ctime, mock_mtime, mock_getsize):
        metadata = self.video_ingestor.extract_metadata(self.dummy_video_path, platform="test_platform")
        self.assertEqual(metadata['video_id'], os.path.basename(self.dummy_video_path))
        self.assertEqual(metadata['platform'], "test_platform")
        self.assertEqual(metadata['file_size_bytes'], 1024*1024)
        self.assertIsNotNone(metadata['file_creation_date_unix'])
        self.assertIsNotNone(metadata['file_modification_date_unix'])

    @patch('cv2.VideoCapture')
    def test_extract_metadata_with_cv2_duration(self, mock_cv2_vc):
        mock_cap_instance = self._create_mock_cv2_video_capture(self.dummy_video_path, frame_count=150, fps=30.0)
        mock_cv2_vc.return_value = mock_cap_instance

        metadata = self.video_ingestor.extract_metadata(self.dummy_video_path)
        self.assertAlmostEqual(metadata['video_duration_seconds'], 5.0, places=2) # 150 frames / 30 fps = 5s

    @patch('cv2.imwrite', return_value=True)
    @patch('cv2.VideoCapture')
    def test_extract_frames_success(self, mock_cv2_vc, mock_cv2_imwrite):
        mock_cap_instance = self._create_mock_cv2_video_capture(self.dummy_video_path, frame_count=60, fps=30.0) # 2s video at 30fps
        mock_cv2_vc.return_value = mock_cap_instance

        # Extract 1 frame per second from a 2s video (30fps original) -> expect 2 frames
        frames_info, actual_fps = self.video_ingestor.extract_frames(self.dummy_video_path, target_extraction_fps=1)

        self.assertEqual(actual_fps, 30.0)
        self.assertEqual(len(frames_info), 2) # Should extract 2 frames (at t=0s, t=1s)
        self.assertEqual(mock_cv2_imwrite.call_count, 2)

        # Check structure of first frame_info
        self.assertIn('frame_path', frames_info[0])
        self.assertTrue(frames_info[0]['frame_path'].endswith('_frame_0.jpg'))
        self.assertIn('frame_number', frames_info[0]) # Original frame number
        self.assertIn('timestamp', frames_info[0])
        self.assertAlmostEqual(frames_info[0]['timestamp'], 0.0 / 30.0, places=2) # First frame at 0ms
        self.assertAlmostEqual(frames_info[1]['timestamp'], 30.0 / 30.0, places=2) # Second frame extracted at 1s (30th frame)

    @patch('cv2.VideoCapture')
    def test_extract_frames_video_not_opened(self, mock_cv2_vc):
        mock_cap_instance = MagicMock(spec=cv2.VideoCapture)
        mock_cap_instance.isOpened.return_value = False
        mock_cv2_vc.return_value = mock_cap_instance

        frames_info, actual_fps = self.video_ingestor.extract_frames("non_existent_video.mp4", target_extraction_fps=1)
        self.assertEqual(len(frames_info), 0)
        self.assertEqual(actual_fps, 0.0)

    @patch('cv2.VideoCapture')
    def test_extract_frames_zero_fps_video(self, mock_cv2_vc):
        # Simulate video where FPS is reported as 0
        mock_cap_instance = self._create_mock_cv2_video_capture(self.dummy_video_path, frame_count=10, fps=0.0)
        mock_cv2_vc.return_value = mock_cap_instance

        with patch.object(self.video_ingestor.logger, 'warning') as mock_log_warning:
            frames_info, actual_fps = self.video_ingestor.extract_frames(self.dummy_video_path, target_extraction_fps=1)
            self.assertEqual(actual_fps, 30.0) # Should default to 30.0
            # Warning should be logged about 0 FPS original video and defaulting
            mock_log_warning.assert_any_call(f"Video FPS is 0 for {self.dummy_video_path}. Using default 30 FPS. Frame timestamps might be inaccurate.")


    def test_create_temporal_segments(self):
        # 10 frames, 1 frame per second (fps=1 for extracted frames, not original video)
        # Timestamps are 0.0, 1.0, ..., 9.0
        sample_frames_info = [
            {'frame_path': f'frame_{i}.jpg', 'frame_number': i*10, 'timestamp': float(i), 'saved_frame_index': i}
            for i in range(10)
        ]
        video_fps_original = 30.0 # Original video FPS, used for segment duration calculation context
        self.video_ingestor.segment_duration = 3 # 3 seconds per segment

        segments = self.video_ingestor.create_temporal_segments(self.dummy_video_path, sample_frames_info, video_fps_original)

        # Expected segments:
        # Seg 0: frames 0,1,2 (timestamps 0,1,2)
        # Seg 1: frames 3,4,5 (timestamps 3,4,5)
        # Seg 2: frames 6,7,8 (timestamps 6,7,8)
        # Seg 3: frame 9 (timestamp 9)
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0]['frame_count'], 3)
        self.assertAlmostEqual(segments[0]['start_time'], 0.0)
        self.assertAlmostEqual(segments[0]['end_time'], 2.0)
        self.assertEqual(segments[1]['frame_count'], 3)
        self.assertAlmostEqual(segments[1]['start_time'], 3.0)
        self.assertAlmostEqual(segments[1]['end_time'], 5.0)
        self.assertEqual(segments[3]['frame_count'], 1)
        self.assertAlmostEqual(segments[3]['start_time'], 9.0)
        self.assertAlmostEqual(segments[3]['end_time'], 9.0)

    def test_create_temporal_segments_no_frames(self):
        segments = self.video_ingestor.create_temporal_segments(self.dummy_video_path, [], 30.0)
        self.assertEqual(len(segments), 0)

    def test_create_temporal_segments_shorter_than_duration(self):
        sample_frames_info = [ {'frame_path': 'f_0.jpg', 'frame_number': 0, 'timestamp': 0.0, 'saved_frame_index': 0} ]
        self.video_ingestor.segment_duration = 5 # 5s
        segments = self.video_ingestor.create_temporal_segments(self.dummy_video_path, sample_frames_info, 30.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]['frame_count'], 1)

    def test_store_video_in_neo4j(self):
        video_id = os.path.basename(self.dummy_video_path)
        sample_metadata = {
            'video_id': video_id, 'platform': 'test', 'creator_id': 'c1',
            'video_duration_seconds': 10.0, 'engagement_metrics': {'likes': 10},
            'filename': video_id, 'file_size_bytes':100,
            'file_creation_date_unix': time.time(), 'file_modification_date_unix': time.time()
        }
        sample_segments = [{
            'segment_id': f'{video_id}_segment_0', 'video_id': video_id,
            'start_time': 0.0, 'end_time': 1.9, 'frame_count': 2,
            'frames': [
                {'frame_path': 'f0.jpg', 'frame_number': 0, 'timestamp': 0.0, 'saved_frame_index': 0},
                {'frame_path': 'f1.jpg', 'frame_number': 30, 'timestamp': 1.0, 'saved_frame_index': 1}
            ]
        },{
            'segment_id': f'{video_id}_segment_1', 'video_id': video_id,
            'start_time': 2.0, 'end_time': 3.9, 'frame_count': 1,
            'frames': [
                {'frame_path': 'f2.jpg', 'frame_number': 60, 'timestamp': 2.0, 'saved_frame_index': 2}
            ]
        }]

        self.video_ingestor.store_video_in_neo4j(sample_metadata, sample_segments)

        # Check Video node query
        video_node_call = self.mock_neo4j_conn.run_query.call_args_list[0]
        self.assertIn("MERGE (v:Video {id: $video_id})", video_node_call[0][0])
        self.assertEqual(video_node_call[1]['params']['video_id'], video_id)
        self.assertEqual(video_node_call[1]['params']['engagement_metrics'], json.dumps({'likes': 10}))

        # Check Segment node query (first segment)
        segment_node_call_1 = self.mock_neo4j_conn.run_query.call_args_list[1]
        self.assertIn("MERGE (s:VideoSegment {id: $segment_id})", segment_node_call_1[0][0])
        self.assertEqual(segment_node_call_1[1]['params']['segment_id'], sample_segments[0]['segment_id'])
        self.assertIn("MERGE (v)-[:HAS_SEGMENT]->(s)", segment_node_call_1[0][0])

        # Check Frame node query (first frame of first segment)
        frame_node_call_1 = self.mock_neo4j_conn.run_query.call_args_list[2] # Video, Seg0, Frame0 from Seg0
        self.assertIn("MERGE (f:Frame {id: $frame_id})", frame_node_call_1[0][0])
        expected_frame_id = f"{video_id}_frame_{sample_segments[0]['frames'][0]['frame_number']}"
        self.assertEqual(frame_node_call_1[1]['params']['frame_id'], expected_frame_id)
        self.assertIn("MERGE (v)-[:CONTAINS_FRAME]->(f)", frame_node_call_1[0][0])
        self.assertEqual(frame_node_call_1[1]['params']['segment_id'], sample_segments[0]['segment_id'])

        # Check temporal relationship query (between seg0 and seg1)
        # Total calls: 1 Video, 2 Segments, 3 Frames, 1 Temporal = 7 calls
        self.assertEqual(self.mock_neo4j_conn.run_query.call_count, 1 + len(sample_segments) + 3 + (len(sample_segments)-1) )
        temporal_call_idx = 1 + len(sample_segments) + sum(s['frame_count'] for s in sample_segments)
        temporal_rel_call = self.mock_neo4j_conn.run_query.call_args_list[temporal_call_idx]
        self.assertIn("MERGE (s1)-[:BEFORE]->(s2)", temporal_rel_call[0][0])
        self.assertEqual(temporal_rel_call[1]['params']['s1_id'], sample_segments[0]['segment_id'])
        self.assertEqual(temporal_rel_call[1]['params']['s2_id'], sample_segments[1]['segment_id'])


    @patch('graphrag.core.video_ingest.VideoIngestor.extract_metadata')
    @patch('graphrag.core.video_ingest.VideoIngestor.extract_frames')
    @patch('graphrag.core.video_ingest.VideoIngestor.create_temporal_segments')
    @patch('graphrag.core.video_ingest.VideoIngestor.store_video_in_neo4j')
    def test_process_video_orchestration(self, mock_store, mock_create_segments, mock_extract_frames, mock_extract_metadata):
        # Setup mock return values
        mock_metadata_val = {'video_id': 'test_vid', 'filename': 'test_vid.mp4'}
        mock_extract_metadata.return_value = mock_metadata_val

        mock_frames_info_val = [{'frame_path': 'f.jpg', 'frame_number': 0, 'timestamp': 0.0}]
        mock_video_fps_val = 30.0
        mock_extract_frames.return_value = (mock_frames_info_val, mock_video_fps_val)

        mock_segments_val = [{'segment_id': 'seg1', 'frames': mock_frames_info_val}]
        mock_create_segments.return_value = mock_segments_val

        segments_result, metadata_result = self.video_ingestor.process_video(self.dummy_video_path, platform="test_p", extraction_fps=1)

        mock_extract_metadata.assert_called_once_with(self.dummy_video_path, "test_p")
        mock_extract_frames.assert_called_once_with(self.dummy_video_path, 1)
        mock_create_segments.assert_called_once_with(self.dummy_video_path, mock_frames_info_val, mock_video_fps_val)
        mock_store.assert_called_once_with(mock_metadata_val, mock_segments_val)

        self.assertEqual(segments_result, mock_segments_val)
        self.assertEqual(metadata_result, mock_metadata_val)

if __name__ == '__main__':
    unittest.main()
