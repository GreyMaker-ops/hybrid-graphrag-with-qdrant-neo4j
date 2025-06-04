import cv2
import os
import json
import logging
import time
import math # For ceiling function in segment calculation
from graphrag.connectors.neo4j_connection import get_connection as get_neo4j_connection

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoIngestor:
    def __init__(self, neo4j_conn=None, temporary_frame_storage_path='temp_frames/', segment_duration: int = 5):
        self.logger = logging.getLogger(__name__)
        self.neo4j = neo4j_conn # Will be get_neo4j_connection() if None, see below
        self.temporary_frame_storage_path = temporary_frame_storage_path
        self.segment_duration = segment_duration

        if not os.path.exists(self.temporary_frame_storage_path):
            os.makedirs(self.temporary_frame_storage_path)
            self.logger.info(f"Created temporary frame storage directory: {self.temporary_frame_storage_path}")

        # Initialize Neo4j connection if not provided
        if self.neo4j is None:
            try:
                self.neo4j = get_neo4j_connection()
                self.logger.info("Successfully connected to Neo4j.")
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {e}. Neo4j operations will be skipped.")
                self.neo4j = None # Ensure it's None if connection failed

    def extract_frames(self, video_path: str, target_extraction_fps: int = 1) -> tuple[list[dict], float]:
        self.logger.info(f"Starting frame extraction for video: {video_path} at {target_extraction_fps} FPS.")
        extracted_frames_info = []
        video_filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        video_actual_fps = 0.0

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Error: Could not open video file: {video_path}")
                return extracted_frames_info, video_actual_fps

            video_actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_actual_fps == 0:
                self.logger.warning(f"Video FPS is 0 for {video_path}. Using default 30 FPS. Frame timestamps might be inaccurate.")
                video_actual_fps = 30.0 # Default if FPS is not available or reliable

            # Ensure target_extraction_fps is not higher than video_actual_fps if video_actual_fps is known and positive
            if video_actual_fps > 0 and target_extraction_fps > video_actual_fps:
                self.logger.warning(f"Target extraction FPS ({target_extraction_fps}) is higher than video FPS ({video_actual_fps}). Clamping to video FPS.")
                target_extraction_fps = video_actual_fps

            frame_extraction_interval = int(video_actual_fps / target_extraction_fps) if target_extraction_fps > 0 else 1
            if frame_extraction_interval <= 0:
                frame_extraction_interval = 1

            original_frame_number = 0
            saved_frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if original_frame_number % frame_extraction_interval == 0:
                    timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    timestamp_sec = timestamp_msec / 1000.0

                    # Fallback if POS_MSEC is not reliable (e.g. returns 0 after first frame)
                    if saved_frame_count > 0 and timestamp_sec == 0 and extracted_frames_info:
                         # Estimate current timestamp based on frame number and FPS
                        timestamp_sec = original_frame_number / video_actual_fps if video_actual_fps > 0 else saved_frame_count / target_extraction_fps

                    frame_filename = f"{video_filename_no_ext}_frame_{saved_frame_count}.jpg"
                    frame_path = os.path.join(self.temporary_frame_storage_path, frame_filename)

                    try:
                        cv2.imwrite(frame_path, frame)
                        extracted_frames_info.append({
                            'frame_path': frame_path,
                            'frame_number': original_frame_number, # Original frame number in video
                            'saved_frame_index': saved_frame_count, # Index of the saved frame
                            'timestamp': timestamp_sec
                        })
                        saved_frame_count += 1
                    except cv2.error as e:
                        self.logger.error(f"OpenCV error writing frame {frame_path}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error writing frame {frame_path}: {e}")

                original_frame_number += 1

            cap.release()
            self.logger.info(f"Extracted {saved_frame_count} frames from {video_path} to {self.temporary_frame_storage_path}. Original FPS: {video_actual_fps:.2f}")

        except cv2.error as e:
            self.logger.error(f"OpenCV error during frame extraction for {video_path}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during frame extraction for {video_path}: {e}")

        return extracted_frames_info, video_actual_fps

    def create_temporal_segments(self, video_path: str, extracted_frames_info: list[dict], video_fps: float) -> list[dict]:
        self.logger.info(f"Creating temporal segments for video: {video_path}")
        segments = []
        if not extracted_frames_info:
            self.logger.warning("No frames provided to create segments.")
            return segments

        video_filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]

        if video_fps <= 0:
            self.logger.warning(f"Video FPS is {video_fps}. Segment creation by duration may be inaccurate. Grouping frames by count or individual frames.")
            # Fallback: if FPS is unknown, group by a fixed number of frames or make each frame a segment.
            # For simplicity, let's assume if fps is 0, timestamps are also unreliable for duration grouping.
            # We'll make each extracted frame its own segment in this extreme fallback.
            for idx, frame_info in enumerate(extracted_frames_info):
                segment_id = f"{video_filename_no_ext}_segment_{idx}"
                segments.append({
                    'segment_id': segment_id,
                    'video_id': os.path.basename(video_path),
                    'start_time': frame_info['timestamp'], # May be inaccurate
                    'end_time': frame_info['timestamp'],   # May be inaccurate
                    'frames': [frame_info],
                    'frame_count': 1
                })
            self.logger.info(f"Created {len(segments)} segments for video {video_path} (fallback due to FPS <= 0).")
            return segments

        # Primary logic: Group by time duration using frame timestamps
        current_segment_frames = []
        segment_index = 0

        if not extracted_frames_info: # Should be caught above, but as a safeguard
            return segments

        segment_start_time = extracted_frames_info[0]['timestamp']

        for i, frame_info in enumerate(extracted_frames_info):
            current_segment_frames.append(frame_info)

            is_last_frame_in_video = (i == len(extracted_frames_info) - 1)
            # Check if current frame pushes segment over duration OR if it's the last frame
            # The condition for creating a segment is:
            # 1. It's the last frame overall.
            # 2. Or, the *next* frame would make the current segment too long (its timestamp minus current segment's start time).
            #    (Need to ensure there *is* a next frame for this check)
            # 3. Or, if the current frame itself already makes the segment longer than duration (useful for single-frame segments if duration is very small).

            time_in_current_segment = frame_info['timestamp'] - segment_start_time

            if is_last_frame_in_video or \
               ( (i + 1 < len(extracted_frames_info)) and (extracted_frames_info[i+1]['timestamp'] - segment_start_time >= self.segment_duration) ) or \
               ( time_in_current_segment >= self.segment_duration and not current_segment_frames ): # Handles very short segments for first frame

                # Finalize current segment
                segment_id = f"{video_filename_no_ext}_segment_{segment_index}"
                segment_end_time = frame_info['timestamp']

                segments.append({
                    'segment_id': segment_id,
                    'video_id': os.path.basename(video_path), # Using filename as video_id for now
                    'start_time': segment_start_time,
                    'end_time': segment_end_time,
                    'frames': list(current_segment_frames),
                    'frame_count': len(current_segment_frames)
                })

                segment_index += 1
                current_segment_frames = []
                if not is_last_frame_in_video: # Prepare for next segment if not the end
                    segment_start_time = extracted_frames_info[i+1]['timestamp']

        self.logger.info(f"Created {len(segments)} segments for video {video_path}.")
        return segments

    def store_video_in_neo4j(self, video_metadata: dict, segments: list[dict]) -> None:
        if not self.neo4j:
            self.logger.warning("Neo4j connection not available. Skipping storage.")
            return

        try:
            self.logger.info(f"Storing video and segment data in Neo4j for video_id: {video_metadata.get('video_id')}")

            # 1. Create/Merge :Video node
            video_query = """
            MERGE (v:Video {id: $video_id})
            ON CREATE SET
                v.platform = $platform,
                v.creator_id = $creator_id,
                v.duration_seconds = $duration_seconds,
                v.engagement_metrics = $engagement_metrics,
                v.filename = $filename,
                v.file_size_bytes = $file_size_bytes,
                v.file_creation_date_unix = $file_creation_date_unix,
                v.file_modification_date_unix = $file_modification_date_unix,
                v.processed_timestamp = timestamp()
            ON MATCH SET
                v.platform = $platform,
                v.creator_id = $creator_id,
                v.duration_seconds = $duration_seconds,
                v.engagement_metrics = $engagement_metrics,
                v.filename = $filename,
                v.file_size_bytes = $file_size_bytes,
                v.file_creation_date_unix = $file_creation_date_unix,
                v.file_modification_date_unix = $file_modification_date_unix,
                v.updated_timestamp = timestamp()
            """
            # engagement_metrics can be a dict, store as JSON string
            video_params = {
                'video_id': video_metadata.get('video_id', os.path.basename(video_metadata.get('filename', 'unknown_video'))), # Ensure video_id exists
                'platform': video_metadata.get('platform'),
                'creator_id': video_metadata.get('creator_id'),
                'duration_seconds': video_metadata.get('video_duration_seconds'),
                'engagement_metrics': json.dumps(video_metadata.get('engagement_metrics', {})),
                'filename': video_metadata.get('filename'),
                'file_size_bytes': video_metadata.get('file_size_bytes'),
                'file_creation_date_unix': video_metadata.get('file_creation_date_unix'),
                'file_modification_date_unix': video_metadata.get('file_modification_date_unix'),
            }
            self.neo4j.run_query(video_query, video_params)
            self.logger.info(f"Merged :Video node for {video_params['video_id']}")

            # 2. Create/Merge :VideoSegment nodes and :Frame nodes
            for seg_idx, segment in enumerate(segments):
                segment_id = segment['segment_id']
                video_id_for_segment = segment.get('video_id', video_params['video_id']) # Use video_id from segment or main video_id

                seg_query = """
                MATCH (v:Video {id: $video_id})
                MERGE (s:VideoSegment {id: $segment_id})
                ON CREATE SET
                    s.video_id = $video_id,
                    s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.frame_count = $frame_count,
                    s.temporal_index = $temporal_index, // Store segment order
                    s.processed_timestamp = timestamp()
                ON MATCH SET
                    s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.frame_count = $frame_count,
                    s.temporal_index = $temporal_index,
                    s.updated_timestamp = timestamp()
                MERGE (v)-[:HAS_SEGMENT]->(s)
                """
                seg_params = {
                    'video_id': video_id_for_segment,
                    'segment_id': segment_id,
                    'start_time': segment.get('start_time'),
                    'end_time': segment.get('end_time'),
                    'frame_count': segment.get('frame_count'),
                    'temporal_index': seg_idx # Add temporal index for ordering
                }
                self.neo4j.run_query(seg_query, seg_params)
                self.logger.debug(f"Merged :VideoSegment node {segment_id} and relationship to Video {video_id_for_segment}")

                for frame_info in segment.get('frames', []):
                    frame_id = f"{video_id_for_segment}_frame_{frame_info['frame_number']}"
                    frame_query = """
                    MATCH (v:Video {id: $video_id})
                    // MATCH (seg:VideoSegment {id: $segment_id}) // Optional: if direct relationship to segment is needed
                    MERGE (f:Frame {id: $frame_id})
                    ON CREATE SET
                        f.video_id = $video_id,
                        f.timestamp = $timestamp,
                        f.frame_number = $frame_number,
                        f.path = $path,
                        f.segment_id = $segment_id, // Store segment_id as property
                        f.saved_frame_index = $saved_frame_index,
                        f.processed_timestamp = timestamp()
                    ON MATCH SET
                        f.timestamp = $timestamp,
                        f.path = $path,
                        f.segment_id = $segment_id,
                        f.saved_frame_index = $saved_frame_index,
                        f.updated_timestamp = timestamp()
                    MERGE (v)-[:CONTAINS_FRAME]->(f)
                    // MERGE (seg)-[:HAS_FRAME_DATA]->(f) // Example of direct segment to frame relationship if needed
                    """
                    frame_params = {
                        'video_id': video_id_for_segment,
                        'frame_id': frame_id,
                        'timestamp': frame_info.get('timestamp'),
                        'frame_number': frame_info.get('frame_number'),
                        'path': frame_info.get('frame_path'),
                        'segment_id': segment_id,
                        'saved_frame_index': frame_info.get('saved_frame_index')
                    }
                    self.neo4j.run_query(frame_query, frame_params)
                    self.logger.debug(f"Merged :Frame node {frame_id} and relationship to Video {video_id_for_segment}")

            self.logger.info(f"Stored {len(segments)} segments and their frames for video {video_params['video_id']}")

            # 3. Create temporal relationships between :VideoSegment nodes
            for i in range(len(segments) - 1):
                s1_id = segments[i]['segment_id']
                s2_id = segments[i+1]['segment_id']
                temporal_query = """
                MATCH (s1:VideoSegment {id: $s1_id})
                MATCH (s2:VideoSegment {id: $s2_id})
                MERGE (s1)-[:BEFORE]->(s2)
                MERGE (s2)-[:AFTER]->(s1)
                """
                self.neo4j.run_query(temporal_query, {'s1_id': s1_id, 's2_id': s2_id})
                self.logger.debug(f"Created temporal relationships between segment {s1_id} and {s2_id}")

            self.logger.info(f"Successfully stored all data in Neo4j for video {video_params['video_id']}.")

        except Exception as e:
            self.logger.error(f"Error storing video data in Neo4j for video {video_metadata.get('video_id')}: {e}", exc_info=True)


    def extract_metadata(self, video_path: str, platform: str = 'unknown') -> dict:
        self.logger.info(f"Extracting metadata for video: {video_path} from platform: {platform}")
        metadata = {
            'platform': platform,
            'video_id': os.path.basename(video_path), # Default to filename if no other ID; this might be overridden by platform-specific ID
            'filename': os.path.basename(video_path),
            'file_size_bytes': None,
            'file_creation_date_unix': None, # Renamed for clarity
            'file_modification_date_unix': None, # Renamed for clarity
            'creator_id': 'placeholder_creator', # Placeholder
            'engagement_metrics': {'likes': 0, 'views': 0}, # Placeholder
            'video_duration_seconds': None # Placeholder, could be filled by opencv if reliable
        }

        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Metadata extraction error: Video file not found at {video_path}")
                return metadata # Return basic metadata with Nones

            metadata['file_size_bytes'] = os.path.getsize(video_path)
            metadata['file_creation_date_unix'] = os.path.getctime(video_path)
            metadata['file_modification_date_unix'] = os.path.getmtime(video_path)

            # Attempt to get video duration using OpenCV
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0 and frame_count > 0:
                        metadata['video_duration_seconds'] = frame_count / fps
                    cap.release()
            except Exception as e:
                self.logger.warning(f"Could not determine video duration using OpenCV for {video_path}: {e}")

            # Simulate platform-specific metadata
            if platform == 'youtube':
                metadata['creator_id'] = 'youtube_creator_example'
                metadata['engagement_metrics'] = {'likes': 1000, 'views': 100000, 'comments': 50}
            elif platform == 'tiktok':
                metadata['creator_id'] = 'tiktok_creator_example'
                metadata['engagement_metrics'] = {'likes': 5000, 'views': 500000, 'shares': 200}

            self.logger.info(f"Successfully extracted metadata for {video_path}: {json.dumps(metadata, indent=2)}")

        except FileNotFoundError:
             self.logger.error(f"Error extracting metadata: File not found {video_path}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during metadata extraction for {video_path}: {e}")

        return metadata

    def process_video(self, video_path: str, platform: str = 'unknown', extraction_fps: int = 1) -> tuple[list[dict], dict]:
        self.logger.info(f"Starting processing for video: {video_path}, platform: {platform}, target extraction FPS: {extraction_fps}, segment duration: {self.segment_duration}s")

        metadata = self.extract_metadata(video_path, platform)
        extracted_frames_info, video_actual_fps = self.extract_frames(video_path, extraction_fps)

        if metadata.get('video_duration_seconds') is None and video_actual_fps > 0 and extracted_frames_info:
            # Estimate duration from extracted frames if not available from metadata header
            # This would be the duration of the extracted portion if extraction did not cover the whole video
            # Or if cap.get(cv2.CAP_PROP_FRAME_COUNT) was unreliable.
            # A more accurate original video duration would be from cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_actual_fps
            # but that's already attempted in extract_metadata.
            # This is a fallback using the last frame's timestamp.
            if extracted_frames_info:
                 metadata['video_duration_seconds'] = extracted_frames_info[-1]['timestamp']


        segments = []
        if extracted_frames_info:
            segments = self.create_temporal_segments(video_path, extracted_frames_info, video_actual_fps)
        else:
            self.logger.warning(f"No frames were extracted from {video_path}, so no segments will be created.")

        self.logger.info(f"Completed processing for video: {video_path}. Created {len(segments)} segments and extracted metadata.")

        if self.neo4j and segments: # Store if Neo4j is available and segments were created
            self.store_video_in_neo4j(metadata, segments)

        return segments, metadata

# Mock Neo4j connection for testing if a live one isn't configured or available
class MockNeo4jConnection:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".MockNeo4jConnection")
        self.logger.info("MockNeo4jConnection initialized. Queries will be printed, not executed.")

    def run_query(self, query, params=None):
        self.logger.info(f"Executing Cypher Query (Mocked):")
        self.logger.info(f"  Query: {query.strip()}")
        if params:
            self.logger.info(f"  Params: {json.dumps(params, indent=2)}")
        return [] # Mocked result

    def close(self):
        self.logger.info("MockNeo4jConnection closed.")


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.info("Starting VideoIngestor example with temporal segmentation and Neo4j (mocked) storage.")

    # --- Setup for testing ---
    # Option 1: Use a real Neo4j connection if available and configured via environment variables
    # neo4j_conn = None # This will make VideoIngestor try to use get_neo4j_connection()
    # Option 2: Explicitly use the mock connection for this example
    use_mock_neo4j = True
    neo4j_conn_instance = MockNeo4jConnection() if use_mock_neo4j else None

    if use_mock_neo4j:
        logger.info("Using MOCKED Neo4j connection for this example run.")
    elif neo4j_conn_instance is None and not os.getenv("NEO4J_URI"): # Or however your get_connection checks
        logger.warning("NEO4J_URI not set, and not using mock. Real Neo4j connection might fail if not configured.")


    dummy_video_filename = "test_video_neo4j.mp4"
    if not os.path.exists(dummy_video_filename):
        try:
            with open(dummy_video_filename, "w") as f:
                f.write("This is a dummy video file for testing Neo4j storage.")
            logger.info(f"Created dummy video file: {dummy_video_filename}")
        except IOError as e:
            logger.error(f"Could not create dummy video file: {e}")

    ingestor = VideoIngestor(
        neo4j_conn=neo4j_conn_instance, # Pass the chosen connection (real or mock)
        temporary_frame_storage_path='test_temp_frames_neo4j/',
        segment_duration=3 # 3s segments
    )

    logger.info(f"Processing dummy video: {dummy_video_filename} (expecting no frames, but metadata and Neo4j calls for video node).")
    # Processing a dummy text file: extract_frames will yield nothing, so segments list will be empty.
    # store_video_in_neo4j will be called with empty segments, should still create Video node.
    segments, metadata = ingestor.process_video(dummy_video_filename, platform='dummy_platform', extraction_fps=1)

    logger.info(f"\n--- Processing Results for {dummy_video_filename} ---")
    logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
    logger.info(f"Segments created: {len(segments)}")
    if not segments : logger.info("(No segments expected for a non-video file, but Video node should be created in Neo4j if connection active/mocked).")

    # --- Simulate processing for a conceptual 'real_video.mp4' to test Neo4j storage logic more fully ---
    logger.info(f"\n--- Simulating processing for 'real_video_sim.mp4' with Neo4j storage ---")

    # Re-use ingestor, or create new one if specific settings needed
    # ingestor = VideoIngestor(neo4j_conn=neo4j_conn_instance, temporary_frame_storage_path='test_temp_frames_neo4j_sim/', segment_duration=2)

    sim_video_filename = "real_video_sim.mp4"
    sim_video_id = "sim_vid_001" # More controlled ID for simulation
    sim_platform = "youtube_sim"

    # Mock metadata that `extract_metadata` would produce
    sim_metadata = {
        'video_id': sim_video_id,
        'filename': sim_video_filename,
        'platform': sim_platform,
        'creator_id': 'sim_creator_789',
        'video_duration_seconds': 10.0, # 10 second video
        'engagement_metrics': {'likes': 150, 'views': 3000},
        'file_size_bytes': 5000000,
        'file_creation_date_unix': time.time(),
        'file_modification_date_unix': time.time(),
    }

    # Mock extracted frames data that `extract_frames` would produce
    sim_video_actual_fps = 10.0 # Simulate a 10 FPS video
    sim_extracted_frames_info = []
    # Simulate extracting 1 frame per second for 10 seconds
    for i in range(10):
        frame_num_original = i * int(sim_video_actual_fps) # Frame number in original 10fps video
        ts = float(i) # Timestamp in seconds
        frame_p = os.path.join(ingestor.temporary_frame_storage_path, f"{sim_video_id}_frame_{i}.jpg")
        # Create dummy frame files for completeness if needed by a real process
        if not os.path.exists(ingestor.temporary_frame_storage_path): os.makedirs(ingestor.temporary_frame_storage_path)
        if not os.path.exists(frame_p): open(frame_p, 'w').write(f"dummy frame {i}")

        sim_extracted_frames_info.append({
            'frame_path': frame_p,
            'frame_number': frame_num_original,
            'saved_frame_index': i,
            'timestamp': ts
        })

    # Manually call the relevant parts of process_video for simulation
    if sim_extracted_frames_info:
        logger.info(f"Simulated extraction of {len(sim_extracted_frames_info)} frames for {sim_video_filename}.")
        sim_segments = ingestor.create_temporal_segments(sim_video_filename, sim_extracted_frames_info, sim_video_actual_fps)

        logger.info(f"Simulated segment creation results (segment_duration={ingestor.segment_duration}s): {len(sim_segments)} segments.")
        for idx, seg_info in enumerate(sim_segments):
            logger.debug(f"  Sim Segment {idx}: ID: {seg_info['segment_id']}, Start: {seg_info['start_time']:.2f}s, End: {seg_info['end_time']:.2f}s, Frames: {seg_info['frame_count']}")

        # Crucially, call store_video_in_neo4j with this simulated data
        if ingestor.neo4j : # Check if connection is available (real or mock)
            # We need to ensure the video_id in metadata matches what segments expect
            sim_metadata['video_id'] = sim_video_id # Ensure this is consistent if create_temporal_segments used os.path.basename
            # Correct video_id in segments if create_temporal_segments used basename of sim_video_filename
            for seg in sim_segments:
                seg['video_id'] = sim_video_id

            ingestor.store_video_in_neo4j(sim_metadata, sim_segments)
        else:
            logger.warning("Simulated Neo4j storage skipped as connection is not available.")
    else:
        logger.info("No mock frames generated for simulation, Neo4j storage test with segments skipped.")

    # Clean up
    logger.info("Cleaning up dummy files and directories...")
    try:
        if os.path.exists(dummy_video_filename):
            os.remove(dummy_video_filename)
            logger.info(f"Cleaned up dummy video file: {dummy_video_filename}")

        # Clean up simulated frame files and directory
        if os.path.exists(ingestor.temporary_frame_storage_path):
            for item in os.listdir(ingestor.temporary_frame_storage_path):
                item_path = os.path.join(ingestor.temporary_frame_storage_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            # Attempt to remove directory if empty
            if not os.listdir(ingestor.temporary_frame_storage_path):
                 os.rmdir(ingestor.temporary_frame_storage_path)
                 logger.info(f"Cleaned up and removed temporary frame storage: {ingestor.temporary_frame_storage_path}")
            else:
                logger.warning(f"Temporary frame storage not empty, did not remove: {ingestor.temporary_frame_storage_path}. Contains: {os.listdir(ingestor.temporary_frame_storage_path)}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

    if isinstance(ingestor.neo4j, MockNeo4jConnection):
        ingestor.neo4j.close()
    elif ingestor.neo4j: # If it's a real connection, it should have a close method
        try:
            ingestor.neo4j.close()
            logger.info("Closed real Neo4j connection.")
        except Exception as e:
            logger.error(f"Error closing real Neo4j connection: {e}")


    logger.info("VideoIngestor example with Neo4j integration finished.")
