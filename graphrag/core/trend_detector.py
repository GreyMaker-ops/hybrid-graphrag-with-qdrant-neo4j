import logging
from enum import Enum
from collections import defaultdict, Counter
import numpy as np # For potential embedding generation and calculations
import uuid # For generating unique IDs for trends
import time # For timestamp generation in example

# Assuming these modules exist based on our plan and exploration
from graphrag.core.visual_analyzer import VisualAnalyzer
from graphrag.connectors.qdrant_connection import QdrantConnection
# from graphrag.core.video_ingest import VideoIngestor # Needed for actual data processing, but maybe not direct import for now
# Potentially import Neo4j connection if direct queries are made from here
# from graphrag.connectors.neo4j_connection import get_connection as get_neo4j_connection


logger = logging.getLogger(__name__)

class TrendType(Enum):
    INGREDIENT = "ingredient"
    TECHNIQUE = "technique"
    AESTHETIC = "aesthetic"
    FORMAT = "format"
    UNKNOWN = "unknown"

class TrendLifecycleStage(Enum):
    EMERGING = "emerging"
    PEAKING = "peaking"
    DECLINING = "declining"
    STABLE = "stable" # Added for completeness
    UNKNOWN = "unknown"

class TrendDetector:
    def __init__(self, visual_analyzer: VisualAnalyzer, qdrant_conn: QdrantConnection, neo4j_conn=None):
        self.visual_analyzer = visual_analyzer
        self.qdrant_conn = qdrant_conn
        self.neo4j_conn = neo4j_conn # For querying frame/video metadata
        self.qdrant_collection_name = "visual_trends_embeddings" # Example collection name

        # Ensure Qdrant collection exists (example vector size)
        # In a real scenario, vector_size would come from the embedding model
        # For now, let's assume visual_analyzer will handle embedding generation and know the size.
        # self.qdrant_conn.create_collection(self.qdrant_collection_name, vector_size=512)
        logger.info("TrendDetector initialized.")

    def _generate_visual_embeddings(self, frame_path: str) -> np.ndarray | None:
        """
        Placeholder for generating visual embeddings for a frame.
        This would likely involve a call to visual_analyzer or a dedicated embedding model.
        """
        logger.debug(f"Placeholder: Generating embedding for {frame_path}")
        # In a real implementation, this would use a model (e.g., CLIP, ResNet)
        # For now, returning a random vector or None
        # Example: return self.visual_analyzer.generate_embedding(frame_path)
        # Let's assume VisualAnalyzer is updated to have a method like 'generate_embedding'
        # For now, a dummy implementation:
        if hasattr(self.visual_analyzer, 'generate_embedding'):
            try:
                embedding = self.visual_analyzer.generate_embedding(frame_path)
                logger.info(f"Successfully generated embedding for {frame_path} via VisualAnalyzer.")
                return embedding
            except Exception as e:
                logger.error(f"VisualAnalyzer failed to generate embedding for {frame_path}: {e}")
                return None
        else:
            # Fallback dummy embedding if VisualAnalyzer doesn't have the method yet
            logger.warning(f"VisualAnalyzer does not have 'generate_embedding' method. Using dummy random embedding for {frame_path}.")
            return np.random.rand(512).astype(np.float32) # Example fixed size

    def process_frame_for_visual_similarity(self, frame_id: str, frame_path: str, frame_metadata: dict):
        """
        Processes a single frame to extract/generate visual features,
        stores them in Qdrant, and updates visual similarity clusters.

        Args:
            frame_id (str): Unique identifier for the frame (e.g., from Neo4j).
            frame_path (str): Filesystem path to the frame image.
            frame_metadata (dict): Additional metadata (timestamp, video_id, creator_id).
        """
        logger.info(f"Processing frame {frame_id} at {frame_path} for visual similarity.")

        embedding = self._generate_visual_embeddings(frame_path)
        if embedding is None:
            logger.warning(f"Could not generate embedding for frame {frame_id}. Skipping.")
            return None

        # Store in Qdrant
        # Qdrant requires list of vectors, ids, and payloads
        # q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, frame_id)) # Deterministic UUID for Qdrant
        payload = {
            "original_frame_id": frame_id,
            "video_id": frame_metadata.get("video_id"),
            "timestamp": frame_metadata.get("timestamp"),
            "creator_id": frame_metadata.get("creator_id"),
            # Potentially add other visual analysis results (tags, objects) from VisualAnalyzer
        }

        # The QdrantConnection upsert_vectors expects lists.
        # self.qdrant_conn.upsert_vectors(
        #     collection_name=self.qdrant_collection_name,
        #     vectors=[embedding.tolist()], # Ensure vector is list of floats
        #     ids=[q_id], # Qdrant expects string or int IDs, our conn might handle UUID conversion
        #     metadata=[payload]
        # )
        # Using the string ID directly as per QdrantConnection's _string_to_uuid internal handling
        self.qdrant_conn.upsert_vectors(
            collection_name=self.qdrant_collection_name,
            vectors=[embedding.tolist()],
            ids=[frame_id], # Using original frame_id, QdrantConnection will make it a UUID
            metadata=[payload]
        )
        logger.info(f"Upserted embedding for frame {frame_id} to Qdrant collection '{self.qdrant_collection_name}'.")

        # Placeholder for actual clustering logic.
        # In a real scenario, you might:
        # 1. Query Qdrant for similar vectors to this new one.
        # 2. If enough similar vectors are found, form or update a cluster.
        # 3. Store cluster information (e.g., in Neo4j or another store).
        # For now, we'll just log that it's a placeholder.
        logger.debug("Placeholder: Visual similarity clustering logic would run here.")
        # Example: find_similar_and_cluster(embedding, frame_id, frame_metadata)
        return {"frame_id": frame_id, "embedding_stored": True, "qdrant_id": frame_id} # Return some status

    def detect_temporal_patterns(self, visual_cluster_id: str, start_time=None, end_time=None) -> dict:
        """
        Analyzes the appearance frequency of a given visual cluster over time.
        This would typically involve querying data sources (like Neo4j) for frame occurrences
        belonging to this cluster within the time range.

        Args:
            visual_cluster_id (str): The ID of the visual cluster to analyze.
            start_time: Optional start time for analysis.
            end_time: Optional end time for analysis.

        Returns:
            dict: Contains pattern data like {'timestamps': [...], 'frequency_per_day': {...}}.
        """
        logger.info(f"Detecting temporal patterns for visual cluster {visual_cluster_id}.")
        # Placeholder: In a real system, query Neo4j or another DB for frames in this cluster
        # and their timestamps.
        # e.g., MATCH (f:Frame)-[:BELONGS_TO_CLUSTER]->(c:VisualCluster {id: visual_cluster_id}) RETURN f.timestamp

        # Mock data for demonstration
        mock_timestamps = sorted([
            1672531200, 1672617600, 1672704000, # Jan 1, 2, 3 2023
            1672790400, 1672790500, # Jan 4 2023 (two occurrences)
            1675209600, 1675296000  # Feb 1, 2 2023
        ])

        # Example: Calculate frequency (e.g., per day)
        frequency_per_day = defaultdict(int)
        for ts in mock_timestamps:
            day_str = str(ts // (24*60*60)) # Simple day bucketing from epoch
            frequency_per_day[day_str] += 1

        logger.debug(f"Temporal pattern for cluster {visual_cluster_id}: {frequency_per_day}")
        return {"cluster_id": visual_cluster_id, "timestamps": mock_timestamps, "frequency_per_day": dict(frequency_per_day)}

    def calculate_trend_velocity(self, temporal_pattern_data: dict) -> float:
        """
        Calculates the velocity (rate of change) of a trend based on its temporal pattern.

        Args:
            temporal_pattern_data (dict): Output from detect_temporal_patterns.
                                         Expected to have 'frequency_per_day' or similar.

        Returns:
            float: A score representing the trend's velocity.
                   Positive for increasing, negative for decreasing.
        """
        logger.info(f"Calculating trend velocity for cluster {temporal_pattern_data.get('cluster_id')}.")

        # Placeholder: Simple velocity calculation (e.g., change in frequency over last two periods)
        # A more robust implementation would use regression or more advanced time-series analysis.
        freq_data = temporal_pattern_data.get("frequency_per_day", {})
        if not freq_data or len(freq_data) < 2:
            logger.warning("Not enough data points to calculate velocity.")
            return 0.0

        # Assuming day_str keys are sortable (they are if derived from incrementing timestamps)
        sorted_days = sorted(freq_data.keys())

        # Example: (freq_latest_period - freq_previous_period) / number_of_periods_diff
        # This is a very naive approach.
        if len(sorted_days) >= 2:
            latest_day_freq = freq_data[sorted_days[-1]]
            previous_day_freq = freq_data[sorted_days[-2]]
            # Ensure we don't divide by zero if periods are consecutive integers
            # Here, we assume periods are just sequential entries, so diff is 1 "period"
            velocity = float(latest_day_freq - previous_day_freq)
        else:
            velocity = 0.0

        logger.debug(f"Calculated velocity: {velocity}")
        return velocity

    def track_cross_creator_adoption(self, visual_cluster_id: str, start_time=None, end_time=None) -> dict:
        """
        Tracks how a visual trend/cluster spreads across different creators.

        Args:
            visual_cluster_id (str): The ID of the visual cluster.
            start_time: Optional start time for analysis.
            end_time: Optional end time for analysis.

        Returns:
            dict: {'creator_adoption_count': int, 'creators': [creator_id_1, ...]}.
        """
        logger.info(f"Tracking cross-creator adoption for visual cluster {visual_cluster_id}.")
        # Placeholder: Query data source for creators associated with frames in this cluster.
        # e.g., MATCH (f:Frame)-[:BELONGS_TO_CLUSTER]->(c:VisualCluster {id: visual_cluster_id})
        #       MATCH (vid:Video {id: f.video_id})
        #       RETURN DISTINCT vid.creator_id

        # Mock data
        mock_creators = ["creatorA", "creatorB", "creatorA", "creatorC"]
        unique_creators = set(mock_creators)

        logger.debug(f"Cross-creator adoption for {visual_cluster_id}: {len(unique_creators)} creators - {list(unique_creators)}")
        return {"cluster_id": visual_cluster_id, "creator_adoption_count": len(unique_creators), "creators": list(unique_creators)}

    def classify_trend(self, visual_cluster_id: str, associated_data: dict) -> TrendType:
        """
        Classifies a trend based on its characteristics.

        Args:
            visual_cluster_id (str): The ID of the visual cluster representing the trend.
            associated_data (dict): Data that might help in classification (e.g., object tags from
                                    VisualAnalyzer, text descriptions, dominant colors).
                                    Example: {'tags': ['food', 'recipe'], 'objects': ['bowl', 'spoon']}

        Returns:
            TrendType: The classified type of the trend.
        """
        logger.info(f"Classifying trend for visual cluster {visual_cluster_id}.")
        # Placeholder: Simple rule-based classification
        tags = associated_data.get("tags", [])
        objects = associated_data.get("objects", [])

        if any(t in ["food", "recipe", "ingredient", "dish"] for t in tags) or            any(o in ["plate", "bowl", "pan"] for o in objects):
            trend_type = TrendType.INGREDIENT # Could also be TECHNIQUE if cooking is detected
        elif "dance" in tags or "filter" in tags or "transition" in tags:
            trend_type = TrendType.FORMAT
        elif "aesthetic" in tags or "vintage" in tags or "minimalist" in tags:
            trend_type = TrendType.AESTHETIC
        else:
            trend_type = TrendType.UNKNOWN

        logger.debug(f"Classified trend {visual_cluster_id} as {trend_type.value}")
        return trend_type

    def calculate_confidence_score(self, trend_data: dict) -> float:
        """
        Calculates a confidence score for an identified trend.

        Args:
            trend_data (dict): A dictionary containing various metrics about the trend,
                               e.g., {'velocity': 2.0, 'adoption_count': 5, 'total_occurrences': 50}.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """
        logger.info(f"Calculating confidence score for trend.")
        # Placeholder: Simple scoring based on available metrics
        score = 0.0
        # Example: score based on number of occurrences and creator adoption, normalized
        occurrences = trend_data.get("total_occurrences", 0)
        adoption = trend_data.get("adoption_count", 0)
        velocity = trend_data.get("velocity", 0.0)

        # Normalize and weight (very basic example)
        score_occurrences = min(occurrences / 100.0, 1.0) # Max score at 100 occurrences
        score_adoption = min(adoption / 10.0, 1.0)     # Max score at 10 creators
        score_velocity = min(abs(velocity) / 5.0, 1.0) if velocity != 0 else 0 # Max score at velocity 5

        # Weighted average
        score = (0.4 * score_occurrences + 0.4 * score_adoption + 0.2 * score_velocity)
        score = max(0.0, min(score, 1.0)) # Ensure it's between 0 and 1

        logger.debug(f"Calculated confidence score: {score:.2f}")
        return score

    def detect_trend_lifecycle_stage(self, trend_data: dict) -> TrendLifecycleStage:
        """
        Detects the current lifecycle stage of a trend.

        Args:
            trend_data (dict): Metrics about the trend, esp. 'velocity' and 'total_occurrences' or 'age'.

        Returns:
            TrendLifecycleStage: The detected lifecycle stage.
        """
        logger.info(f"Detecting trend lifecycle stage.")
        velocity = trend_data.get("velocity", 0.0)
        total_occurrences = trend_data.get("total_occurrences", 0)
        # Could also use trend "age" or duration of observation

        stage = TrendLifecycleStage.UNKNOWN

        if velocity > 1.0 and total_occurrences < 50 : # Arbitrary thresholds
            stage = TrendLifecycleStage.EMERGING
        elif velocity > 0.5 and total_occurrences >= 50 :
            stage = TrendLifecycleStage.PEAKING
        elif abs(velocity) <= 0.5 and total_occurrences > 20: # Low velocity but consistent
             stage = TrendLifecycleStage.STABLE
        elif velocity < -0.5 :
            stage = TrendLifecycleStage.DECLINING

        logger.debug(f"Detected lifecycle stage: {stage.value}")
        return stage

    def identify_trends_from_data(self, processed_frames_data: list[dict]) -> list[dict]:
        """
        Main orchestration method to identify and characterize trends from processed frame data.
        This method would iterate over frames, manage visual similarity processing,
        and then for identified clusters/patterns, run the detection and characterization.

        Args:
            processed_frames_data (list[dict]): A list of dictionaries, where each dict
                                                contains info about a frame like
                                                {'frame_id': 'xyz', 'frame_path': '/path/to/frame.jpg',
                                                 'metadata': {'timestamp': 123, 'video_id': 'v1', 'creator_id': 'c1'}}

        Returns:
            list[dict]: A list of identified trends, each with its characteristics.
        """
        logger.info(f"Starting trend identification from {len(processed_frames_data)} processed frames.")

        # 1. Process all frames for visual similarity and store embeddings
        qdrant_results = []
        for frame_data in processed_frames_data:
            # Assume VisualAnalyzer has already processed the frame for objects, tags etc.
            # and that info is part of frame_data['metadata'] or accessible via frame_id
            # For now, we just pass the core frame info.
            # In a real scenario, ensure Qdrant collection is created with correct vector size
            # This might be done once in __init__ if embedding size is fixed and known.
            # self.qdrant_conn.create_collection(self.qdrant_collection_name, vector_size=EMBEDDING_DIM)

            res = self.process_frame_for_visual_similarity(
                frame_id=frame_data['frame_id'],
                frame_path=frame_data['frame_path'],
                frame_metadata=frame_data['metadata']
            )
            if res:
                qdrant_results.append(res)

        logger.info(f"Processed {len(qdrant_results)} frames for Qdrant storage.")

        # 2. Placeholder: Identify visual clusters from Qdrant
        # This is a complex step. In a real system, it might involve:
        # - Iterating through stored embeddings in Qdrant.
        # - Performing similarity searches to find neighbors.
        # - Applying a clustering algorithm (e.g., DBSCAN on demand, or regular batch clustering).
        # - Assigning cluster IDs to frames.
        # For now, let's mock some visual clusters.
        # Assume each "cluster" is identified by one of its frame_ids for simplicity,
        # or a newly generated cluster_id.

        # Mocking: Let's assume we found a few "key frames" that represent potential clusters
        # and we use their frame_ids as proxies for cluster_ids for now.
        # This is a MAJOR simplification.

        # Create a map from frame_id to frame_data for easy lookup during mock cluster seed selection
        frame_data_map = {f['frame_id']: f for f in processed_frames_data}

        mock_potential_cluster_seeds = []
        if qdrant_results:
            # Let's say first few unique video frames are our "cluster seeds"
            seen_video_ids_for_seeds = set()
            for res in qdrant_results: # res['qdrant_id'] is the frame_id used for upsert
                # Need to get the video_id from the original processed_frames_data using the frame_id
                original_frame_data = frame_data_map.get(res['qdrant_id'])
                if original_frame_data:
                    video_id = original_frame_data['metadata'].get('video_id')
                    if video_id not in seen_video_ids_for_seeds:
                        mock_potential_cluster_seeds.append(res['qdrant_id']) # Using frame_id as cluster_id proxy
                        seen_video_ids_for_seeds.add(video_id)
                    if len(mock_potential_cluster_seeds) >= 3: break # Max 3 mock clusters


        identified_trends = []
        # Let's use the mock_potential_cluster_seeds as our "cluster_ids"
        # In reality, cluster_ids would come from a clustering process.

        for cluster_id_proxy in mock_potential_cluster_seeds: # cluster_id_proxy is actually a frame_id here
            logger.info(f"--- Analyzing mock visual cluster based on seed frame: {cluster_id_proxy} ---")

            # In a real scenario, you'd get all frames belonging to this cluster.
            # For the mock, we assume the cluster's properties are primarily derived from this seed frame's metadata
            # and some global mock data.

            # Simulate getting associated data for classification (e.g. from VisualAnalyzer via Neo4j node for the frame)
            # This would be based on the actual visual content of frames in the cluster.
            # For now, use some mock tags based on the frame_id for variety.
            mock_associated_data_for_classification = {'tags': [], 'objects': []}
            # Get the original frame_path for heuristic classification
            original_frame_info = frame_data_map.get(cluster_id_proxy, {})
            frame_path_for_classification = original_frame_info.get('frame_path', "")

            if "food" in frame_path_for_classification:
                 mock_associated_data_for_classification['tags'].extend(['food', 'recipe'])
                 mock_associated_data_for_classification['objects'].extend(['bowl'])
            elif "dance" in frame_path_for_classification:
                 mock_associated_data_for_classification['tags'].extend(['dance', 'filter'])
            else: # Default if no specific keyword in path
                 mock_associated_data_for_classification['tags'].extend(['aesthetic', 'vintage'])


            temporal_patterns = self.detect_temporal_patterns(cluster_id_proxy)
            velocity = self.calculate_trend_velocity(temporal_patterns)
            adoption = self.track_cross_creator_adoption(cluster_id_proxy) # Uses mock data internally for now

            trend_type = self.classify_trend(cluster_id_proxy, mock_associated_data_for_classification)

            # For confidence and lifecycle, we need more aggregated data about the cluster
            # Let's use the mock data from adoption and temporal_patterns
            num_occurrences = sum(temporal_patterns.get('frequency_per_day', {}).values())

            trend_summary_data = {
                "cluster_id_proxy": cluster_id_proxy, # This is actually a frame_id in this mock
                "velocity": velocity,
                "adoption_count": adoption.get("creator_adoption_count", 0),
                "total_occurrences": num_occurrences,
                "example_creators": adoption.get("creators", []),
                "temporal_frequency": temporal_patterns.get("frequency_per_day"),
                "type": trend_type,
                # 'visual_characteristics': mock_associated_data_for_classification # Could be too verbose
            }

            confidence = self.calculate_confidence_score(trend_summary_data)
            lifecycle = self.detect_trend_lifecycle_stage(trend_summary_data)

            trend_summary_data["confidence_score"] = confidence
            trend_summary_data["lifecycle_stage"] = lifecycle.value # Store the string value

            identified_trends.append(trend_summary_data)
            logger.info(f"Identified trend candidate: {trend_summary_data}")

        logger.info(f"Completed trend identification. Found {len(identified_trends)} potential trends (mocked clusters).")
        return identified_trends


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting TrendDetector standalone example.")

    # Mock VisualAnalyzer and QdrantConnection
    class MockVisualAnalyzer:
        def __init__(self):
            logger.info("MockVisualAnalyzer initialized.")

        def generate_embedding(self, frame_path: str) -> np.ndarray:
            logger.info(f"MockVisualAnalyzer: Generating dummy embedding for {frame_path}")
            return np.random.rand(512).astype(np.float32) # Must match expected size

        def analyze_frame(self, frame_path: str, use_gemini: bool = True) -> dict:
            # Mock the output of the full analysis to get tags/objects
            logger.info(f"MockVisualAnalyzer: Analyzing frame {frame_path}")
            mock_analysis = {
                "frame_path": frame_path,
                "detected_objects": [{'type': 'object', 'description': 'mock_object', 'confidence': 0.9}],
                "scene_description": {'type': 'scene_description', 'description': 'A mock scene.'},
                "errors": []
            }
            if "food" in frame_path: # Simple heuristic for mock data
                mock_analysis["detected_objects"].append({'type': 'object', 'description': 'plate'})
            return mock_analysis


    class MockQdrantConnection:
        def __init__(self, url=None, api_key=None):
            self.collections = {}
            logger.info("MockQdrantConnection initialized.")

        def create_collection(self, collection_name=None, vector_size=None):
            if collection_name not in self.collections:
                self.collections[collection_name] = {"vectors": [], "metadata": [], "ids": []}
                logger.info(f"MockQdrant: Created collection '{collection_name}' with vector size {vector_size}.")
            return True

        def upsert_vectors(self, collection_name, vectors, ids, metadata=None):
            if collection_name not in self.collections:
                # Create collection on the fly if it doesn't exist, assuming a default vector size if not provided earlier
                self.create_collection(collection_name, vector_size=len(vectors[0]) if vectors else 512)

            self.collections[collection_name]["vectors"].extend(vectors)
            self.collections[collection_name]["ids"].extend(ids)
            self.collections[collection_name]["metadata"].extend(metadata if metadata else [{} for _ in ids])
            logger.info(f"MockQdrant: Upserted {len(vectors)} vectors to '{collection_name}'. Total: {len(self.collections[collection_name]['ids'])}.")
            return True

        def search(self, collection_name, query_vector, limit=10, filter_condition=None):
            logger.info(f"MockQdrant: Searching in '{collection_name}' (returning empty list for now).")
            return []

    mock_va = MockVisualAnalyzer()
    mock_qdrant = MockQdrantConnection()

    trend_detector = TrendDetector(visual_analyzer=mock_va, qdrant_conn=mock_qdrant)

    trend_detector.qdrant_conn.create_collection(
        collection_name=trend_detector.qdrant_collection_name,
        vector_size=512
    )

    dummy_frames_data = []
    creators = ["creatorX", "creatorY", "creatorZ"]
    videos_per_creator = 2
    frames_per_video = 3
    base_timestamp = int(time.time()) - (10 * 24 * 60 * 60)

    for c_idx, creator_id in enumerate(creators):
        for v_idx in range(videos_per_creator):
            video_id = f"vid_{c_idx}_{v_idx}"
            for f_idx in range(frames_per_video):
                frame_id = f"{video_id}_frame_{f_idx}"
                frame_path_suffix = ""
                if c_idx == 0 and v_idx == 0 and f_idx == 0: frame_path_suffix = "_food"
                elif c_idx == 1 and v_idx == 0 and f_idx == 0: frame_path_suffix = "_dance"

                frame_path = f"/tmp/dummy_frames/{frame_id}{frame_path_suffix}.jpg" #nosec
                timestamp = base_timestamp + (c_idx * videos_per_creator + v_idx) * (24*60*60) + f_idx * 3600

                dummy_frames_data.append({
                    "frame_id": frame_id,
                    "frame_path": frame_path,
                    "metadata": {
                        "timestamp": timestamp,
                        "video_id": video_id,
                        "creator_id": creator_id,
                    }
                })

    logger.info(f"Created {len(dummy_frames_data)} dummy frame data entries for testing.")

    identified_trends = trend_detector.identify_trends_from_data(dummy_frames_data)

    logger.info(f"
--- Main Example Finished: Identified {len(identified_trends)} Trends ---")
    for i, trend in enumerate(identified_trends):
        logger.info(f"Trend {i+1}:")
        logger.info(f"  Cluster ID (proxy): {trend.get('cluster_id_proxy')}")
        logger.info(f"  Type: {trend.get('type').value if trend.get('type') else 'N/A'}")
        logger.info(f"  Velocity: {trend.get('velocity'):.2f}")
        logger.info(f"  Adoption Count: {trend.get('adoption_count')}")
        logger.info(f"  Total Occurrences: {trend.get('total_occurrences')}")
        logger.info(f"  Confidence Score: {trend.get('confidence_score'):.2f}")
        logger.info(f"  Lifecycle Stage: {trend.get('lifecycle_stage')}")
        logger.info(f"  Example Creators: {trend.get('example_creators')}")
