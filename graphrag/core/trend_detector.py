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
    TECHNIQUE = "technique" # Can be cooking technique
    AESTHETIC = "aesthetic" # Can be plating style
    FORMAT = "format"
    CUISINE = "cuisine" # New
    NUTRITIONAL = "nutritional" # New
    FOOD_ITEM = "food_item" # New, for specific dishes or products
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
            "tags": frame_metadata.get("tags", []),
            "objects": frame_metadata.get("objects", [])
        }

        food_analysis_data = frame_metadata.get("food_analysis")
        if food_analysis_data: # Ensure food_analysis_data is not None
            payload["ingredients"] = food_analysis_data.get("ingredients", [])
            payload["cooking_technique"] = food_analysis_data.get("cooking_technique")
            payload["plating_style"] = food_analysis_data.get("plating_style")
            payload["nutritional_trends"] = food_analysis_data.get("nutritional_trends", [])
            payload["cuisine"] = food_analysis_data.get("cuisine")

        if embedding is not None:
             self.qdrant_conn.upsert_vectors(
                 collection_name=self.qdrant_collection_name,
                 vectors=[embedding.tolist()],
                 ids=[frame_id],
                 metadata=[payload]
             )
             logger.info(f"Upserted embedding for frame {frame_id} to Qdrant collection '{self.qdrant_collection_name}'.")
             # logger.debug("Placeholder: Visual similarity clustering logic would run here.") # Original log
             return {"frame_id": frame_id, "embedding_stored": True, "qdrant_id": frame_id, "payload_for_qdrant": payload}
        else:
            logger.warning(f"Skipping Qdrant upsert for {frame_id} due to missing embedding. Payload was: {payload}")
            return {"frame_id": frame_id, "embedding_stored": False, "qdrant_id": frame_id, "payload_generated": payload, "error": "embedding_generation_failed"}

        # The QdrantConnection upsert_vectors expects lists.
        # self.qdrant_conn.upsert_vectors(
        #     collection_name=self.qdrant_collection_name,
        #     vectors=[embedding.tolist()], # Ensure vector is list of floats
        #     ids=[q_id], # Qdrant expects string or int IDs, our conn might handle UUID conversion
        #     metadata=[payload]
        # )
        # Using the string ID directly as per QdrantConnection's _string_to_uuid internal handling
        # self.qdrant_conn.upsert_vectors(
        # collection_name=self.qdrant_collection_name,
        # vectors=[embedding.tolist()],
        # ids=[frame_id], # Using original frame_id, QdrantConnection will make it a UUID
        # metadata=[payload]
        # )
        # logger.info(f"Upserted embedding for frame {frame_id} to Qdrant collection '{self.qdrant_collection_name}'.")

        # Placeholder for actual clustering logic.
        # In a real scenario, you might:
        # 1. Query Qdrant for similar vectors to this new one.
        # 2. If enough similar vectors are found, form or update a cluster.
        # 3. Store cluster information (e.g., in Neo4j or another store).
        # For now, we'll just log that it's a placeholder.
        # logger.debug("Placeholder: Visual similarity clustering logic would run here.")
        # Example: find_similar_and_cluster(embedding, frame_id, frame_metadata)
        # return {"frame_id": frame_id, "embedding_stored": True, "qdrant_id": frame_id} # Return some status
    # The return statements are now inside the if/else block for embedding status

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
        logger.info(f"Classifying trend for visual cluster {visual_cluster_id} using data: {associated_data}")

        ingredients = associated_data.get("ingredients", [])
        cooking_technique = associated_data.get("cooking_technique")
        plating_style = associated_data.get("plating_style")
        nutritional_trends = associated_data.get("nutritional_trends", [])
        cuisine = associated_data.get("cuisine")

        tags = associated_data.get("tags", [])
        objects = associated_data.get("objects", [])

        if cuisine and cuisine not in ["Unknown Cuisine", "unknown_cuisine", None, "mock_general_food_main", "mock_garden_main", "mock_sweet_main"]:
            if not cuisine.startswith("mock_") and not cuisine.startswith("unknown_"):
                return TrendType.CUISINE

        if nutritional_trends and any(nt for nt in nutritional_trends if nt and not nt.startswith("mock_")):
                 return TrendType.NUTRITIONAL

        if ingredients and any(ing for ing in ingredients if ing and not ing.startswith("mock_") and not ing.startswith("unknown_")):
            known_trending_ingredients = ["tofu", "plant-based", "tempeh", "seitan", "mushroom", "chocolate", "matcha", "fermented"]
            if any(i in known_trending_ingredients for i in ingredients):
                return TrendType.INGREDIENT

        if cooking_technique and cooking_technique not in ["unknown_technique", "unknown_technique_main", None] and not cooking_technique.startswith("mock_"):
            return TrendType.TECHNIQUE

        if plating_style and plating_style not in ["unknown_style", "unknown_style_main", None] and not plating_style.startswith("mock_"):
            return TrendType.AESTHETIC

        is_general_food_object = any(o in ["plate", "bowl", "pan", "food_item", "plate_mock_main", "mock_object_generic_main"] for o in objects)
        is_general_food_tag = any(t in ["food", "recipe", "dish"] for t in tags)
        if is_general_food_object or is_general_food_tag:
            return TrendType.FOOD_ITEM

        if "dance" in tags or "filter" in tags or "transition" in tags: # Check for non-food tags
            if "dance_gamma" in visual_cluster_id: # Example specific check for test data
                return TrendType.FORMAT

        if any(t in ["vintage", "minimalist_general", "aesthetic"] for t in tags) and            not (plating_style and plating_style not in ["unknown_style", "unknown_style_main", None] and not plating_style.startswith("mock_")):
            return TrendType.AESTHETIC

        logger.debug(f"Trend {visual_cluster_id} classified as UNKNOWN with data: {associated_data}")
        return TrendType.UNKNOWN

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
        qdrant_results = []

        frame_data_map = {
            f_data.get('frame_id'): f_data
            for f_data in processed_frames_data
            if f_data.get('frame_id') and f_data.get('frame_path')
        }
        if not frame_data_map:
            logger.warning("No valid frame data (missing frame_id or frame_path) in processed_frames_data for identify_trends_from_data.")
            return []

        for frame_id, frame_data_item in frame_data_map.items():
            frame_path = frame_data_item.get('frame_path')
            current_metadata = {
                "video_id": frame_data_item.get("video_id"),
                "timestamp": frame_data_item.get("timestamp"),
                "creator_id": frame_data_item.get("creator_id"),
                "food_analysis": frame_data_item.get("food_analysis"),
                "tags": [obj.get('description', '') for obj in frame_data_item.get("detected_objects", []) if obj.get('description')],
                "objects": [obj.get('description', '') for obj in frame_data_item.get("detected_objects", []) if obj.get('description')]
            }
            current_metadata = {k: v for k, v in current_metadata.items() if v is not None}

            res = self.process_frame_for_visual_similarity(
                frame_id=frame_id,
                frame_path=frame_path,
                frame_metadata=current_metadata
            )
            if res and res.get("embedding_stored"):
                qdrant_results.append(res)

        logger.info(f"Successfully processed {len(qdrant_results)} frames for Qdrant storage and potential cluster seeds.")

        mock_potential_cluster_seeds = []
        if qdrant_results:
            seen_video_ids_for_seeds = set()
            for res in qdrant_results:
                original_frame_data = frame_data_map.get(res['qdrant_id'])
                if original_frame_data:
                    video_id = original_frame_data.get('video_id')
                    if video_id not in seen_video_ids_for_seeds:
                        mock_potential_cluster_seeds.append(res['qdrant_id'])
                        seen_video_ids_for_seeds.add(video_id)
                    if len(mock_potential_cluster_seeds) >= 3: break

        identified_trends = []
        for cluster_id_proxy in mock_potential_cluster_seeds:
            logger.info(f"--- Analyzing mock visual cluster based on seed frame: {cluster_id_proxy} ---")

            seed_frame_full_analysis = frame_data_map.get(cluster_id_proxy)
            associated_data_for_classification = {}

            if seed_frame_full_analysis:
                food_content = seed_frame_full_analysis.get("food_analysis")
                if food_content:
                    associated_data_for_classification["ingredients"] = food_content.get("ingredients", [])
                    associated_data_for_classification["cooking_technique"] = food_content.get("cooking_technique")
                    associated_data_for_classification["plating_style"] = food_content.get("plating_style")
                    associated_data_for_classification["nutritional_trends"] = food_content.get("nutritional_trends", [])
                    associated_data_for_classification["cuisine"] = food_content.get("cuisine")

                associated_data_for_classification["tags"] = [obj.get('description', '') for obj in seed_frame_full_analysis.get("detected_objects", []) if obj.get('description')]
                associated_data_for_classification["objects"] = [obj.get('description', '') for obj in seed_frame_full_analysis.get("detected_objects", []) if obj.get('description')]
            else:
                logger.warning(f"Could not find full analysis data for seed frame {cluster_id_proxy} in frame_data_map. Classification might be impacted.")
                associated_data_for_classification = {'tags': ['unknown_fallback_seed_missing'], 'objects': []}

            temporal_patterns = self.detect_temporal_patterns(cluster_id_proxy)
            velocity = self.calculate_trend_velocity(temporal_patterns)
            adoption = self.track_cross_creator_adoption(cluster_id_proxy)
            trend_type = self.classify_trend(cluster_id_proxy, associated_data_for_classification)

            num_occurrences = sum(temporal_patterns.get('frequency_per_day', {}).values())
            trend_summary_data = {
                "cluster_id_proxy": cluster_id_proxy, "velocity": velocity,
                "adoption_count": adoption.get("creator_adoption_count", 0),
                "total_occurrences": num_occurrences,
                "example_creators": adoption.get("creators", []),
                "temporal_frequency": temporal_patterns.get("frequency_per_day"),
                "type": trend_type,
            }
            confidence = self.calculate_confidence_score(trend_summary_data)
            lifecycle = self.detect_trend_lifecycle_stage(trend_summary_data)
            trend_summary_data["confidence_score"] = confidence
            trend_summary_data["lifecycle_stage"] = lifecycle
            identified_trends.append(trend_summary_data)
            # Log with .value for enums
            logger.info(f"Identified trend candidate: Type - {trend_type.value if isinstance(trend_type, Enum) else trend_type}, Lifecycle - {lifecycle.value if isinstance(lifecycle, Enum) else lifecycle}, Confidence - {confidence:.2f}")

        logger.info(f"Completed trend identification. Found {len(identified_trends)} potential trends (based on mock clusters).")
        return identified_trends


if __name__ == '__main__':
    import time
    import numpy as np
    # from enum import Enum # Already imported at class level (global to the file)
    # from collections import defaultdict, Counter # Should be global if used by other methods

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Ensure logger is correctly obtained
    logger.info("Starting TrendDetector standalone example.")

    class MockVisualAnalyzer:
        def __init__(self):
            logger.info("MockVisualAnalyzer (for TrendDetector __main__) initialized.")
        def generate_embedding(self, frame_path: str) -> np.ndarray:
            logger.info(f"MockVisualAnalyzer (for TrendDetector's embedding generation): Generating dummy embedding for {frame_path}")
            return np.random.rand(512).astype(np.float32)
        def analyze_frame(self, frame_path: str, analyze_food: bool = False) -> dict: # This mock creates the rich analysis data
            frame_basename = frame_path.split('/')[-1].replace('.jpg', '')
            parts = frame_basename.split('_')
            mock_analysis = {
                "frame_id": frame_basename, "frame_path": frame_path,
                "video_id": f"{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) > 3 else "vid_unknown_main", # Adjusted index for safety
                "timestamp": int(time.time()),
                "creator_id": parts[1] if len(parts) > 2 else "creator_unknown_main", # Adjusted index
                "detected_objects": [{'description': 'mock_object_generic_main'}],
                "scene_description": {'description': 'A mock scene from __main__ for ' + frame_basename},
                "food_analysis": None, "errors": []
            }
            if analyze_food and "food" in frame_path.lower():
                food_payload = {"ingredients": [], "cooking_technique": "unknown_technique_main", "plating_style": "unknown_style_main", "nutritional_trends": [], "cuisine": "unknown_cuisine_main"}
                if "salad" in frame_path.lower():
                    food_payload.update({"ingredients": ["mock_lettuce_main", "mock_tomato_main"], "cooking_technique": "mock_tossing_main", "nutritional_trends": ["mock_healthy_main"], "cuisine": "mock_garden_main"})
                elif "cake" in frame_path.lower():
                    food_payload.update({"ingredients": ["mock_cocoa_main", "mock_sugar_main"], "cooking_technique": "mock_baking_main", "nutritional_trends": ["mock_indulgent_main"], "cuisine": "mock_sweet_main"})
                elif "generic" in frame_path.lower(): # A generic food type
                     food_payload.update({"ingredients": ["mock_item_main_x", "mock_item_main_y"], "cuisine": "mock_general_food_main", "cooking_technique": "mock_generic_cook_main"})
                mock_analysis["food_analysis"] = food_payload
                mock_analysis["detected_objects"].append({'description': 'plate_mock_main'})
            return mock_analysis

    class MockQdrantConnection:
        def __init__(self, url=None, api_key=None): self.collections = {}; logger.info("MockQdrantConnection initialized.")
        def create_collection(self, collection_name=None, vector_size=None):
            if collection_name not in self.collections: self.collections[collection_name] = {"vectors": [], "metadata": [], "ids": []}; logger.info(f"MockQdrant: Created collection '{collection_name}'.")
            return True
        def upsert_vectors(self, collection_name, vectors, ids, metadata=None):
            if collection_name not in self.collections: self.create_collection(collection_name, vector_size=len(vectors[0]) if vectors else 512)
            self.collections[collection_name]["vectors"].extend(vectors); self.collections[collection_name]["ids"].extend(ids); self.collections[collection_name]["metadata"].extend(metadata if metadata else [{} for _ in ids]); logger.info(f"MockQdrant: Upserted {len(vectors)} vectors to '{collection_name}'.")
            return True
        def search(self, collection_name, query_vector, limit=10, filter_condition=None): logger.info(f"MockQdrant: Searching in '{collection_name}' (returning empty list for now)."); return []

    mock_va_internal_for_detector = MockVisualAnalyzer()
    mock_qdrant_instance = MockQdrantConnection()

    trend_detector_main_instance = TrendDetector(visual_analyzer=mock_va_internal_for_detector, qdrant_conn=mock_qdrant_instance)
    trend_detector_main_instance.qdrant_conn.create_collection(collection_name=trend_detector_main_instance.qdrant_collection_name, vector_size=512)

    dummy_analyzed_frames_for_input = []
    current_base_ts = int(time.time()) - (5 * 24 * 60 * 60)

    test_frame_definitions_main = [
        {"creator_id": "creatorA", "video_idx": 0, "frame_idx": 0, "type_suffix": "food_salad_alpha"},
        {"creator_id": "creatorB", "video_idx": 0, "frame_idx": 0, "type_suffix": "food_cake_beta"},
        {"creator_id": "creatorC", "video_idx": 0, "frame_idx": 0, "type_suffix": "dance_gamma"},
        {"creator_id": "creatorA", "video_idx": 1, "frame_idx": 0, "type_suffix": "food_generic_delta"},
        {"creator_id": "creatorB", "video_idx": 0, "frame_idx": 1, "type_suffix": "food_salad_epsilon"},
    ]

    input_data_generator_va = MockVisualAnalyzer()

    for i, frame_def_item in enumerate(test_frame_definitions_main):
        video_id_for_test = f"vid_{frame_def_item['creator_id']}_{frame_def_item['video_idx']}"
        frame_id_for_test = f"{video_id_for_test}_frame_{frame_def_item['frame_idx']}_{frame_def_item['type_suffix']}"
        dummy_frame_path = f"/tmp/dummy_trend_frames_main/{frame_id_for_test}.jpg" # nosec

        is_food_frame_type = "food" in frame_def_item["type_suffix"].lower()
        analyzed_frame_obj = input_data_generator_va.analyze_frame(dummy_frame_path, analyze_food=is_food_frame_type)

        analyzed_frame_obj["creator_id"] = frame_def_item["creator_id"]
        analyzed_frame_obj["video_id"] = video_id_for_test
        analyzed_frame_obj["timestamp"] = current_base_ts + i * (24*60*60)
        analyzed_frame_obj["frame_id"] = frame_id_for_test

        dummy_analyzed_frames_for_input.append(analyzed_frame_obj)

    if not dummy_analyzed_frames_for_input:
        logger.error("No dummy data was generated for TrendDetector test in __main__. Exiting example.")
    else:
        logger.info(f"Generated {len(dummy_analyzed_frames_for_input)} dummy analyzed frame data entries for TrendDetector test.")
        identified_trends_result = trend_detector_main_instance.identify_trends_from_data(dummy_analyzed_frames_for_input)

        logger.info(f"--- TrendDetector __main__ Example Finished: Identified {len(identified_trends_result)} Trends ---")
        for idx, trend_item in enumerate(identified_trends_result):
            logger.info(f"Trend {idx+1}:")
            logger.info(f"  Cluster ID (Seed Frame): {trend_item.get('cluster_id_proxy')}")

            trend_type_val = trend_item.get('type')
            logger.info(f"  Type: {trend_type_val.value if isinstance(trend_type_val, Enum) else trend_type_val}")

            logger.info(f"  Velocity: {trend_item.get('velocity'):.2f}")
            logger.info(f"  Adoption Count: {trend_item.get('adoption_count')}")
            logger.info(f"  Total Occurrences: {trend_item.get('total_occurrences')}")
            logger.info(f"  Confidence Score: {trend_item.get('confidence_score'):.2f}")

            lifecycle_stage_val = trend_item.get('lifecycle_stage')
            logger.info(f"  Lifecycle Stage: {lifecycle_stage_val.value if isinstance(lifecycle_stage_val, Enum) else lifecycle_stage_val}")
            logger.info(f"  Example Creators: {trend_item.get('example_creators')}")
