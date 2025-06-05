import logging
import os
import json # For example usage
from .food_analyzer import FoodAnalyzer

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.food_analyzer = FoodAnalyzer()
        self.logger.info("VisualAnalyzer initialized. (Models would be loaded here in a real scenario)")

    def analyze_frame_rf_detr(self, frame_path: str) -> list[dict]:
        self.logger.info(f"Analyzing frame with RF-DETR (placeholder) for: {frame_path}")
        if not os.path.exists(frame_path):
            self.logger.error(f"RF-DETR: Frame not found at {frame_path}")
            return [{"type": "error", "description": "Frame not found for RF-DETR analysis"}]

        # Placeholder: Mock detected objects
        mock_objects = [
            {'type': 'object', 'description': 'cat', 'confidence': 0.95, 'bounding_box': [10, 20, 50, 60]},
            {'type': 'object', 'description': 'table', 'confidence': 0.88, 'bounding_box': [5, 50, 150, 100]}
        ]
        self.logger.debug(f"RF-DETR mock analysis for {frame_path}: {mock_objects}")
        return mock_objects

    def describe_scene_blip2(self, frame_path: str) -> dict:
        self.logger.info(f"Describing scene with BLIP-2 (placeholder) for: {frame_path}")
        if not os.path.exists(frame_path):
            self.logger.error(f"BLIP-2: Frame not found at {frame_path}")
            return {'type': 'error', 'description': 'Frame not found for BLIP-2 analysis'}

        # Placeholder: Mock scene description
        mock_description = {'type': 'scene_description', 'description': 'A cat sitting on a table in a sunlit room.'}
        self.logger.debug(f"BLIP-2 mock description for {frame_path}: {mock_description}")
        return mock_description

    def enhance_analysis_gemini(self, frame_path: str, existing_analysis: dict) -> dict:
        self.logger.info(f"Enhancing analysis with Gemini Vision API (placeholder) for: {frame_path}")
        if not os.path.exists(frame_path):
            self.logger.error(f"Gemini: Frame not found at {frame_path}")
            return {'type': 'error', 'description': 'Frame not found for Gemini enhancement'}

        # Placeholder: Mock enhanced analysis
        # This could use existing_analysis to make it seem more real
        enhanced_details = {
            'type': 'enhanced_description',
            'description': f"The scene at {frame_path} appears to involve a {existing_analysis.get('detected_objects', [{}])[0].get('description', 'unknown object')}. The room has vintage decor.",
            'additional_tags': ['pet', 'indoor', 'vintage', 'mock_tag']
        }
        self.logger.debug(f"Gemini mock enhancement for {frame_path}: {enhanced_details}")
        return enhanced_details

    def analyze_food_image(self, frame_path: str) -> dict:
        """
        Performs food-specific analysis on an image.
        """
        self.logger.info(f"Performing food-specific analysis for frame: {frame_path}")
        food_analysis_results = {}
        errors = []

        try:
            food_analysis_results["ingredients"] = self.food_analyzer.recognize_ingredients(frame_path)
        except Exception as e:
            self.logger.error(f"Error during ingredient recognition for {frame_path}: {e}")
            errors.append(f"Ingredient recognition failed: {str(e)}")

        try:
            food_analysis_results["cooking_technique"] = self.food_analyzer.classify_cooking_technique(frame_path)
        except Exception as e:
            self.logger.error(f"Error during cooking technique classification for {frame_path}: {e}")
            errors.append(f"Cooking technique classification failed: {str(e)}")

        try:
            food_analysis_results["plating_style"] = self.food_analyzer.detect_plating_style(frame_path)
        except Exception as e:
            self.logger.error(f"Error during plating style detection for {frame_path}: {e}")
            errors.append(f"Plating style detection failed: {str(e)}")

        try:
            food_analysis_results["nutritional_trends"] = self.food_analyzer.detect_nutritional_trends(frame_path)
        except Exception as e:
            self.logger.error(f"Error during nutritional trend detection for {frame_path}: {e}")
            errors.append(f"Nutritional trend detection failed: {str(e)}")

        try:
            food_analysis_results["cuisine"] = self.food_analyzer.classify_cuisine(frame_path)
        except Exception as e:
            self.logger.error(f"Error during cuisine classification for {frame_path}: {e}")
            errors.append(f"Cuisine classification failed: {str(e)}")

        if errors:
            food_analysis_results["errors"] = errors # Store errors within the food_analysis_results structure

        self.logger.info(f"Food analysis for {frame_path} completed.")
        return food_analysis_results

    def enhance_food_description_gemini(self, frame_path: str, food_analysis_data: dict) -> dict:
        """
        Enhances food descriptions using Gemini Vision API with fine-tuned prompts.
        Placeholder implementation.
        """
        self.logger.info(f"Enhancing food description with Gemini Vision API (placeholder) for: {frame_path}")
        # Ensure os is imported if this is the first use in this method, or rely on module-level import
        if not os.path.exists(frame_path):
            self.logger.error(f"Gemini Food: Frame not found at {frame_path}")
            return {'type': 'error', 'description': 'Frame not found for Gemini food enhancement'}

        ingredients_str = ", ".join(food_analysis_data.get("ingredients", ["unknown ingredients"]))
        technique = food_analysis_data.get("cooking_technique", "unknown technique")
        plating = food_analysis_data.get("plating_style", "unknown style")
        cuisine = food_analysis_data.get("cuisine", "unknown cuisine")

        enhanced_description = {
            'type': 'enhanced_food_description',
            'description': f"This dish, likely a {cuisine} preparation, appears to be made with {ingredients_str}. "                            f"It seems to have been prepared using {technique} and is presented in a {plating} style. "                            f"The visual cues suggest a focus on fresh components and vibrant colors (mock Gemini description).",
            'sensory_tags': ['fresh', 'vibrant', 'appetizing_mock_tag'],
            'estimated_calories': "300-500 kcal (mock estimate)"
        }
        self.logger.debug(f"Gemini mock food enhancement for {frame_path}: {enhanced_description}")
        return enhanced_description

    def create_unified_output(self, frame_path: str,
                              rf_detr_results: list[dict],
                              blip2_description: dict,
                              gemini_enhancements: dict = None,
                              food_analysis_results: dict = None, # New parameter
                              gemini_food_enhancements: dict = None # New parameter
                              ) -> dict:
        self.logger.debug(f"Creating unified output for frame: {frame_path}")

        unified_output = {
            "frame_path": frame_path,
            "detected_objects": rf_detr_results,
            "scene_description": blip2_description,
            "errors": []
        }

        if gemini_enhancements:
            unified_output["enhanced_analysis"] = gemini_enhancements

        if food_analysis_results:
            unified_output["food_analysis"] = food_analysis_results
            if "errors" in food_analysis_results and food_analysis_results["errors"]: # Check if errors list is not empty
                for err in food_analysis_results["errors"]:
                     unified_output["errors"].append(f"FoodAnalysis: {err}")
                # Not deleting food_analysis_results["errors"] to keep original structure

        if gemini_food_enhancements:
            unified_output["enhanced_food_description"] = gemini_food_enhancements
            if gemini_food_enhancements.get('type') == 'error':
                 unified_output["errors"].append(f"GeminiFood: {gemini_food_enhancements['description']}")

        if rf_detr_results and isinstance(rf_detr_results, list) and len(rf_detr_results) > 0 and rf_detr_results[0].get('type') == 'error':
            unified_output["errors"].append(f"RF-DETR: {rf_detr_results[0]['description']}")
        if blip2_description and blip2_description.get('type') == 'error':
            unified_output["errors"].append(f"BLIP-2: {blip2_description['description']}")
        if gemini_enhancements and gemini_enhancements.get('type') == 'error':
             unified_output["errors"].append(f"Gemini: {gemini_enhancements['description']}")

        if unified_output["errors"]:
            unified_output["errors"] = sorted(list(set(unified_output["errors"])))

        self.logger.info(f"Unified analysis for {frame_path} created.")
        return unified_output

    def analyze_frame(self, frame_path: str, use_gemini: bool = True, analyze_food: bool = False) -> dict:
        self.logger.info(f"Starting full analysis for frame: {frame_path} (Food analysis: {analyze_food})")
        errors_list = []

        rf_results = []
        blip_description = {}
        gemini_enhancement = None
        food_results = None
        gemini_food_enhancement = None

        try:
            rf_results = self.analyze_frame_rf_detr(frame_path)
        except Exception as e:
            self.logger.error(f"Error during RF-DETR analysis for {frame_path}: {e}")
            errors_list.append(f"RF-DETR analysis failed: {str(e)}")

        try:
            blip_description = self.describe_scene_blip2(frame_path)
        except Exception as e:
            self.logger.error(f"Error during BLIP-2 analysis for {frame_path}: {e}")
            errors_list.append(f"BLIP-2 analysis failed: {str(e)}")

        preliminary_analysis_for_gemini = {
            "detected_objects": rf_results if isinstance(rf_results, list) and not (rf_results and rf_results[0].get('type') == 'error') else [],
            "scene_description": blip_description if blip_description.get('type') != 'error' else {}
        }

        if use_gemini:
            try:
                gemini_enhancement = self.enhance_analysis_gemini(frame_path, preliminary_analysis_for_gemini)
            except Exception as e:
                self.logger.error(f"Error during Gemini enhancement for {frame_path}: {e}")
                errors_list.append(f"Gemini enhancement failed: {str(e)}")

        if analyze_food:
            try:
                food_results = self.analyze_food_image(frame_path)
            except Exception as e:
                self.logger.error(f"Error during Food Analysis for {frame_path}: {e}")
                errors_list.append(f"Food Analysis failed: {str(e)}")

            if use_gemini and food_results and not (food_results.get("errors") and len(food_results.get("ingredients", [])) == 0) :
                try:
                    gemini_food_enhancement = self.enhance_food_description_gemini(frame_path, food_results)
                except Exception as e:
                    self.logger.error(f"Error during Gemini food enhancement for {frame_path}: {e}")
                    errors_list.append(f"Gemini food enhancement failed: {str(e)}")

        unified_output = self.create_unified_output(
            frame_path,
            rf_results,
            blip_description,
            gemini_enhancement,
            food_results,
            gemini_food_enhancement
        )

        for err in errors_list:
            if err not in unified_output["errors"]:
                 unified_output["errors"].append(err)

        if unified_output["errors"]:
             unified_output["errors"] = sorted(list(set(unified_output["errors"])))

        if unified_output["errors"]:
            self.logger.warning(f"Completed analysis for {frame_path} with errors: {unified_output['errors']}")
        else:
            self.logger.info(f"Successfully completed analysis for frame: {frame_path}")

        return unified_output

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting VisualAnalyzer example.")

    dummy_frame_filename = "test_frame.jpg"
    dummy_food_frame_filename = "test_food_salad_frame.jpg"

    for fname in [dummy_frame_filename, dummy_food_frame_filename]:
        if not os.path.exists(fname):
            try:
                with open(fname, "w") as f:
                    f.write(f"This is a dummy image file for {fname}.")
                logger.info(f"Created dummy frame file: {fname}")
            except IOError as e:
                logger.error(f"Could not create dummy frame file {fname}: {e}")

    analyzer = VisualAnalyzer()

    logger.info(f"Analyzing dummy frame: {dummy_frame_filename} (with Gemini, no food)")
    analysis_result_with_gemini = analyzer.analyze_frame(dummy_frame_filename, use_gemini=True, analyze_food=False)
    logger.info(f"Analysis Result (with Gemini, no food):\n{json.dumps(analysis_result_with_gemini, indent=2)}")

    logger.info(f"\nAnalyzing dummy food frame: {dummy_food_frame_filename} (with Gemini, with food)")
    food_analysis_result = analyzer.analyze_frame(dummy_food_frame_filename, use_gemini=True, analyze_food=True)
    logger.info(f"Food Analysis Result (with Gemini, with food):\n{json.dumps(food_analysis_result, indent=2)}")

    logger.info(f"\nAnalyzing dummy food frame: {dummy_food_frame_filename} (no Gemini, with food)")
    food_analysis_no_gemini_result = analyzer.analyze_frame(dummy_food_frame_filename, use_gemini=False, analyze_food=True)
    logger.info(f"Food Analysis Result (no Gemini, with food):\n{json.dumps(food_analysis_no_gemini_result, indent=2)}")

    non_existent_file = "non_existent_frame.jpg"
    logger.info(f"\nAnalyzing non-existent frame: {non_existent_file}")
    analysis_non_existent = analyzer.analyze_frame(non_existent_file, analyze_food=True)
    logger.info(f"Analysis Result (non-existent file):\n{json.dumps(analysis_non_existent, indent=2)}")

    for fname in [dummy_frame_filename, dummy_food_frame_filename]:
        try:
            if os.path.exists(fname):
                os.remove(fname)
                logger.info(f"Cleaned up dummy frame file: {fname}")
        except Exception as e:
            logger.error(f"Error during cleanup of {fname}: {e}")

    logger.info("VisualAnalyzer example finished.")
