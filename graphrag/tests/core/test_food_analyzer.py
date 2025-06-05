import unittest
import os
from graphrag.core.food_analyzer import FoodAnalyzer

class TestFoodAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = FoodAnalyzer()
        self.test_image_salad = "test_image_salad.jpg"
        self.test_image_cake_grill = "test_image_cake_grill_fancy.jpg"
        self.test_image_pasta = "italian_pasta_image.jpg"
        self.test_description_healthy = "a very healthy salad bowl"
        self.test_description_indulgent_cake = "an indulgent chocolate cake"

    def test_recognize_ingredients(self):
        self.assertEqual(self.analyzer.recognize_ingredients(self.test_image_salad), ["lettuce", "tomato", "cucumber"])
        self.assertEqual(self.analyzer.recognize_ingredients(self.test_image_cake_grill), ["flour", "sugar", "chocolate"])
        self.assertEqual(self.analyzer.recognize_ingredients("unknown_food.jpg"), ["unknown_ingredient_1", "unknown_ingredient_2"])

    def test_classify_cooking_technique(self):
        self.assertEqual(self.analyzer.classify_cooking_technique(self.test_image_cake_grill), "grilling")
        self.assertEqual(self.analyzer.classify_cooking_technique("fried_chicken.jpg"), "frying")
        self.assertEqual(self.analyzer.classify_cooking_technique("baked_bread.jpg"), "unknown_technique")

    def test_detect_plating_style(self):
        self.assertEqual(self.analyzer.detect_plating_style(self.test_image_cake_grill), "artistic")
        self.assertEqual(self.analyzer.detect_plating_style("simple_dish.jpg"), "minimalist")
        self.assertEqual(self.analyzer.detect_plating_style("another_dish.jpg"), "unknown_style")

    def test_define_food_trend_taxonomy(self):
        taxonomy = self.analyzer.define_food_trend_taxonomy()
        self.assertIn("categories", taxonomy)
        self.assertIn("trends", taxonomy)
        self.assertIsInstance(taxonomy["categories"], list)
        self.assertIsInstance(taxonomy["trends"], dict)
        self.assertTrue(len(taxonomy["categories"]) > 0)
        self.assertTrue(len(taxonomy["trends"].keys()) > 0)

    def test_detect_nutritional_trends(self):
        self.assertEqual(self.analyzer.detect_nutritional_trends(self.test_description_healthy), ["healthy", "low-calorie"])
        self.assertEqual(self.analyzer.detect_nutritional_trends(self.test_image_salad), ["healthy", "low-calorie"])
        self.assertEqual(self.analyzer.detect_nutritional_trends(self.test_description_indulgent_cake), ["indulgent", "high-calorie"])
        self.assertEqual(self.analyzer.detect_nutritional_trends(self.test_image_cake_grill), ["indulgent", "high-calorie"])
        self.assertEqual(self.analyzer.detect_nutritional_trends("a balanced meal.jpg"), ["balanced"])

    def test_classify_cuisine(self):
        self.assertEqual(self.analyzer.classify_cuisine(self.test_image_pasta), "Italian")
        self.assertEqual(self.analyzer.classify_cuisine("mexican_tacos.jpg"), "Mexican")
        self.assertEqual(self.analyzer.classify_cuisine("french_soup.jpg"), "Unknown Cuisine")

if __name__ == '__main__':
    unittest.main()
