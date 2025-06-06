# graphrag.core.__init__.py

from .trend_detector import TrendDetector, TrendType, TrendLifecycleStage
from .trend_predictor import TrendPredictor
from .food_analyzer import FoodAnalyzer
from .marketing_insights import MarketingInsights

__all__ = [
    "TrendDetector",
    "TrendType",
    "TrendLifecycleStage",
    "TrendPredictor",
    "FoodAnalyzer",
    "MarketingInsights",
]
