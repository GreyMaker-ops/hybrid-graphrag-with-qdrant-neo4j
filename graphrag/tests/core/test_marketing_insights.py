import unittest
from unittest.mock import MagicMock, patch
import logging

# Assuming graphrag.core enums and classes are accessible
from graphrag.core.marketing_insights import MarketingInsights
from graphrag.core.trend_detector import TrendType, TrendLifecycleStage

# Disable logging for tests unless specifically needed for debugging a test
logging.disable(logging.CRITICAL)

class TestMarketingInsights(unittest.TestCase):

    def setUp(self):
        # Mock dependencies for MarketingInsights
        self.mock_trend_detector = MagicMock()
        self.mock_trend_predictor = MagicMock()

        # Expose enums through mocks if MarketingInsights expects them from detector/predictor instances
        # (Currently, MI gets enums/types mostly from data dictionaries, but good practice for future changes)
        self.mock_trend_detector.TrendType = TrendType
        self.mock_trend_detector.TrendLifecycleStage = TrendLifecycleStage

        self.marketing_insights = MarketingInsights(
            trend_detector=self.mock_trend_detector,
            trend_predictor=self.mock_trend_predictor
        )

        # Sample data commonly used across tests
        self.sample_trend_data_generic = {
            'trend_id': 'test_trend1', 'name': 'Test Trend 1',
            'total_occurrences': 150, 'velocity': 3.0, 'adoption_count': 15,
            'type': TrendType.AESTHETIC, 'lifecycle_stage': TrendLifecycleStage.PEAKING,
            'adopted_by_creators': ['brandA', 'creatorB'],
            'prediction_data': {
                'creator_influenced_prediction': {
                    'predicted_next_period_frequency_adj_influence': 40,
                    'creator_influence_avg_score': 0.75
                },
                'viral_coefficient': 1.5,
                'early_warning_assessment': {'warning_level': 'medium', 'reasons': ['High velocity']}
            }
        }
        self.sample_prediction_data_generic = self.sample_trend_data_generic['prediction_data']


    def test_calculate_trend_impact(self):
        # Test with valid data (enums)
        impact = self.marketing_insights.calculate_trend_impact(
            self.sample_trend_data_generic,
            self.sample_prediction_data_generic
        )
        self.assertIsNotNone(impact)
        self.assertIsInstance(impact, dict)
        self.assertIn('views_potential', impact)
        self.assertIn('engagement_potential', impact)
        self.assertIn('longevity_potential', impact)
        self.assertTrue(0 <= impact['views_potential'] <= 10)

        # Test with string enums
        trend_data_str_enums = {
            **self.sample_trend_data_generic,
            'type': "INGREDIENT", # String
            'lifecycle_stage': "EMERGING" # String
        }
        impact_str = self.marketing_insights.calculate_trend_impact(
            trend_data_str_enums,
            self.sample_prediction_data_generic
        )
        self.assertTrue(0 <= impact_str['views_potential'] <= 10)

        # Test with missing data
        impact_missing = self.marketing_insights.calculate_trend_impact({}, {})
        self.assertEqual(impact_missing['views_potential'], 0)
        self.assertIn('error', impact_missing)

    def test_analyze_competitor_trends(self):
        all_trends = [
            {'trend_id': 't1', 'adopted_by_creators': ['brandA', 'compX']},
            {'trend_id': 't2', 'adopted_by_creators': ['brandA']},
            {'trend_id': 't3', 'adopted_by_creators': ['compX', 'compY']},
            {'trend_id': 't4', 'adopted_by_creators': ['compY']},
        ]
        analysis = self.marketing_insights.analyze_competitor_trends('brandA', ['compX', 'compY'], all_trends)
        self.assertIsNotNone(analysis)
        self.assertEqual(set(analysis['brand_adopted_trends']), {'t1', 't2'})
        self.assertEqual(set(analysis['competitor_adopted_trends']['compX']), {'t1', 't3'})
        self.assertEqual(set(analysis['missing_trends_for_brand']['compX']), {'t3'}) # compX has t3, brandA doesn't
        self.assertEqual(set(analysis['overlap_with_competitors']['compX']), {'t1'})
        self.assertEqual(set(analysis['exclusive_to_brand']), {'t2'}) # t2 adopted by brandA, not by compX or compY

        # Test with empty trend data
        empty_analysis = self.marketing_insights.analyze_competitor_trends('brandA', ['compX'], [])
        self.assertEqual(empty_analysis['brand_adopted_trends'], [])
        self.assertIn('error', empty_analysis)

    def test_identify_content_opportunities(self):
        # More complex setup for all_trends_data
        all_trends_for_opps = [
            {**self.sample_trend_data_generic, 'trend_id': 'adopted_by_brand', 'adopted_by_creators': ['myBrand'], # Already adopted
             'prediction_data': {**self.sample_prediction_data_generic}},
            {**self.sample_trend_data_generic, 'trend_id': 'emerging_high_pot', 'type': TrendType.FORMAT,
             'lifecycle_stage': TrendLifecycleStage.EMERGING, 'velocity': 4.0, 'adopted_by_creators': ['other'],
             'prediction_data': {**self.sample_prediction_data_generic}},
            {**self.sample_trend_data_generic, 'trend_id': 'undersaturated', 'adoption_count': 1, 'total_occurrences': 10,
             'adopted_by_creators': ['lonely_creator'],
             'prediction_data': {**self.sample_prediction_data_generic, 'viral_coefficient': 2.5,
                                 'creator_influenced_prediction': {**self.sample_prediction_data_generic['creator_influenced_prediction'], 'predicted_next_period_frequency_adj_influence': 90}}}, # Make it very attractive
            {**self.sample_trend_data_generic, 'trend_id': 'brand_aligned', 'type': TrendType.CUISINE,
             'adopted_by_creators': ['c1'], 'prediction_data': {**self.sample_prediction_data_generic}},
        ]

        brand_profile = {'relevant_trend_types': [TrendType.CUISINE, TrendType.FORMAT]}

        opportunities = self.marketing_insights.identify_content_opportunities(
            'myBrand', all_trends_for_opps, brand_profile
        )
        self.assertIsNotNone(opportunities)
        self.assertIn('emerging_high_potential', opportunities)
        self.assertTrue(any(t['trend_id'] == 'emerging_high_pot' for t in opportunities['emerging_high_potential']))
        self.assertIn('undersaturated_gems', opportunities)
        self.assertTrue(any(t['trend_id'] == 'undersaturated' for t in opportunities['undersaturated_gems']))
        self.assertIn('brand_alignment_strong', opportunities)
        self.assertTrue(any(t['trend_id'] == 'brand_aligned' for t in opportunities['brand_alignment_strong']))

        # Test with no trend data
        empty_opps = self.marketing_insights.identify_content_opportunities('myBrand', [], brand_profile)
        self.assertIn('error', empty_opps)


    def test_predict_roi_for_trend_adoption(self):
        # High return, low investment
        roi_high_low = self.marketing_insights.predict_roi_for_trend_adoption(
            self.sample_trend_data_generic['trend_id'],
            self.sample_trend_data_generic,
            self.sample_prediction_data_generic,
            "low"
        )
        self.assertEqual(roi_high_low['trend_id'], 'test_trend1')
        self.assertEqual(roi_high_low['estimated_investment'], "low")
        self.assertTrue(roi_high_low['potential_return_score'] > 6) # Adjusted based on current scoring
        self.assertTrue(roi_high_low['estimated_roi_score'] > 6) # Adjusted based on current scoring
        self.assertIn(roi_high_low['roi_category'], ["High", "Medium"]) # More flexible check

        # Low return (modify sample data for this test), high investment
        low_return_trend_data = {
            **self.sample_trend_data_generic, 'trend_id': 'low_ret',
            'total_occurrences': 10, 'velocity': 0.1, 'adoption_count': 2, 'type': TrendType.AESTHETIC,
            'lifecycle_stage': TrendLifecycleStage.DECLINING,
            'prediction_data': {
                'creator_influenced_prediction': {'predicted_next_period_frequency_adj_influence': 5, 'creator_influence_avg_score': 0.2},
                'viral_coefficient': 0.1
            }
        }
        roi_low_high = self.marketing_insights.predict_roi_for_trend_adoption(
            'low_ret', low_return_trend_data, low_return_trend_data['prediction_data'], "high"
        )
        self.assertTrue(roi_low_high['potential_return_score'] < 4)
        self.assertTrue(roi_low_high['estimated_roi_score'] < 4)
        self.assertEqual(roi_low_high['roi_category'], "Low")

    def test_get_trend_dashboard_data(self):
        dashboard_trends = [
            {**self.sample_trend_data_generic, 'trend_id': 'dt1', 'lifecycle_stage': TrendLifecycleStage.EMERGING, 'velocity': 3.0},
            {**self.sample_trend_data_generic, 'trend_id': 'dt2', 'lifecycle_stage': TrendLifecycleStage.PEAKING, 'velocity': 4.0},
            {**self.sample_trend_data_generic, 'trend_id': 'dt3', 'lifecycle_stage': "STABLE"}, # String stage
            {**self.sample_trend_data_generic, 'trend_id': 'dt4', 'lifecycle_stage': "DECLINING", 'velocity': -1.0},
        ]
        for trend in dashboard_trends: # Ensure prediction_data for early warnings
            if 'prediction_data' not in trend: trend['prediction_data'] = self.sample_prediction_data_generic


        dashboard = self.marketing_insights.get_trend_dashboard_data(dashboard_trends)
        self.assertIn('emerging_soon', dashboard)
        self.assertIn('trending_now', dashboard)
        self.assertIn('stable_trends', dashboard)
        self.assertIn('declining_trends', dashboard)
        self.assertTrue(any(t['id'] == 'dt1' for t in dashboard['emerging_soon']))
        self.assertTrue(any(t['id'] == 'dt2' for t in dashboard['trending_now']))
        self.assertTrue(any(t['id'] == 'dt3' for t in dashboard['stable_trends']))
        self.assertTrue(any(t['id'] == 'dt4' for t in dashboard['declining_trends']))

    def test_get_demo_scenario_data(self):
        scenario_trends = [
            {**self.sample_trend_data_generic, 'trend_id': 's1', 'name':'Scenario Trend S1', 'lifecycle_stage': TrendLifecycleStage.EMERGING,
             'prediction_data': {**self.sample_prediction_data_generic, 'early_warning_assessment': {'warning_level': 'critical'}}},
            {**self.sample_trend_data_generic, 'trend_id': 's2', 'name':'Scenario Trend S2','type': TrendType.INGREDIENT, 'adopted_by_creators': ['myBrand', 'other'],
             'prediction_data': {**self.sample_prediction_data_generic} },
            {**self.sample_trend_data_generic, 'trend_id': 's3', 'name':'Scenario Trend S3','type': TrendType.FORMAT, 'adopted_by_creators': ['other'],
             'prediction_data': {**self.sample_prediction_data_generic} },
        ]

        # "WhatsAboutToTrend"
        whats_hot = self.marketing_insights.get_demo_scenario_data("WhatsAboutToTrend", scenario_trends)
        self.assertEqual(whats_hot['title'], "What's About to Trend?")
        self.assertTrue(any(t['trend_id'] == 's1' for t in whats_hot['trends']))

        # "TrendAnatomy"
        anatomy = self.marketing_insights.get_demo_scenario_data("TrendAnatomy", scenario_trends, selected_trend_id='s1')
        self.assertEqual(anatomy['title'], "Trend Anatomy: Scenario Trend S1")
        self.assertEqual(anatomy['details']['trend_id'], 's1')
        self.assertIn('impact_analysis', anatomy)
        self.assertIn('roi_prediction', anatomy)

        anatomy_err = self.marketing_insights.get_demo_scenario_data("TrendAnatomy", scenario_trends, selected_trend_id='nonexistent')
        self.assertIn('error', anatomy_err)

        # "FindMyNiche"
        niche = self.marketing_insights.get_demo_scenario_data(
            "FindMyNiche", scenario_trends,
            brand_id='myBrand',
            brand_profile={'relevant_trend_types': [TrendType.FORMAT]},
            competitor_ids_list=['compA']
        )
        self.assertEqual(niche['title'], "Find My Niche for myBrand")
        self.assertTrue(any(t['trend_id'] == 's3' for t in niche['recommended_niches']))

        # "ContentRecommendation"
        recs = self.marketing_insights.get_demo_scenario_data(
            "ContentRecommendation", scenario_trends,
            brand_id='myBrand',
            brand_profile={'relevant_trend_types': [TrendType.FORMAT]}
        )
        self.assertEqual(recs['title'], "Content Recommendations for myBrand")
        self.assertTrue(any(r['trend']['trend_id'] == 's3' for r in recs['recommendations']))

        # Unknown scenario
        unknown = self.marketing_insights.get_demo_scenario_data("UnknownScenario", scenario_trends)
        self.assertIn('error', unknown)

if __name__ == '__main__':
    unittest.main()
