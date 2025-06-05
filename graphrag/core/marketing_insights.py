import logging
import json # For pretty printing the dict in main
from collections import defaultdict
from graphrag.core.trend_detector import TrendDetector, TrendType, TrendLifecycleStage # Assuming these enums might be useful
from graphrag.core.trend_predictor import TrendPredictor
# Potentially add imports for Neo4jConnection or QdrantConnection if direct access is needed later

logger = logging.getLogger(__name__)

class MarketingInsights:
    def __init__(self, trend_detector: TrendDetector, trend_predictor: TrendPredictor, neo4j_conn=None, qdrant_conn=None):
        self.trend_detector = trend_detector
        self.trend_predictor = trend_predictor
        self.neo4j_conn = neo4j_conn
        self.qdrant_conn = qdrant_conn
        logger.info("MarketingInsights initialized.")

    def calculate_trend_impact(self, trend_data: dict, prediction_data: dict) -> dict:
        if not trend_data or not prediction_data:
            logger.warning("Trend data or prediction data is missing for impact calculation.")
            return {"views_potential": 0, "engagement_potential": 0, "longevity_potential": 0, "error": "Missing input data"}

        trend_identifier = trend_data.get('cluster_id_proxy') or trend_data.get('trend_id', 'N/A')
        logger.debug(f"Calculating trend impact for trend: {trend_identifier}")

        views_score = 0; velocity_score = 0; predicted_freq_score = 0; viral_score = 0 # Ensure defined for all paths
        try:
            occurrences_score = min(trend_data.get('total_occurrences', 0) / 200.0, 1.0)
            velocity_score = min(abs(trend_data.get('velocity', 0)) / 5.0, 1.0)
            predicted_freq_adj = prediction_data.get('creator_influenced_prediction', {}).get('predicted_next_period_frequency_adj_influence', 0)
            predicted_freq_score = min(predicted_freq_adj / 50.0, 1.0)
            viral_coeff = prediction_data.get('viral_coefficient', 0)
            viral_score = min(viral_coeff / 2.0, 1.0)
            creator_influence = prediction_data.get('creator_influenced_prediction', {}).get('creator_influence_avg_score', 0)
            influence_score = min(creator_influence / 1.0, 1.0)
            views_score = (0.25 * occurrences_score + 0.25 * velocity_score + 0.20 * predicted_freq_score +
                           0.15 * viral_score + 0.15 * influence_score) * 10
        except Exception as e: logger.error(f"Error calculating views_score for {trend_identifier}: {e}", exc_info=True)

        engagement_score = 0
        trend_type = trend_data.get('type')
        try:
            adoption_score = min(trend_data.get('adoption_count', 0) / 20.0, 1.0)
            trend_type_engagement_factor = 0.5
            if trend_type:
                if isinstance(trend_type, str):
                    trend_type_str = trend_type.upper()
                    if trend_type_str in TrendType.__members__: trend_type = TrendType[trend_type_str]
                    else: logger.warning(f"Unknown trend type string for {trend_identifier}: {trend_type}"); trend_type = None
                if isinstance(trend_type, TrendType): # Check if it became an enum
                    if trend_type in [TrendType.FORMAT, TrendType.TECHNIQUE]: trend_type_engagement_factor = 0.8
                    elif trend_type in [TrendType.AESTHETIC, TrendType.INGREDIENT]: trend_type_engagement_factor = 0.6
            engagement_score = (0.35 * adoption_score + 0.25 * velocity_score +
                                0.20 * viral_score + 0.20 * trend_type_engagement_factor) * 10
        except Exception as e: logger.error(f"Error calculating engagement_score for {trend_identifier}: {e}", exc_info=True)

        longevity_score = 0
        try:
            lifecycle_stage = trend_data.get('lifecycle_stage')
            if isinstance(lifecycle_stage, str):
                 lifecycle_stage_str = lifecycle_stage.upper()
                 if lifecycle_stage_str in TrendLifecycleStage.__members__: lifecycle_stage = TrendLifecycleStage[lifecycle_stage_str]
                 else: logger.warning(f"Unknown lifecycle_stage string for {trend_identifier}: {lifecycle_stage}"); lifecycle_stage = None
            lifecycle_factor = 0.5
            if isinstance(lifecycle_stage, TrendLifecycleStage): # Check if it became an enum
                if lifecycle_stage == TrendLifecycleStage.EMERGING: lifecycle_factor = 0.7
                elif lifecycle_stage == TrendLifecycleStage.PEAKING: lifecycle_factor = 0.8
                elif lifecycle_stage == TrendLifecycleStage.STABLE: lifecycle_factor = 0.9
                elif lifecycle_stage == TrendLifecycleStage.DECLINING: lifecycle_factor = 0.2
            trend_type_longevity_factor = 0.5
            if isinstance(trend_type, TrendType): # Check if it became an enum
                 if trend_type in [TrendType.CUISINE, TrendType.NUTRITIONAL]: trend_type_longevity_factor = 0.8
                 elif trend_type in [TrendType.INGREDIENT, TrendType.TECHNIQUE]: trend_type_longevity_factor = 0.6
                 elif trend_type in [TrendType.AESTHETIC, TrendType.FORMAT]: trend_type_longevity_factor = 0.3
            longevity_score = (0.4 * lifecycle_factor + 0.3 * predicted_freq_score +
                               0.3 * trend_type_longevity_factor) * 10
        except Exception as e: logger.error(f"Error calculating longevity_score for {trend_identifier}: {e}", exc_info=True)

        return {"views_potential": round(max(0, min(views_score, 10)), 1),
                "engagement_potential": round(max(0, min(engagement_score, 10)), 1),
                "longevity_potential": round(max(0, min(longevity_score, 10)), 1)}

    def analyze_competitor_trends(self, brand_id: str, competitor_ids_list: list[str], all_trends_data: list[dict]) -> dict:
        logger.debug(f"Analyzing competitor trends for brand '{brand_id}' against competitors: {competitor_ids_list}")
        if not all_trends_data:
            logger.warning("No trends data provided for competitive analysis.")
            return {"brand_adopted_trends": [], "competitor_adopted_trends": {}, "missing_trends_for_brand": {}, "overlap_with_competitors": {}, "exclusive_to_brand": [], "exclusive_to_competitors": {}, "error": "Missing all_trends_data"}
        brand_adopted_set = set(); competitor_adopted_map = {comp_id: set() for comp_id in competitor_ids_list}; all_competitors_adopted_set = set()
        for trend in all_trends_data:
            trend_id = trend.get('trend_id')
            if not trend_id: logger.warning(f"Skipping trend with missing 'trend_id': {trend}"); continue
            adopted_by = trend.get('adopted_by_creators', [])
            if not isinstance(adopted_by, list): logger.warning(f"Trend '{trend_id}' has invalid 'adopted_by_creators': {adopted_by}. Skipping."); continue
            if brand_id in adopted_by: brand_adopted_set.add(trend_id)
            for comp_id in competitor_ids_list:
                if comp_id in adopted_by: competitor_adopted_map[comp_id].add(trend_id); all_competitors_adopted_set.add(trend_id)
        missing_trends_for_brand, overlap_with_competitors, exclusive_to_competitors_map = {}, {}, {}
        for comp_id in competitor_ids_list:
            comp_set = competitor_adopted_map[comp_id]
            missing_trends_for_brand[comp_id] = sorted(list(comp_set - brand_adopted_set))
            overlap_with_competitors[comp_id] = sorted(list(brand_adopted_set.intersection(comp_set)))
            exclusive_to_competitors_map[comp_id] = sorted(list(comp_set - brand_adopted_set))
        exclusive_to_brand_list = sorted(list(brand_adopted_set - all_competitors_adopted_set))
        return {"brand_adopted_trends": sorted(list(brand_adopted_set)), "competitor_adopted_trends": {k: sorted(list(v)) for k, v in competitor_adopted_map.items()}, "missing_trends_for_brand": missing_trends_for_brand, "overlap_with_competitors": overlap_with_competitors, "exclusive_to_brand": exclusive_to_brand_list, "exclusive_to_competitors": exclusive_to_competitors_map}

    def identify_content_opportunities(self, brand_id: str, all_trends_data: list[dict], brand_profile: dict = None) -> dict:
        logger.debug(f"Identifying content opportunities for brand '{brand_id}'.")
        if not all_trends_data: return {"error": "Missing all_trends_data"}
        opportunities = defaultdict(list); brand_adopted_trends_set = set(); trend_details_map = {}
        for trend in all_trends_data:
            trend_id = trend.get('trend_id')
            if not trend_id: continue
            trend_details_map[trend_id] = trend
            if brand_id in trend.get('adopted_by_creators', []): brand_adopted_trends_set.add(trend_id)
        relevant_trend_types = set()
        if brand_profile and 'relevant_trend_types' in brand_profile:
            for tt in brand_profile['relevant_trend_types']:
                if isinstance(tt, str) and tt.upper() in TrendType.__members__: relevant_trend_types.add(TrendType[tt.upper()])
                elif isinstance(tt, TrendType): relevant_trend_types.add(tt)
        for trend_id, trend in trend_details_map.items():
            if trend_id in brand_adopted_trends_set: continue
            impact = self.calculate_trend_impact(trend, trend.get('prediction_data', {}))
            stage_str = trend.get('lifecycle_stage'); stage = None
            if isinstance(stage_str, str) and stage_str.upper() in TrendLifecycleStage.__members__: stage = TrendLifecycleStage[stage_str.upper()]
            elif isinstance(stage_str, TrendLifecycleStage): stage = stage_str
            if stage == TrendLifecycleStage.EMERGING and (impact['views_potential'] > 6 or impact['engagement_potential'] > 6) and impact['longevity_potential'] > 4: opportunities['emerging_high_potential'].append(trend)
            if len(trend.get('adopted_by_creators', [])) < 10 and (impact['views_potential'] > 7 or impact['engagement_potential'] > 7) and impact['longevity_potential'] > 5: opportunities['undersaturated_gems'].append(trend)
            type_str = trend.get('type'); current_trend_type = None
            if isinstance(type_str, str) and type_str.upper() in TrendType.__members__: current_trend_type = TrendType[type_str.upper()]
            elif isinstance(type_str, TrendType): current_trend_type = type_str
            if relevant_trend_types and current_trend_type in relevant_trend_types:
                if impact['views_potential'] > 5 or impact['engagement_potential'] > 5: opportunities['brand_alignment_strong'].append(trend)
            elif not relevant_trend_types and (impact['views_potential'] > 6 or impact['engagement_potential'] > 6): opportunities['brand_alignment_strong_general_fallback'].append(trend)
            if len(trend.get('adopted_by_creators', [])) <= 5 and (impact['views_potential'] > 7 or impact['engagement_potential'] > 7) and impact['longevity_potential'] > 6:
                if not any(t['trend_id'] == trend_id for t in opportunities['undersaturated_gems']): opportunities['competitive_whitespace'].append(trend)
        final_opportunities = {}
        for key, trend_list in opportunities.items():
            unique_trends = []; seen_ids = set()
            for t_obj in trend_list:
                if t_obj['trend_id'] not in seen_ids: unique_trends.append(t_obj); seen_ids.add(t_obj['trend_id'])
            final_opportunities[key] = unique_trends
        return final_opportunities

    def predict_roi_for_trend_adoption(self, trend_id: str, trend_data_for_impact: dict, prediction_data_for_impact: dict, estimated_investment_level: str) -> dict:
        logger.debug(f"Predicting ROI for trend '{trend_id}' with investment '{estimated_investment_level}'.")
        impact = self.calculate_trend_impact(trend_data_for_impact, prediction_data_for_impact)
        return_score = round(max(0, min((0.4*impact.get('views_potential',0) + 0.4*impact.get('engagement_potential',0) + 0.2*impact.get('longevity_potential',0)), 10)), 1)
        cost_map = {"low": 3.0, "medium": 6.0, "high": 9.0}; cost = cost_map.get(estimated_investment_level.lower(), 7.0)
        raw_roi = return_score / cost if cost > 0 else 0
        scaled_roi = round(max(0, min((raw_roi / (10.0/3.0)) * 10.0, 10)), 1)
        category = "High" if scaled_roi > 7 else "Medium" if scaled_roi > 4 else "Low"
        return {'trend_id': trend_id, 'estimated_investment': estimated_investment_level, 'potential_return_score': return_score, 'estimated_roi_score': scaled_roi, 'roi_category': category, 'justification': f"Return {return_score}/10, Investment '{estimated_investment_level}' (cost {cost}). ROI Category: {category} ({scaled_roi}/10)."}

    def get_trend_dashboard_data(self, all_trends_data: list[dict]) -> dict:
        logger.debug("Generating trend dashboard data.")
        dashboard = defaultdict(list)
        if not all_trends_data: return dict(dashboard)
        for trend in all_trends_data:
            trend_id = trend.get('trend_id', 'Unknown'); name = trend.get('name', trend_id)
            stage_str = trend.get('lifecycle_stage'); velocity = trend.get('velocity', 0); stage = None
            if isinstance(stage_str, str) and stage_str.upper() in TrendLifecycleStage.__members__: stage = TrendLifecycleStage[stage_str.upper()]
            elif isinstance(stage_str, TrendLifecycleStage): stage = stage_str
            summary = {'id': trend_id, 'name': name, 'type': str(trend.get('type', 'Unknown')), 'stage': str(stage.value if stage else 'Unknown'), 'velocity': velocity, 'occurrences': trend.get('total_occurrences', 0)}
            if 'prediction_data' in trend and 'early_warning_assessment' in trend['prediction_data']: summary['warning_level'] = trend['prediction_data']['early_warning_assessment'].get('warning_level')
            if stage == TrendLifecycleStage.PEAKING and velocity > 1: dashboard['trending_now'].append(summary)
            elif stage == TrendLifecycleStage.EMERGING and velocity > 0: dashboard['emerging_soon'].append(summary)
            elif stage == TrendLifecycleStage.STABLE: dashboard['stable_trends'].append(summary)
            elif stage == TrendLifecycleStage.DECLINING: dashboard['declining_trends'].append(summary)
        return dict(dashboard)

    def get_demo_scenario_data(self, scenario_name: str, all_trends_data: list[dict], brand_id: str = None, brand_profile: dict = None, competitor_ids_list: list[str] = None, selected_trend_id: str = None, investment_level: str = "medium") -> dict:
        logger.debug(f"Generating demo data for scenario: {scenario_name}")
        if scenario_name == "WhatsAboutToTrend":
            hot_trends = []
            for trend in all_trends_data:
                pred = trend.get('prediction_data', {}); warning = pred.get('early_warning_assessment', {}).get('warning_level')
                stage_str = trend.get('lifecycle_stage'); stage = None
                if isinstance(stage_str, str) and stage_str.upper() in TrendLifecycleStage.__members__: stage = TrendLifecycleStage[stage_str.upper()]
                elif isinstance(stage_str, TrendLifecycleStage): stage = stage_str
                if warning in ['high', 'critical']: hot_trends.append({**trend, 'reason': 'Strong Early Warning'})
                elif stage == TrendLifecycleStage.EMERGING and trend.get('velocity',0) > 2.0: hot_trends.append({**trend, 'reason': 'Emerging High Velocity'})
            return {"title": "What's About to Trend?", "trends": hot_trends[:10]}
        elif scenario_name == "TrendAnatomy":
            if not selected_trend_id: return {"error": "selected_trend_id required"}
            trend = next((t for t in all_trends_data if t.get('trend_id') == selected_trend_id), None)
            if not trend: return {"error": f"Trend {selected_trend_id} not found."}
            impact = self.calculate_trend_impact(trend, trend.get('prediction_data', {}))
            roi = self.predict_roi_for_trend_adoption(selected_trend_id, trend, trend.get('prediction_data', {}), investment_level)
            return {"title": f"Trend Anatomy: {trend.get('name', selected_trend_id)}", "details": trend, "impact_analysis": impact, "roi_prediction": roi}
        elif scenario_name == "FindMyNiche":
            if not brand_id or not competitor_ids_list: return {"error": "brand_id and competitor_ids_list required"}
            opps = self.identify_content_opportunities(brand_id, all_trends_data, brand_profile)
            niche_trends = opps.get('brand_alignment_strong', []) + opps.get('competitive_whitespace', []) + opps.get('undersaturated_gems', [])
            unique_niche = []; seen_ids = set()
            for t in niche_trends:
                if t['trend_id'] not in seen_ids: unique_niche.append(t); seen_ids.add(t['trend_id'])
            return {"title": f"Find My Niche for {brand_id}", "recommended_niches": unique_niche[:10], "opportunities_summary": {k: len(v) for k,v in opps.items()}}
        elif scenario_name == "ContentRecommendation":
            if not brand_id: return {"error": "brand_id required"}
            opps = self.identify_content_opportunities(brand_id, all_trends_data, brand_profile)
            candidates = opps.get('emerging_high_potential', []) + opps.get('brand_alignment_strong', []) + opps.get('undersaturated_gems', [])
            recs = []
            for trend in candidates:
                roi = self.predict_roi_for_trend_adoption(trend['trend_id'], trend, trend.get('prediction_data', {}), investment_level)
                if roi['estimated_roi_score'] > 5: recs.append({'trend': trend, 'roi': roi})
            recs.sort(key=lambda x: x['roi']['estimated_roi_score'], reverse=True)
            final_recs = []; seen_ids = set()
            for r in recs:
                if r['trend']['trend_id'] not in seen_ids: final_recs.append(r); seen_ids.add(r['trend']['trend_id'])
                if len(final_recs) >=5: break
            return {"title": f"Content Recommendations for {brand_id}", "recommendations": final_recs}
        else: return {"error": f"Unknown demo scenario: {scenario_name}"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("MarketingInsights script executed directly for testing.")

    class MockTrendDetectorGlobal:
        def __init__(self): self.TrendType = TrendType; self.TrendLifecycleStage = TrendLifecycleStage
    class MockTrendPredictorGlobal: pass
    mock_detector_main = MockTrendDetectorGlobal()
    mock_predictor_main = MockTrendPredictorGlobal()
    marketing_insights_instance = MarketingInsights(mock_detector_main, mock_predictor_main)
    logger.info("MarketingInsights instance created with global mock dependencies.")

    # Minimal test data for brevity in this example
    sample_trends_main = [
        {'trend_id': 't1', 'name': 'Trend Alpha', 'type': "AESTHETIC", 'lifecycle_stage': "EMERGING", 'velocity': 3.0, 'total_occurrences': 20, 'prediction_data': {'early_warning_assessment': {'warning_level': 'high'}, 'creator_influenced_prediction': {'predicted_next_period_frequency_adj_influence': 50}, 'viral_coefficient': 1.8}, 'adopted_by_creators': ['c1', 'brandMain']},
        {'trend_id': 't2', 'name': 'Trend Beta', 'type': "FORMAT", 'lifecycle_stage': "PEAKING", 'velocity': 5.0, 'total_occurrences': 150, 'prediction_data': {'creator_influenced_prediction': {'predicted_next_period_frequency_adj_influence': 20}, 'viral_coefficient': 0.5}, 'adopted_by_creators': ['c2']},
        {'trend_id': 't3', 'name': 'Trend Gamma', 'type': TrendType.INGREDIENT, 'lifecycle_stage': TrendLifecycleStage.STABLE, 'velocity': 0.5, 'total_occurrences': 300, 'prediction_data': {}, 'adopted_by_creators': ['c3']},
        {'trend_id': 't4', 'name': 'Trend Delta (Declining)', 'type': "TECHNIQUE", 'lifecycle_stage': "DECLINING", 'velocity': -1.0, 'total_occurrences': 50, 'prediction_data': {}, 'adopted_by_creators': ['c4']},
    ]

    logger.info("\n--- Testing get_trend_dashboard_data ---")
    dashboard_output = marketing_insights_instance.get_trend_dashboard_data(sample_trends_main)
    logger.info(f"Trend Dashboard Data: {json.dumps(dashboard_output, indent=2)}")

    logger.info("\n--- Testing get_demo_scenario_data ---")
    scenario_whats_hot = marketing_insights_instance.get_demo_scenario_data("WhatsAboutToTrend", sample_trends_main)
    logger.info(f"Demo Scenario - What's About to Trend: {json.dumps(scenario_whats_hot, indent=2)}")
    scenario_anatomy = marketing_insights_instance.get_demo_scenario_data("TrendAnatomy", sample_trends_main, selected_trend_id='t1')
    logger.info(f"Demo Scenario - Trend Anatomy (t1): {json.dumps(scenario_anatomy, indent=2)}")
    scenario_niche = marketing_insights_instance.get_demo_scenario_data("FindMyNiche", sample_trends_main, brand_id='brandMain', brand_profile={'relevant_trend_types': ["FORMAT", "INGREDIENT"]}, competitor_ids_list=['compA'])
    logger.info(f"Demo Scenario - Find My Niche: {json.dumps(scenario_niche, indent=2)}")
    scenario_recs = marketing_insights_instance.get_demo_scenario_data("ContentRecommendation", sample_trends_main, brand_id='brandMain', brand_profile={'relevant_trend_types': ["AESTHETIC"]}, investment_level="low")
    logger.info(f"Demo Scenario - Content Recommendation: {json.dumps(scenario_recs, indent=2)}")
    scenario_unknown = marketing_insights_instance.get_demo_scenario_data("UnknownScenario", sample_trends_main)
    logger.info(f"Demo Scenario - Unknown: {json.dumps(scenario_unknown, indent=2)}")
