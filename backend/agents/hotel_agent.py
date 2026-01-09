# backend/agents/hotel_agent.py
"""Hotel Agent with ReAct Pattern for business travel hotel search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import HotelQuery, HotelSearchResult, Hotel
from data.loaders import HotelDataLoader
from typing import List, Dict, Any, Optional
import json


class HotelAgent(BaseReActAgent):
    """Agentic Hotel Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="HotelAgent", agent_role="Hotel Booking Specialist",
            model_name=model_name, max_iterations=5, verbose=verbose
        )
        self.loader = HotelDataLoader()
        self.tools = self._register_tools()
        self.search_history: List[Dict] = []
    
    def _should_stop_early(self, observation: str) -> bool:
        obs_lower = observation.lower()
        has_hotels = self.state.get_belief("available_hotels") is not None
        signals = ["top 3", "best hotels", "recommend", "final", "selection complete", "comparison of"]
        return has_hotels and any(s in obs_lower for s in signals)
    
    def _extract_best_result_from_state(self) -> dict:
        """Extract diverse hotel options across quality tiers for PolicyAgent."""
        hotels = self.state.get_belief("available_hotels", [])
        if not hotels:
            return {"result": "No hotels found"}
        
        # Group by star rating
        by_stars = {}
        for h in hotels:
            stars = h.get('stars', 3)
            by_stars.setdefault(stars, []).append(h)
        
        # Sort each tier by distance (closer is better)
        for stars in by_stars:
            by_stars[stars].sort(key=lambda h: h.get('distance_to_business_center_km', 10))
        
        # Build diverse selection: prioritize quality (5★ → 4★ → 3★)
        selected = []
        for stars in [5, 4, 3, 2, 1]:
            if stars in by_stars and by_stars[stars]:
                selected.append(by_stars[stars][0])
        
        # Fill up to 5 options
        for stars in sorted(by_stars.keys(), reverse=True):
            for h in by_stars[stars]:
                if h not in selected and len(selected) < 5:
                    selected.append(h)
        
        return {"top_3_hotels": [h['hotel_id'] for h in selected[:5]],
                "reasoning": "Diverse selection across quality tiers for budget optimization."}
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        return {
            "search_hotels": AgentAction(
                name="search_hotels",
                description="Search for hotels in a city with optional filters",
                parameters={"city": "str", "max_price_per_night": "int (optional)",
                           "min_stars": "int (optional)", "max_distance_km": "float (optional)"},
                function=self._tool_search_hotels
            ),
            "get_hotel_details": AgentAction(
                name="get_hotel_details",
                description="Get detailed information about a specific hotel",
                parameters={"hotel_id": "str"},
                function=self._tool_get_hotel_details
            ),
            "compare_hotels": AgentAction(
                name="compare_hotels",
                description="Compare multiple hotels on specific criteria",
                parameters={"hotel_ids": "list", "criteria": "str - 'price', 'location', 'quality', or 'overall'"},
                function=self._tool_compare_hotels
            ),
            "check_amenities": AgentAction(
                name="check_amenities",
                description="Check if hotels have specific amenities",
                parameters={"hotel_ids": "list", "required_amenities": "list"},
                function=self._tool_check_amenities
            ),
            "analyze_area_options": AgentAction(
                name="analyze_area_options",
                description="Analyze hotel options in different business areas",
                parameters={"city": "str"},
                function=self._tool_analyze_area_options
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Hotel Booking Specialist for business travel.
PRIORITIES: Proximity to business center (<2km ideal), 3+ stars, WiFi, balance quality and price.
REASONING: Consider location, quality, amenities, price, and convenience for each option."""
    
    def _tool_search_hotels(self, city: str, max_price_per_night: Optional[int] = None,
                            min_stars: Optional[int] = None, max_distance_km: Optional[float] = None) -> str:
        hotels = self.loader.search(city=city, max_price_per_night=max_price_per_night,
                                    min_stars=min_stars, max_distance_to_center_km=max_distance_km)
        if not hotels:
            return f"No hotels found in {city} matching criteria."
        
        self.state.add_belief("search_city", city)
        self.state.add_belief("available_hotels", hotels)
        self.state.add_belief("hotel_count", len(hotels))
        
        result = [f"Found {len(hotels)} hotels in {city}:"]
        for h in hotels[:10]:
            result.append(f"  - {h['hotel_id']}: {h['name']}, {h['stars']}★, "
                         f"${h['price_per_night_usd']}/night, {h['distance_to_business_center_km']:.1f}km")
        if len(hotels) > 10:
            result.append(f"  ... and {len(hotels) - 10} more")
        
        self.search_history.append({"city": city, "max_price": max_price_per_night, "results": len(hotels)})
        return "\n".join(result)
    
    def _tool_get_hotel_details(self, hotel_id: str) -> str:
        if isinstance(hotel_id, list):
            return "\n\n".join(self._tool_get_hotel_details(hid) for hid in hotel_id[:3])
        
        hotels = self.state.get_belief("available_hotels", [])
        for h in hotels:
            if h['hotel_id'] == hotel_id:
                return (f"Hotel {hotel_id}: {h['name']}, {h['stars']}★, ${h['price_per_night_usd']}/night, "
                       f"{h['distance_to_business_center_km']:.2f}km from center, {h['business_area']}")
        return f"Hotel {hotel_id} not found."
    
    def _tool_compare_hotels(self, hotel_ids: List[str], criteria: str = "overall") -> str:
        hotels = self.state.get_belief("available_hotels", [])
        hotel_dict = {h['hotel_id']: h for h in hotels}
        to_compare = [hotel_dict[hid] for hid in hotel_ids if hid in hotel_dict]
        
        if not to_compare:
            return "No valid hotels to compare."
        
        if criteria == "price":
            sorted_h = sorted(to_compare, key=lambda x: x['price_per_night_usd'])
            return "\n".join(f"{i+1}. {h['hotel_id']}: ${h['price_per_night_usd']}/night" for i, h in enumerate(sorted_h))
        elif criteria == "location":
            sorted_h = sorted(to_compare, key=lambda x: x['distance_to_business_center_km'])
            return "\n".join(f"{i+1}. {h['hotel_id']}: {h['distance_to_business_center_km']:.1f}km" for i, h in enumerate(sorted_h))
        elif criteria == "quality":
            sorted_h = sorted(to_compare, key=lambda x: -x['stars'])
            return "\n".join(f"{i+1}. {h['hotel_id']}: {h['stars']}★" for i, h in enumerate(sorted_h))
        else:  # overall
            def score(h):
                loc = h['distance_to_business_center_km'] / 5
                price = h['price_per_night_usd'] / 500
                quality = (5 - h['stars']) / 5
                return 0.4 * loc + 0.3 * price + 0.3 * quality
            sorted_h = sorted(to_compare, key=score)
            return "\n".join(f"{i+1}. {h['hotel_id']}: {h['name']}, {h['stars']}★, ${h['price_per_night_usd']}/night"
                           for i, h in enumerate(sorted_h))
    
    def _tool_check_amenities(self, hotel_ids: List[str], required_amenities: List[str]) -> str:
        hotels = self.state.get_belief("available_hotels", [])
        hotel_dict = {h['hotel_id']: h for h in hotels}
        
        results = []
        for hid in hotel_ids:
            if hid in hotel_dict:
                h = hotel_dict[hid]
                hotel_amenities = set(h.get('amenities', []))
                required = set(required_amenities)
                missing = required - hotel_amenities
                status = "✓ All" if not missing else f"✗ Missing: {', '.join(missing)}"
                results.append(f"  {hid}: {status}")
        return f"Amenity Check:\n" + "\n".join(results) if results else "No valid hotels."
    
    def _tool_analyze_area_options(self, city: str) -> str:
        hotels = self.loader.search(city=city)
        if not hotels:
            return f"No hotels in {city}"
        
        areas = {}
        for h in hotels:
            area = h.get('business_area', 'Unknown')
            areas.setdefault(area, []).append(h)
        
        result = [f"Hotel Distribution in {city}:"]
        for area, area_hotels in areas.items():
            prices = [h['price_per_night_usd'] for h in area_hotels]
            result.append(f"  {area}: {len(area_hotels)} hotels, ${min(prices)}-${max(prices)}")
        return "\n".join(result)
    
    def search_hotels(self, query: HotelQuery) -> HotelSearchResult:
        """Main entry point for hotel search using ReAct reasoning."""
        self.reset_state()
        
        constraints = []
        if query.max_price_per_night: constraints.append(f"Max ${query.max_price_per_night}/night")
        if query.min_stars: constraints.append(f"Min {query.min_stars}★")
        if query.max_distance_to_center_km: constraints.append(f"Within {query.max_distance_to_center_km}km")
        if query.required_amenities: constraints.append(f"Must have: {', '.join(query.required_amenities)}")
        
        goal = f"""Find top 3 best hotels for business in {query.city}
Constraints: {'; '.join(constraints) if constraints else 'None'}

1. Search for hotels 2. Analyze options 3. Compare top candidates 4. Select 3 best with reasoning
Return JSON: {{"top_3_hotels": [hotel IDs], "reasoning": "explanation"}}"""

        result = self.run(goal)
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    top_ids = parsed.get("top_3_hotels", [])
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    top_ids = final.get("top_3_hotels", [])
                    llm_reasoning = final.get("reasoning", str(final))
                else:
                    top_ids, llm_reasoning = [], str(final)
                
                available = self.state.get_belief("available_hotels", [])
                hotel_dict = {h['hotel_id']: h for h in available}
                top_hotels = [Hotel(**hotel_dict[hid]) for hid in top_ids[:5] if hid in hotel_dict]
                
                if not top_hotels and available:
                    top_hotels = [Hotel(**h) for h in available[:3]]
                    llm_reasoning = "Fallback selection."
            except Exception as e:
                available = self.state.get_belief("available_hotels", [])
                top_hotels = [Hotel(**h) for h in available[:3]] if available else []
                llm_reasoning = f"Fallback: {e}"
        else:
            hotels = self.loader.search(city=query.city, max_price_per_night=query.max_price_per_night,
                                        min_stars=query.min_stars, max_distance_to_center_km=query.max_distance_to_center_km,
                                        required_amenities=query.required_amenities)
            top_hotels = [Hotel(**h) for h in hotels[:3]] if hotels else []
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}. Used fallback."
        
        self.log_message("orchestrator", f"Found {len(top_hotels)} recommended hotels", "result")
        
        reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
        return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
