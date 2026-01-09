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
        
        goal = f"""Find best hotels for business in {query.city}
Constraints: {'; '.join(constraints) if constraints else 'None'}

1. Search for hotels 2. Analyze options across price tiers 3. Compare top candidates
Return JSON: {{"top_hotels": [hotel IDs from various star ratings], "reasoning": "explanation"}}"""

        result = self.run(goal)
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    llm_reasoning = final.get("reasoning", str(final))
                else:
                    llm_reasoning = str(final)
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
        
        # IMPORTANT: Return DIVERSE options by star rating
        # PolicyAgent needs variety to make informed budget decisions
        available = self.state.get_belief("available_hotels", [])
        
        if not available:
            # Fallback: search directly
            hotels = self.loader.search(city=query.city, max_price_per_night=query.max_price_per_night,
                                        min_stars=query.min_stars, max_distance_to_center_km=query.max_distance_to_center_km,
                                        required_amenities=query.required_amenities)
            available = hotels if hotels else []
        
        # Build diverse set: include options from each star level
        diverse_hotels = []
        seen_ids = set()
        
        # Group by stars
        by_stars = {5: [], 4: [], 3: [], 2: [], 1: []}
        for h in available:
            stars = h.get('stars', 3)
            if stars in by_stars:
                by_stars[stars].append(h)
        
        # Score hotels by business value (proximity + price)
        def business_score(h):
            distance = h.get('distance_to_business_center_km', 5)
            price = h.get('price_per_night_usd', 200)
            return (distance, price)  # Closer and cheaper first (within same star level)
        
        # Take best from each star level (prioritize higher stars)
        for stars in [5, 4, 3, 2, 1]:
            star_hotels = sorted(by_stars[stars], key=business_score)
            for h in star_hotels[:2]:  # Top 2 from each star level
                if h['hotel_id'] not in seen_ids:
                    seen_ids.add(h['hotel_id'])
                    diverse_hotels.append(h)
        
        # Ensure we have at least 5 options
        if len(diverse_hotels) < 5:
            for h in sorted(available, key=lambda x: -x.get('stars', 3)):
                if h['hotel_id'] not in seen_ids and len(diverse_hotels) < 8:
                    seen_ids.add(h['hotel_id'])
                    diverse_hotels.append(h)
        
        top_hotels = [Hotel(**h) for h in diverse_hotels[:10]]  # Up to 10 diverse options
        
        self.log_message("orchestrator", f"Found {len(top_hotels)} diverse hotel options", "result")
        
        reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
        return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
    
    def refine_proposal(self, feedback: Dict[str, Any], previous_hotels: List[Dict]) -> HotelSearchResult:
        """
        CNP NEGOTIATION: Refine hotel proposal based on PolicyAgent feedback.
        Uses fast deterministic filtering (no LLM) for quick negotiation rounds.
        
        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_insufficient" | "location_issue"
                - max_price_per_night: suggested max price (if budget issue)
                - min_stars: suggested minimum stars (if quality issue)
                - max_distance_km: suggested max distance (if location issue)
                - reasoning: PolicyAgent's explanation
            previous_hotels: Hotels from previous proposal to avoid repeating
        
        Returns:
            Refined HotelSearchResult with new proposal
        """
        issue = feedback.get("issue", "general")
        city = feedback.get("city", "")
        
        if self.verbose:
            print(f"    [HotelAgent] Refining for: {issue}")
        
        # Get all available hotels from previous search (stored in state)
        available = self.state.get_belief("available_hotels", [])
        
        # If no cached hotels, do a quick search using the loader
        if not available:
            if self.verbose:
                print(f"    [HotelAgent] No cached hotels, searching...")
            all_hotels = self.loader.search(city=city)
            available = all_hotels if all_hotels else []
            self.state.add_belief("available_hotels", available)
        
        previous_ids = {h.get('hotel_id') for h in previous_hotels}
        
        # Filter based on feedback constraints (FAST - no LLM needed)
        filtered = []
        for h in available:
            if h.get('hotel_id') in previous_ids:
                continue  # Skip previously rejected
            
            if issue == "budget_exceeded":
                max_price = feedback.get("max_price_per_night", 200)
                if h.get('price_per_night_usd', 999) <= max_price:
                    filtered.append(h)
            elif issue == "quality_insufficient":
                min_stars = feedback.get("min_stars", 4)
                if h.get('stars', 0) >= min_stars:
                    filtered.append(h)
            elif issue == "location_issue":
                max_distance = feedback.get("max_distance_km", 2.0)
                if h.get('distance_to_business_center_km', 99) <= max_distance:
                    filtered.append(h)
            else:
                filtered.append(h)
        
        # Sort by relevance to feedback
        if issue == "budget_exceeded":
            filtered.sort(key=lambda x: x.get('price_per_night_usd', 999))
        elif issue == "quality_insufficient":
            filtered.sort(key=lambda x: (-x.get('stars', 0), x.get('price_per_night_usd', 999)))
        elif issue == "location_issue":
            filtered.sort(key=lambda x: x.get('distance_to_business_center_km', 99))
        else:
            filtered.sort(key=lambda x: (-x.get('stars', 0), x.get('price_per_night_usd', 999)))
        
        refined_hotels = [Hotel(**h) for h in filtered[:8]]
        
        reasoning = f"Refined {len(refined_hotels)} hotels addressing: {issue}"
        if self.verbose:
            print(f"    [HotelAgent] Refined: {len(refined_hotels)} options")
        
        self.log_message("orchestrator", f"Refined proposal: {len(refined_hotels)} hotels (addressing {issue})", "negotiation")
        
        return HotelSearchResult(
            query=HotelQuery(city=city),
            hotels=refined_hotels,
            reasoning=reasoning
        )
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
