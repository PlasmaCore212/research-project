# backend/agents/hotel_agent.py
"""Hotel Agent with ReAct Pattern for business travel hotel search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import HotelQuery, HotelSearchResult, Hotel
from data.loaders import HotelDataLoader
from typing import List, Dict, Any, Optional
import json


class HotelAgent(BaseReActAgent):
    """Agentic Hotel Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "qwen2.5:14b", verbose: bool = True):
        super().__init__(
            agent_name="HotelAgent", agent_role="Hotel Booking Specialist",
            model_name=model_name, max_iterations=10, verbose=verbose
        )
        self.loader = HotelDataLoader()
        self.tools = self._register_tools()
    
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
                description="Compare multiple hotels on specific criteria. REQUIRES: hotel_ids as a list of strings (e.g. ['HT0001', 'HT0002']).",
                parameters={"hotel_ids": "list[str] REQUIRED - list of hotel IDs to compare",
                           "criteria": "str (optional) - 'price', 'location', 'quality', or 'overall'"},
                function=self._tool_compare_hotels
            ),

            "analyze_area_options": AgentAction(
                name="analyze_area_options",
                description="Analyze hotel options in different business areas",
                parameters={"city": "str"},
                function=self._tool_analyze_area_options
            ),
            "analyze_options": AgentAction(
                name="analyze_options",
                description="Analyze available hotels by price tier and quality. Use after search_hotels.",
                parameters={},
                function=self._tool_analyze_options
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Hotel Booking Specialist finding business travel accommodations.

AVAILABLE TOOLS (these are the ONLY tools that exist):
• search_hotels(city="SF") - Search hotels with amenities included. Call FIRST.
• compare_hotels(hotel_ids=["HT001","HT002"]) - Compare hotels by ID
• get_hotel_details(hotel_id="HT001") - Get ONE hotel's details
• analyze_options() - Summarize by star rating and price
• finish(result) - Return final selection

‼️ FORBIDDEN - These tools DO NOT EXIST:
✗ filter_hotels - DOES NOT EXIST
✗ sort_hotels - DOES NOT EXIST  
✗ inspect_hotel - DOES NOT EXIST
✗ select_hotels - DOES NOT EXIST
✗ check_amenities - DOES NOT EXIST (amenities are shown in search_hotels results)
If you need to filter, use compare_hotels() with specific IDs instead!

GOAL: Select 6 DIVERSE hotels (1 from each: 5★, 4★, two 3★, two 2★).

CORRECT WORKFLOW:
1. search_hotels(city="...") → See all options with amenities, grouped by stars
2. Pick hotel IDs from each star tier in the search results
3. compare_hotels(hotel_ids=["HT001","HT002","HT003"]) → Compare your picks
4. finish(result) → Return your 6 diverse selections as JSON

Return: {"selected_hotels": ["HT001", "HT002", ...], "reasoning": "..."}"""
    
    def _tool_search_hotels(self, city: str, max_price_per_night: Optional[int] = None,
                            min_stars: Optional[int] = None, max_distance_km: Optional[float] = None,
                            **kwargs) -> str:
        """Search hotels. Extra kwargs are ignored to handle LLM parameter variations."""
        # ALWAYS search without star filter to show diverse options
        hotels = self.loader.search(city=city, max_price_per_night=max_price_per_night,
                                    max_distance_to_center_km=max_distance_km)
        if not hotels:
            return f"No hotels found in {city} matching criteria."
        
        self.state.add_belief("search_city", city)
        self.state.add_belief("available_hotels", hotels)
        self.state.add_belief("hotel_count", len(hotels))
        
        # Group by star rating for clear display
        by_stars = {5: [], 4: [], 3: [], 2: [], 1: []}
        for h in hotels:
            stars = h.get('stars', 3)
            if stars in by_stars:
                by_stars[stars].append(h)
        
        result = [f"Found {len(hotels)} hotels in {city}:"]
        
        # Show hotels grouped by star rating
        for stars in [5, 4, 3, 2]:
            if by_stars[stars]:
                result.append(f"\n📍 {stars}★ HOTELS ({len(by_stars[stars])} options):")
                for h in sorted(by_stars[stars], key=lambda x: x['price_per_night_usd'])[:3]:
                    amenities = ", ".join(h.get('amenities', [])[:4]) or "No amenities listed"
                    result.append(f"  - {h['hotel_id']}: {h['name']}, ${h['price_per_night_usd']}/night, "
                                 f"{h['distance_to_business_center_km']:.1f}km | Amenities: {amenities}")
        
        return "\n".join(result)
    
    def _tool_get_hotel_details(self, hotel_id: str, **kwargs) -> str:
        """Get hotel details. Extra kwargs are ignored."""
        if isinstance(hotel_id, list):
            return "\n\n".join(self._tool_get_hotel_details(hid) for hid in hotel_id[:3])
        
        hotels = self.state.get_belief("available_hotels", [])
        for h in hotels:
            if h['hotel_id'] == hotel_id:
                return (f"Hotel {hotel_id}: {h['name']}, {h['stars']}★, ${h['price_per_night_usd']}/night, "
                       f"{h['distance_to_business_center_km']:.2f}km from center, {h['business_area']}")
        return f"Hotel {hotel_id} not found."
    
    def _tool_compare_hotels(self, hotel_ids: List[str], criteria: str = "overall", **kwargs) -> str:
        """Compare hotels. Extra kwargs are ignored."""
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
    
    
    def _tool_analyze_area_options(self, city: str = None, **kwargs) -> str:
        """Analyze hotel area options. Extra kwargs are ignored."""
        # Use city from state if not provided
        if not city:
            city = self.state.get_belief("search_city", "NYC")
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
    
    def _tool_analyze_options(self, **kwargs) -> str:
        """Analyze available hotels by price tier and quality."""
        hotels = self.state.get_belief("available_hotels", [])
        if not hotels:
            return "No hotels to analyze. Run search_hotels first."
        
        # Group by price tier
        budget = [h for h in hotels if h.get('price_per_night_usd', 0) < 200]
        midrange = [h for h in hotels if 200 <= h.get('price_per_night_usd', 0) < 400]
        premium = [h for h in hotels if h.get('price_per_night_usd', 0) >= 400]
        
        # Group by star rating
        by_stars = {}
        for h in hotels:
            stars = h.get('stars', 3)
            by_stars.setdefault(stars, []).append(h)
        
        result = [f"Hotel Analysis ({len(hotels)} total):"]
        result.append(f"  Budget (<$200): {len(budget)} hotels")
        result.append(f"  Mid-Range ($200-$400): {len(midrange)} hotels")
        result.append(f"  Premium (>$400): {len(premium)} hotels")
        result.append("")
        result.append("By Star Rating:")
        for stars in sorted(by_stars.keys(), reverse=True):
            prices = [h.get('price_per_night_usd', 0) for h in by_stars[stars]]
            result.append(f"  {stars}★: {len(by_stars[stars])} hotels, ${min(prices):.0f}-${max(prices):.0f}/night")
        
        return "\n".join(result)
    
    def search_hotels(self, query: HotelQuery) -> HotelSearchResult:
        """Main entry point for hotel search using ReAct reasoning."""
        self.reset_state()
        
        constraints = []
        if query.max_price_per_night: constraints.append(f"Max ${query.max_price_per_night}/night")
        if query.min_stars: constraints.append(f"Min {query.min_stars}★")
        if query.max_distance_to_center_km: constraints.append(f"Within {query.max_distance_to_center_km}km")
        if query.required_amenities: constraints.append(f"Must have: {', '.join(query.required_amenities)}")
        
        # Include meeting location in goal
        meeting_context = ""
        if query.meeting_location:
            lat = query.meeting_location.get("lat")
            lon = query.meeting_location.get("lon")
            if lat and lon:
                meeting_context = f"\n⭐ Meeting location provided - consider proximity."
        
        # AGENTIC PROMPT: Let the LLM decide what's diverse
        goal = f"""Find hotels for business trip in {query.city}

YOUR TASK: Select exactly 6 DIVERSE hotel options to send to PolicyAgent.{meeting_context}

DIVERSITY REQUIREMENTS (MANDATORY):
- At least 1 luxury hotel (5★)
- At least 1 upscale hotel (4★)
- At least 2 mid-range hotels (3★)
- At least 2 budget hotels (2★)

WHY DIVERSITY MATTERS:
PolicyAgent needs options across ALL star ratings to make budget trade-offs.
YOU don't know the budget - so include ALL tiers!

WORKFLOW:
1. search_hotels(city="{query.city}")
2. Review the results - note 5★, 4★, 3★, and 2★ options
3. Use compare_hotels() to compare options across different tiers
4. finish() with your 6 diverse selections

Return JSON: {{"selected_hotels": ["HT001", "HT002", ...], "reasoning": "explanation of diversity"}}"""

        result = self.run(goal)
        
        # Parse LLM's selection
        selected_ids = []
        llm_reasoning = ""
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    selected_ids = parsed.get("selected_hotels", parsed.get("top_hotels", []))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    selected_ids = final.get("selected_hotels", final.get("top_hotels", []))
                    llm_reasoning = final.get("reasoning", str(final))
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
        
        # Get ALL hotels (no filters) for diversity
        all_hotels = self.loader.search(city=query.city)
        
        if not all_hotels:
            all_hotels = self.state.get_belief("available_hotels", [])
        
        # Calculate distance to meeting
        if query.meeting_location:
            meeting_lat = query.meeting_location.get("lat")
            meeting_lon = query.meeting_location.get("lon")
            if meeting_lat and meeting_lon:
                from data.data_generator import haversine_distance
                for h in all_hotels:
                    hotel_lat = h.get("latitude")
                    hotel_lon = h.get("longitude")
                    if hotel_lat and hotel_lon:
                        h["distance_to_meeting_km"] = haversine_distance(
                            hotel_lat, hotel_lon, meeting_lat, meeting_lon
                        )
                    else:
                        h["distance_to_meeting_km"] = h.get("distance_to_business_center_km", 10)
        
        # TRUST THE LLM'S SELECTION
        if selected_ids:
            hotel_dict = {h['hotel_id']: h for h in all_hotels}
            selected_hotels = [hotel_dict[hid] for hid in selected_ids if hid in hotel_dict]
            
            # If LLM selected valid hotels, use them
            if selected_hotels:
                top_hotels = [Hotel(**h) for h in selected_hotels[:6]]
                self.log_message("orchestrator", f"LLM selected {len(top_hotels)} hotels: {[h.hotel_id for h in top_hotels]}", "result")
                reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
                return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
        
        # FALLBACK: If LLM didn't select valid hotels, use simple diversity
        print(f"    [HotelAgent] LLM selection failed, using fallback diversity")
        fallback_hotels = sorted(all_hotels, key=lambda x: x.get('price_per_night_usd', 0))[:6]
        top_hotels = [Hotel(**h) for h in fallback_hotels]
        
        self.log_message("orchestrator", f"Fallback proposal: {len(top_hotels)} hotels", "result")
        reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
        return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
    
    def refine_proposal(self, feedback: Dict[str, Any], previous_hotels: List[Dict]) -> HotelSearchResult:
        """
        CNP NEGOTIATION: Refine hotel proposal based on PolicyAgent feedback.
        
        TRULY AGENTIC: Uses LLM reasoning to decide how to respond to feedback,
        considering trade-offs and making autonomous decisions about which
        hotels best address the PolicyAgent's concerns.
        
        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "location_issue"
                - target_price_min/max: suggested price range (for upgrade)
                - max_price_per_night: suggested max price (if budget issue)
                - reasoning: PolicyAgent's explanation
            previous_hotels: Hotels from previous proposal
        
        Returns:
            Refined HotelSearchResult with new proposal and reasoning
        """
        issue = feedback.get("issue", "general")
        city = feedback.get("city", "")
        
        # Get PolicyAgent's target price constraints
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)
        should_re_search = feedback.get("re_search", False)
        
        if self.verbose:
            print(f"    [HotelAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [HotelAgent] Target price range: ${target_min}-${target_max}/night")
        
        # RE-SEARCH with PolicyAgent's constraints
        # This is the key enhancement - agents actually use their tools again
        if should_re_search:
            if self.verbose:
                print(f"    [HotelAgent] Re-searching ALL hotels with price filter...")
            all_hotels = self.loader.search(city=city)
            
            # Filter to target price range
            if target_min > 0 or target_max < 9999:
                filtered = [h for h in all_hotels 
                           if target_min <= h.get('price_per_night_usd', 0) <= target_max]
                if filtered:
                    available = filtered
                    if self.verbose:
                        print(f"    [HotelAgent] Found {len(available)} hotels in ${target_min}-${target_max}/night range")
                else:
                    # No hotels in exact range, get closest
                    available = all_hotels
                    if self.verbose:
                        print(f"    [HotelAgent] No hotels in exact range, using all {len(available)} hotels")
            else:
                available = all_hotels
        else:
            # Fallback: use cached
            all_hotels = self.loader.search(city=city)
            available = all_hotels if all_hotels else []
        
        self.state.add_belief("available_hotels", available)
        
        if not available:
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=[],
                reasoning="No hotels available in this city."
            )
        
        # Build context for LLM reasoning
        # CRITICAL: Sort by relevance to the issue before showing to LLM
        if issue == "budget_exceeded":
            # For budget issues, show CHEAPEST hotels first so LLM can select them
            sorted_available = sorted(available, key=lambda x: x.get('price_per_night_usd', 999))
        elif issue == "quality_upgrade" and target_min > 0:
            # For quality upgrade with target, sort by distance from target midpoint
            target_mid = (target_min + target_max) / 2
            sorted_available = sorted(available, key=lambda x: abs(x.get('price_per_night_usd', 0) - target_mid))
        else:
            # For quality issues, show premium hotels first
            sorted_available = sorted(available, key=lambda x: (-x.get('stars', 0), x.get('price_per_night_usd', 0)))
        
        hotel_summary = []
        for h in sorted_available[:20]:  # Show 20 options sorted by relevance
            amenities = ", ".join(h.get('amenities', [])[:3])
            hotel_summary.append(
                f"{h['hotel_id']}: {h['name']} {h['stars']}★ "
                f"${h['price_per_night_usd']}/night {h['distance_to_business_center_km']:.1f}km "
                f"[{amenities}]"
            )
        
        # LLM prompt for agentic reasoning
        prompt = f"""You are a Hotel Booking Specialist agent. The PolicyAgent has rejected your proposal.

FEEDBACK FROM POLICY AGENT:
- Issue: {issue}
- Reasoning: {feedback.get('reasoning', 'No details provided')}
- Constraint: {feedback.get('max_price_per_night', feedback.get('min_stars', 'Not specified'))}

AVAILABLE HOTELS IN {city}:
{chr(10).join(hotel_summary)}

YOUR TASK:
Analyze the feedback and select the best hotels that address the PolicyAgent's concerns.
Consider trade-offs: if budget is tight, prioritize price over stars; if quality is requested, prioritize stars and location.

For budget issues: Find the cheapest options that still meet basic business needs (WiFi, close to center).
For quality issues: Find premium options (5★, excellent location, full amenities).

Return JSON with your reasoning:
{{"selected_hotels": ["hotel_id1", "hotel_id2", ...], "reasoning": "Your explanation of why these hotels best address the feedback", "addresses_constraint": true/false}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            selected_ids = set(result.get("selected_hotels", []))
            reasoning = result.get("reasoning", "Selected based on feedback analysis")
            
            # Get the selected hotels
            selected = [h for h in available if h['hotel_id'] in selected_ids]
            
            # Fallback: if LLM didn't select valid hotels, use heuristic
            if not selected:
                if self.verbose:
                    print(f"    [HotelAgent] LLM selection empty, using fallback")
                if issue == "budget_exceeded":
                    selected = sorted(available, key=lambda x: x.get('price_per_night_usd', 999))[:8]
                    reasoning = "Fallback: Selecting cheapest available hotels."
                elif issue == "quality_upgrade":
                    # For quality upgrade, select PREMIUM options (4-5★)
                    premium = [h for h in available if h.get('stars', 3) >= 4]
                    if premium:
                        selected = sorted(premium, key=lambda x: (-x.get('stars', 0), x.get('distance_to_business_center_km', 99)))[:8]
                        reasoning = "Quality upgrade: Selecting 4-5★ premium hotels."
                    else:
                        # No 4-5★, select best 3★ options
                        selected = sorted(available, key=lambda x: (-x.get('stars', 0), x.get('distance_to_business_center_km', 99)))[:8]
                        reasoning = "Quality upgrade: Selecting best available hotels."
                else:
                    selected = sorted(available, key=lambda x: (-x.get('stars', 0), x.get('distance_to_business_center_km', 99)))[:8]
                    reasoning = "Fallback: Selecting premium hotels by stars and location."
            else:
                # IMPORTANT: For budget issues, ALWAYS include the absolute cheapest option
                if issue == "budget_exceeded":
                    cheapest = min(available, key=lambda x: x.get('price_per_night_usd', 999))
                    if cheapest not in selected:
                        selected.insert(0, cheapest)
                # For quality upgrade, include premium options
                elif issue == "quality_upgrade":
                    premium = [h for h in available if h.get('stars', 3) >= 4]
                    for h in sorted(premium, key=lambda x: -x.get('stars', 0))[:3]:
                        if h not in selected:
                            selected.insert(0, h)
            
            refined_hotels = [Hotel(**h) for h in selected[:8]]
            
            if self.verbose:
                if refined_hotels:
                    prices = [h.price_per_night_usd for h in refined_hotels]
                    stars = [h.stars for h in refined_hotels]
                    print(f"    [HotelAgent] Selected {len(refined_hotels)} hotels ({min(stars)}-{max(stars)}★, ${min(prices)}-${max(prices)}/night)")
                    print(f"    [HotelAgent] Reasoning: {reasoning[:80]}...")
            
            self.log_message("policy_agent", f"Refined: {len(refined_hotels)} hotels - {reasoning[:100]}", "negotiation")
            
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=refined_hotels,
                reasoning=reasoning
            )
            
        except Exception as e:
            if self.verbose:
                print(f"    [HotelAgent] LLM error: {e}, using fallback")
            
            # Fallback to heuristic selection
            if issue == "budget_exceeded":
                selected = sorted(available, key=lambda x: x.get('price_per_night_usd', 999))[:8]
            else:
                selected = sorted(available, key=lambda x: (-x.get('stars', 0), x.get('distance_to_business_center_km', 99)))[:8]
            
            refined_hotels = [Hotel(**h) for h in selected]
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=refined_hotels,
                reasoning=f"Fallback selection addressing {issue}"
            )
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
