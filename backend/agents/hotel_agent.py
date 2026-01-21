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
        """Extract SINGLE hotel when agent fails to call finish()."""
        hotels = self.state.get_belief("available_hotels", [])
        if not hotels:
            return {"result": "No hotels found"}

        # NO BIAS - just use the first hotel from search results
        # The LLM should have made a decision, this is just emergency fallback
        first_hotel = hotels[0]

        return {"selected_hotels": [first_hotel['hotel_id']],
                "reasoning": "Fallback: Agent did not make a selection, returning first available option."}
    
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
        return """You are a Hotel Booking Specialist. Find the single best hotel for this trip.

AVAILABLE TOOLS:
1. search_hotels(city) - Search for available hotels
2. analyze_options() - Get overview of options by quality and price
3. compare_hotels(hotel_ids=[...]) - Compare specific hotels
4. get_hotel_details(hotel_id) - Get detailed info about one hotel
5. finish(result={...}) - Submit your final selection

YOUR TASK:
Find ONE hotel that offers good value for the trip.
You don't know the budget - use your judgment to balance quality and cost.

Consider all available options:
- Hotels range from 2 to 5 stars - all quality levels are valid choices
- Consider stars, amenities, location, and price together
- Cheaper is not always better, more expensive is not always better
- Make your own decision based on overall value

Use your tools effectively:
- search_hotels to see what's available
- analyze_options to understand the range
- compare_hotels to evaluate specific options
- finish when you've made your decision

Return format: {"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this is the best choice"}"""
    
    def _tool_search_hotels(self, city: str, max_price_per_night: Optional[int] = None,
                            min_stars: Optional[int] = None, max_distance_km: Optional[int] = None,
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
                result.append(f"\n{stars}-star HOTELS ({len(by_stars[stars])} options):")
                for h in sorted(by_stars[stars], key=lambda x: x['price_per_night_usd'])[:3]:
                    amenities = ", ".join(h.get('amenities', [])[:4]) or "No amenities listed"
                    # Show meeting distance if available, otherwise business center distance
                    distance = h.get('distance_to_meeting_km', h.get('distance_to_business_center_km', 0))
                    distance_label = "from meeting" if 'distance_to_meeting_km' in h else "from center"
                    result.append(f"  - {h['hotel_id']}: {h['name']}, ${h['price_per_night_usd']}/night, "
                                 f"{distance:.1f}km {distance_label} | Amenities: {amenities}")
        
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
            sorted_h = sorted(to_compare, key=lambda x: x.get('distance_to_meeting_km', x.get('distance_to_business_center_km', 99)))
            return "\n".join(f"{i+1}. {h['hotel_id']}: {h.get('distance_to_meeting_km', h.get('distance_to_business_center_km', 0)):.1f}km" for i, h in enumerate(sorted_h))
        elif criteria == "quality":
            sorted_h = sorted(to_compare, key=lambda x: -x['stars'])
            return "\n".join(f"{i+1}. {h['hotel_id']}: {h['stars']}★" for i, h in enumerate(sorted_h))
        else:  # overall - show all details, sorted by ID (no bias)
            sorted_h = sorted(to_compare, key=lambda h: h['hotel_id'])
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
                meeting_context = f"\nMeeting location provided - consider proximity."

        # Include required amenities
        amenities_context = ""
        if query.required_amenities:
            amenities_context = f"\nREQUIRED amenities: {', '.join(query.required_amenities)} - hotel must have these."
        
        # Goal prompt for this specific search
        goal = f"""Find the best hotel in {query.city}.{meeting_context}{amenities_context}

Explore available options and select ONE hotel that offers good overall value.

Consider different quality levels and price points:
- Compare options from different star ratings (2-star through 5-star)
- Consider amenities, location, and price together
- Make your own decision on the best balance of quality and cost

Use your tools to explore before deciding:
- search_hotels() to see what's available
- analyze_options() to understand the quality and price distribution
- compare_hotels() to evaluate specific options
- finish() when you've made your decision

Return: {{"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this hotel offers good value"}}"""

        result = self.run(goal)
        
        # Parse LLM's selection
        selected_ids = []
        llm_reasoning = ""
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    selected_ids = parsed.get("selected_hotels", parsed.get("top_hotels", parsed.get("top_options", [])))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    selected_ids = final.get("selected_hotels", final.get("top_hotels", final.get("top_options", [])))
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
            # Also build name-to-id lookup for fallback matching
            name_to_id = {h.get('name', '').lower(): h['hotel_id'] for h in all_hotels}
            
            # Handle multiple formats:
            # - list of strings (IDs): ["HT0031", "HT0026"]
            # - list of dicts with hotel_id: [{"hotel_id": "HT0031", ...}]
            # - list of dicts with hotel_name: [{"hotel_name": "Holiday Inn...", ...}]
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    if item.startswith('HT'):
                        normalized_ids.append(item)  # Already an ID
                    else:
                        # Might be a hotel name - try to find it
                        matched_id = name_to_id.get(item.lower())
                        if matched_id:
                            normalized_ids.append(matched_id)
                elif isinstance(item, dict):
                    if 'hotel_id' in item:
                        normalized_ids.append(item['hotel_id'])
                    elif 'hotel_name' in item:
                        # Try to match by name
                        name = item['hotel_name'].lower()
                        matched_id = name_to_id.get(name)
                        if matched_id:
                            normalized_ids.append(matched_id)
            
            selected_hotels = [hotel_dict[hid] for hid in normalized_ids if hid in hotel_dict]
            
            # If LLM selected valid hotels, use them (take only the first one)
            if selected_hotels:
                top_hotels = [Hotel(**h) for h in selected_hotels[:1]]  # ONLY 1 hotel
                self.log_message("orchestrator", f"LLM selected best hotel: {top_hotels[0].hotel_id}", "result")
                reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
                return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
        
        # FALLBACK: If LLM didn't select valid hotels, use first option
        print(f"    [HotelAgent] LLM selection failed, using fallback (first available)")

        # NO BIAS - just use first available hotel
        fallback_hotels = all_hotels[:1]  # ONLY 1 hotel
        top_hotels = [Hotel(**h) for h in fallback_hotels]

        self.log_message("orchestrator", f"Fallback proposal: 1 hotel (first available)", "result")
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
        # NO BIAS - show hotels in neutral order (by price for consistency)
        # The LLM will decide which one to pick based on the feedback
        sorted_available = sorted(available, key=lambda x: x.get('price_per_night_usd', 0))

        hotel_summary = []
        for h in sorted_available[:20]:  # Show 20 options
            amenities = ", ".join(h.get('amenities', [])[:3])
            hotel_summary.append(
                f"{h['hotel_id']}: {h['name']} {h['stars']}★ "
                f"${h['price_per_night_usd']}/night {h['distance_to_business_center_km']:.1f}km "
                f"[{amenities}]"
            )
        
        # LLM prompt for agentic reasoning - SINGLE OPTION ONLY
        prompt = f"""You are a Hotel Booking Specialist agent. The PolicyAgent has rejected your proposal.

FEEDBACK FROM POLICY AGENT:
- Issue: {issue}
- Reasoning: {feedback.get('reasoning', 'No details provided')}
- Target Price Range: ${feedback.get('target_price_min', 50)}-${feedback.get('target_price_max', 500)}/night

AVAILABLE HOTELS IN {city}:
{chr(10).join(hotel_summary[:10])}

YOUR TASK:
Analyze the feedback and target price range, then select EXACTLY ONE hotel that best addresses the PolicyAgent's concerns.
Consider the trade-offs between price, star rating, location, and amenities.

Return JSON with EXACTLY ONE hotel ID:
{{"selected_hotel": "HT0XXX", "reasoning": "Brief explanation of why this hotel best addresses the feedback"}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            # Get SINGLE hotel - check both singular and plural keys
            selected_id = result.get("selected_hotel") or (result.get("selected_hotels", []) or [None])[0]
            reasoning = result.get("reasoning", "Selected based on feedback analysis")
            
            # Get the selected hotel - ONLY ONE
            selected = [h for h in available if h['hotel_id'] == selected_id][:1]
            
            # Fallback: if LLM didn't select valid hotels, use first option (no bias)
            if not selected:
                if self.verbose:
                    print(f"    [HotelAgent] LLM selection empty, using fallback (first available)")
                selected = available[:1]
                reasoning = f"Fallback: LLM did not select, using first available option."
            
            refined_hotels = [Hotel(**h) for h in selected[:1]]  # ONLY 1 hotel
            
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

            # Fallback: first available (no bias)
            selected = available[:1]

            refined_hotels = [Hotel(**h) for h in selected]
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=refined_hotels,
                reasoning=f"Fallback: LLM error, using first available option."
            )
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
