# backend/agents/hotel_agent.py
"""Hotel Agent with ReAct Pattern for business travel hotel search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import HotelQuery, HotelSearchResult, Hotel
from data.loaders import HotelDataLoader
from typing import List, Dict, Any, Optional
import json


class HotelAgent(BaseReActAgent):
    """Agentic Hotel Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "mistral-small", verbose: bool = True):
        super().__init__(
            agent_name="HotelAgent", agent_role="Hotel Booking Specialist",
            model_name=model_name, max_iterations=15, verbose=verbose
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
                description="Search for ALL hotels in a city. Returns all available hotels. Call this ONCE.",
                parameters={"city": "str"},
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
            "filter_by_price_range": AgentAction(
                name="filter_by_price_range",
                description="Filter available hotels to a specific price range. Use this when you need to narrow down options by price.",
                parameters={"min_price": "int - minimum price per night", "max_price": "int - maximum price per night"},
                function=self._tool_filter_by_price_range
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
        return """You are a Hotel Booking Specialist. Select the most appropriate hotel for this trip.

AVAILABLE TOOLS:
1. search_hotels(city) - Returns ALL hotels in a city. Call ONCE at startup.
2. analyze_options() - Analyze hotels by star rating and price. Use AFTER search.
3. filter_by_price_range(min_price, max_price) - Filter hotels to a specific price range. Use when you need to narrow by price.
4. compare_hotels(hotel_ids=[...]) - Compare specific hotels. Use for final decision.
5. get_hotel_details(hotel_id) - Get detailed info about one hotel.
6. finish(result={...}) - Submit your final selection.

CRITICAL WORKFLOW:
1. Call search_hotels(city) exactly ONCE - it returns ALL hotels
2. Use analyze_options() to understand the distribution
3. (Optional) Use filter_by_price_range() if you need to narrow by price
4. Use compare_hotels() to evaluate 2-4 specific options
5. Call finish() with your selection

DO NOT call search_hotels more than once - you already have all hotels after the first call.
Use analyze_options, filter_by_price_range, and compare_hotels to narrow down your selection.

YOUR TASK:
Analyze available hotels and select ONE that best meets the trip requirements.
You don't know the budget constraints - reason independently about what makes sense.

Consider all available options:
- All star ratings (2-5 stars) are equally valid choices
- Evaluate stars, amenities, location, and price
- Reason about tradeoffs between these factors
- Make an autonomous decision based on your analysis

Return format: {"selected_hotels": ["HOTEL_ID"], "reasoning": "Your analysis and decision"}"""
    
    def _tool_search_hotels(self, city: str, **kwargs) -> str:
        """Search ALL hotels in a city. Extra kwargs are ignored."""
        # Return ALL hotels - no filtering at search level
        hotels = self.loader.search(city=city)
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
        
        # Show hotels grouped by star rating - NO PRICE BIAS
        # Sort by hotel_id within each star rating for neutral ordering
        for stars in [5, 4, 3, 2]:
            if by_stars[stars]:
                result.append(f"\n{stars}-star HOTELS ({len(by_stars[stars])} options):")
                # Sort by hotel_id (neutral) instead of price (biased)
                for h in sorted(by_stars[stars], key=lambda x: x['hotel_id'])[:4]:
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

    def _tool_filter_by_price_range(self, min_price: int, max_price: int, **kwargs) -> str:
        """Filter available hotels to a specific price range. Extra kwargs are ignored."""
        hotels = self.state.get_belief("available_hotels", [])
        return self._filter_items_by_price(
            items=hotels,
            min_price=min_price,
            max_price=max_price,
            price_key='price_per_night_usd',
            item_type='hotels',
            group_by_key='stars'
        )

    def _format_item_summary(self, item: Dict, price_key: str) -> str:
        """Format hotel summary with details."""
        amenities = ", ".join(item.get('amenities', [])[:3]) or "No amenities"
        return f"{item['hotel_id']}: {item['name']}, ${item[price_key]}/night | {amenities}"

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
        goal = f"""Select the most appropriate hotel in {query.city}.{meeting_context}{amenities_context}

Analyze available options and select ONE hotel based on your reasoning.

Consider different quality levels and price points:
- Evaluate options from different star ratings (2-star through 5-star)
- Consider amenities, location, and price
- Reason about tradeoffs between these factors

Use your tools to gather information:
- search_hotels() to see what's available
- analyze_options() to understand the quality and price distribution
- compare_hotels() to evaluate specific options
- finish() when you've made your selection

Return: {{"selected_hotels": ["HOTEL_ID"], "reasoning": "Your analysis and decision"}}"""

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

        TRULY AGENTIC: Uses ReAct reasoning loop to autonomously decide how to
        respond to feedback, using available tools including filter_by_price_range.

        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "location_issue"
                - target_price_min/max: suggested price range
                - reasoning: PolicyAgent's explanation
            previous_hotels: Hotels from previous proposal

        Returns:
            Refined HotelSearchResult with new proposal and reasoning
        """
        self.reset_state()

        issue = feedback.get("issue", "general")
        city = feedback.get("city", "")
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)

        if self.verbose:
            print(f"    [HotelAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [HotelAgent] Target price range: ${target_min}-${target_max}/night")

        # Load ALL hotels into state (no pre-filtering)
        all_hotels = self.loader.search(city=city)
        self.state.add_belief("available_hotels", all_hotels)
        self.state.add_belief("search_city", city)

        if not all_hotels:
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=[],
                reasoning="No hotels available in this city."
            )

        # Create goal for ReAct reasoning - agent decides how to use its tools
        goal = f"""The PolicyAgent has requested a hotel refinement based on feedback.

FEEDBACK:
- Issue: {issue}
- Reasoning: {feedback.get('reasoning', 'No details provided')}
- Suggested Price Range: ${target_min}-${target_max}/night

YOUR TASK:
Analyze the feedback and use your tools to find the BEST hotel that addresses this feedback.

You have access to:
- search_hotels() - Already loaded {len(all_hotels)} hotels in {city}
- analyze_options() - See distribution by price and stars
- filter_by_price_range(min_price, max_price) - Narrow options by price
- compare_hotels(hotel_ids=[...]) - Compare specific options
- get_hotel_details(hotel_id) - Get details on a specific hotel

STRATEGY:
1. Use analyze_options() to understand what's available
2. If the feedback mentions a price range, consider using filter_by_price_range()
3. Compare a few options using compare_hotels()
4. Select the BEST hotel that addresses the feedback
5. Call finish() with your selection

Return format: {{"selected_hotels": ["HOTEL_ID"], "reasoning": "Your analysis and decision"}}"""

        # Run ReAct reasoning loop
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

        # Get current available hotels from state (may have been filtered by agent)
        available = self.state.get_belief("available_hotels", all_hotels)

        # TRUST THE LLM'S SELECTION
        if selected_ids:
            hotel_dict = {h['hotel_id']: h for h in available}
            selected_hotels = [hotel_dict[hid] for hid in selected_ids if hid in hotel_dict]

            if selected_hotels:
                refined_hotels = [Hotel(**h) for h in selected_hotels[:1]]  # ONLY 1 hotel

                if self.verbose:
                    prices = [h.price_per_night_usd for h in refined_hotels]
                    stars = [h.stars for h in refined_hotels]
                    print(f"    [HotelAgent] Selected {len(refined_hotels)} hotels ({min(stars)}-{max(stars)}★, ${min(prices)}-${max(prices)}/night)")
                    print(f"    [HotelAgent] Reasoning: {llm_reasoning[:80]}...")

                self.log_message("policy_agent", f"Refined: {len(refined_hotels)} hotels - {llm_reasoning[:100]}", "negotiation")

                return HotelSearchResult(
                    query=HotelQuery(city=city),
                    hotels=refined_hotels,
                    reasoning=llm_reasoning
                )

        # FALLBACK: If LLM didn't select valid hotels, use first option (no bias)
        if self.verbose:
            print(f"    [HotelAgent] LLM selection failed, using fallback (first available)")

        fallback_hotels = [Hotel(**h) for h in available[:1]]
        return HotelSearchResult(
            query=HotelQuery(city=city),
            hotels=fallback_hotels,
            reasoning=f"Fallback: LLM did not select, using first available option."
        )
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
