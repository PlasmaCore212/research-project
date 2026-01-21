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
                description="Search for hotels in a city. Returns ALL hotels with their amenities. Call this ONCE only - you cannot filter by amenities here, just see what's available.",
                parameters={"city": "str", "min_stars": "int (optional)", "max_price": "int (optional)"},
                function=self._tool_search_hotels
            ),
            "analyze_hotels": AgentAction(
                name="analyze_hotels",
                description="Analyze hotel distribution by star rating (2-5 stars). Shows price ranges and distances. Use this to understand what's available.",
                parameters={},
                function=self._tool_analyze_hotels
            ),
            "compare_hotels": AgentAction(
                name="compare_hotels",
                description="Compare specific hotels side-by-side. Use this after narrowing down to a few candidates.",
                parameters={"hotel_ids": "list[str] REQUIRED - list of hotel IDs like ['HT0001', 'HT0002']",
                           "criteria": "str (optional) - 'price', 'location', 'quality', or 'overall' (default)"},
                function=self._tool_compare_hotels
            ),
            "filter_by_price_range": AgentAction(
                name="filter_by_price_range",
                description="Filter available hotels to a specific price range per night. Use this to narrow down options when you have a target budget.",
                parameters={"min_price": "int REQUIRED - minimum price per night", "max_price": "int REQUIRED - maximum price per night"},
                function=self._tool_filter_by_price_range
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Hotel Booking Specialist. Select the best hotel for this trip.

WORKFLOW:
1. search_hotels - Load all options (call ONCE - don't call again!)
2. analyze_hotels - See star ratings and price ranges
3. compare_hotels - Compare specific hotel IDs from search results
4. finish - Return your selection

CRITICAL RULES:
- search_hotels shows ALL hotels with amenities - only call ONCE
- To find hotels with specific amenities: read the search results, pick hotel IDs with those amenities
- DO NOT call search_hotels multiple times trying to filter
- NO QUALITY BIAS: Don't assume 2-3 star is best - consider 4-5 star value

Return format: {"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this hotel"}

IMPORTANT: Select exactly ONE hotel only - not multiple options!"""
    
    def _tool_search_hotels(self, city: str, min_stars: Optional[int] = None, 
                            max_price: Optional[int] = None, **kwargs) -> str:
        """Search hotels with filters. Extra kwargs are ignored."""
        hotels = self.loader.search(city=city)
        if not hotels:
            return f"No hotels found in {city}."
        
        # Apply filters
        if min_stars:
            hotels = [h for h in hotels if h.get('stars', 0) >= min_stars]
        if max_price:
            hotels = [h for h in hotels if h.get('price_per_night_usd', 9999) <= max_price]
            
        if not hotels:
            return f"No hotels found in {city} matching criteria (min {min_stars}★, max ${max_price})."
        
        self.state.add_belief("search_city", city)
        self.state.add_belief("available_hotels", hotels)
        self.state.add_belief("hotel_count", len(hotels))
        
        # Group by star rating for clear display
        by_stars = {5: [], 4: [], 3: [], 2: [], 1: []}
        for h in hotels:
            stars = h.get('stars', 3)
            if stars in by_stars:
                by_stars[stars].append(h)
        
        filter_desc = []
        if min_stars: filter_desc.append(f"min {min_stars}★")
        if max_price: filter_desc.append(f"max ${max_price}")
        filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
        
        result = [f"Found {len(hotels)} hotels in {city}{filter_str}:"]
        
        # Show hotels grouped by star rating
        for stars in [5, 4, 3, 2]:
            if by_stars[stars]:
                result.append(f"\n{stars}-star HOTELS ({len(by_stars[stars])} options):")
                # Sort by hotel_id (neutral)
                for h in sorted(by_stars[stars], key=lambda x: x['hotel_id'])[:4]:
                    amenities = ", ".join(h.get('amenities', [])[:4]) or "No amenities listed"
                    distance = h.get('distance_to_meeting_km', h.get('distance_to_business_center_km', 0))
                    distance_label = "from meeting" if 'distance_to_meeting_km' in h else "from center"
                    result.append(f"  - {h['hotel_id']}: {h['name']}, ${h['price_per_night_usd']}/night, "
                                 f"{distance:.1f}km {distance_label} | Amenities: {amenities}")
        
        if len(hotels) > 20:
            result.append(f"\n... and {len(hotels)-16} more. Use filters to narrow down.")
            
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
    
    def _tool_analyze_hotels(self, **kwargs) -> str:
        """Analyze available hotels by price tier and quality."""
        hotels = self.state.get_belief("available_hotels", [])
        if not hotels:
            return "No hotels to analyze. Run search_hotels first."

        # Group by star rating
        by_stars = {}
        for h in hotels:
            stars = h.get('stars', 3)
            by_stars.setdefault(stars, []).append(h)
    
        result = [f"Hotel Analysis ({len(hotels)} total):"]
        result.append("")
        
        for stars in sorted(by_stars.keys(), reverse=True):
            store_hotels = by_stars[stars]
            prices = [h.get('price_per_night_usd', 0) for h in store_hotels]
            avg_price = sum(prices) / len(prices) if prices else 0
            distances = [h.get('distance_to_business_center_km', 0) for h in store_hotels]
            
            result.append(f"  {stars}★ Hotels: {len(store_hotels)} options")
            result.append(f"    Price range: ${min(prices)} - ${max(prices)} (Avg: ${avg_price:.0f})")
            if distances:
                 result.append(f"    Distance range: {min(distances):.1f}km - {max(distances):.1f}km from center")
            result.append("")
            
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

STEPS:
1. search_hotels() - Call ONCE to load ALL hotels (shows amenities in results)
2. analyze_hotels() - See star rating distribution
3. compare_hotels() - Pick hotel IDs from search results and compare them
4. finish() - Return your choice

IMPORTANT:
- search_hotels() returns ALL hotels with amenities listed - only call it ONCE
- Don't call search_hotels() again to "filter" - it doesn't work that way
- Pick specific hotel IDs from the search results to compare

Return: {{"selected_hotels": ["HOTEL_ID"], "reasoning": "Brief explanation"}}"""

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
                    selected_ids = (
                        final.get("selected_hotels") or 
                        final.get("selected_hotel") or
                        final.get("top_hotels") or 
                        final.get("top_options") or
                        final.get("selection") or
                        final.get("hotel_id") or
                        []
                    )
                    # Handle single string case
                    if isinstance(selected_ids, str):
                        selected_ids = [selected_ids]

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
            # - description strings: ["HT0031: Holiday Inn..., 3★, $265/night"]
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract hotel ID from description strings like "HT0031: Holiday Inn..., 3★, $265/night"
                    if item.startswith('HT') and ':' in item:
                        hotel_id = item.split(':')[0].strip()
                        normalized_ids.append(hotel_id)
                    elif item.startswith('HT'):
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

        # Load hotels
        all_hotels = self.loader.search(city=city)
        
        # PROGRAMMATICALLY ENFORCE PRICE CONSTRAINT from feedback
        # This ensures the agent cannot "forget" to filter
        filtered_hotels = all_hotels
        if target_max < 9999 or target_min > 0:
            filtered_hotels = [
                h for h in all_hotels 
                if target_min <= h.get('price_per_night_usd', 0) <= target_max
            ]
            if not filtered_hotels:
                # If filter is too strict, revert to all but warn
                filtered_hotels = all_hotels
                issue += " (Warning: No hotels found in target price range, showing all)"
            else:
                 if self.verbose:
                    print(f"    [HotelAgent] Enforcing price filter: ${target_min}-${target_max} ({len(filtered_hotels)} options)")

        self.state.add_belief("available_hotels", filtered_hotels)
        self.state.add_belief("search_city", city)

        if not filtered_hotels:
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=[],
                reasoning="No hotels available in this city."
            )

        # Create goal for ReAct reasoning
        goal = f"""TASK: Refine your hotel proposal based on orchestrator feedback.

=== FEEDBACK FROM ORCHESTRATOR ===
Issue: {issue}
Reasoning: {feedback.get('reasoning', 'No details provided')}
Target Price Range: ${target_min} to ${target_max} per night

=== CURRENT SITUATION ===
✓ {len(filtered_hotels)} hotels are ALREADY LOADED and filtered to target price
✓ You do NOT need to search again
✓ Your job: Analyze → Compare → Select ONE hotel

=== YOUR AVAILABLE TOOLS ===
1. analyze_hotels()
2. filter_by_price_range(min_price=X, max_price=Y)
3. compare_hotels(hotel_ids=["HT0001", "HT0002", ...])
4. finish(result={{"selected_hotels": ["HT_ID"], "reasoning": "..."}})

⚠️ TOOLS THAT DO NOT EXIST (DO NOT USE):
- search_hotels (already done!)
- filter_hotels (doesn't exist!)
- search (doesn't exist!)

=== STEP-BY-STEP INSTRUCTIONS ===

STEP 1: Analyze available hotels
   Tool: analyze_hotels()
   Why: See what star ratings are available in the filtered range
   
STEP 2: Compare specific options
   Tool: compare_hotels(hotel_ids=["HT0001", "HT0005", "HT0010"])
   Why: Compare 2-4 hotels from DIFFERENT star levels
   Note: Use actual hotel IDs from the analyze results!
   
STEP 3: Select the SINGLE best hotel
   Tool: finish(result={{"selected_hotels": ["SINGLE_HOTEL_ID"], "reasoning": "..."}})
   Why: Return exactly ONE hotel that best addresses the feedback
   
=== CRITICAL RULES ===
1. DO NOT call search_hotels - hotels are already loaded!
2. DO NOT invent tools like "filter_hotels" or "search"
3. MUST call analyze_hotels FIRST
4. Return exactly ONE hotel ID, not multiple
5. Consider ALL star ratings in target range (2★, 3★, 4★, 5★)
6. Don't assume lower stars are better - consider value and quality

=== EXAMPLE WORKFLOW ===
Iteration 1: analyze_hotels()
Iteration 2: compare_hotels(hotel_ids=["HT0005", "HT0010"])
Iteration 3: finish(result={{"selected_hotels": ["HT0005"], "reasoning": "4★ offers best value"}})

NOW EXECUTE THIS WORKFLOW!"""

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

            # Normalize IDs - handle both raw IDs and description strings
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract ID from "HT0031: Holiday Inn..., 3★, $265/night" or just "HT0031"
                    if item.startswith('HT') and ':' in item:
                        hotel_id = item.split(':')[0].strip()
                    else:
                        hotel_id = item.strip()
                    normalized_ids.append(hotel_id)
                else:
                    normalized_ids.append(item)

            selected_hotels = [hotel_dict[hid] for hid in normalized_ids if hid in hotel_dict]

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
