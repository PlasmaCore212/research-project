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
        """Register all tools (for backward compatibility)."""
        return self._register_negotiation_tools()
    
    def _register_search_tools(self) -> Dict[str, AgentAction]:
        """Register tools for INITIAL SEARCH only (no price filtering)."""
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
            )
        }
    
    def _register_negotiation_tools(self) -> Dict[str, AgentAction]:
        """Register tools for NEGOTIATION (includes price filtering)."""
        tools = self._register_search_tools()
        tools["filter_by_price_range"] = AgentAction(
            name="filter_by_price_range",
            description="Filter available hotels to a specific price range per night. Use this during negotiation when you have a target budget from the orchestrator.",
            parameters={"min_price": "int REQUIRED - minimum price per night", "max_price": "int REQUIRED - maximum price per night"},
            function=self._tool_filter_by_price_range
        )
        return tools
    
    def _get_system_prompt(self) -> str:
        return """You are a Hotel Booking Specialist. Your job: Select ONE best hotel.

=== YOUR ONLY AVAILABLE TOOLS ===
You can ONLY use these exact tool names. Any other tool name will error.

1. search_hotels
   Parameters: city (required), min_stars (optional), max_price (optional)
   Returns all available hotels with amenities. Call ONCE at start.
   Example: {"action": "search_hotels", "action_input": {"city": "SF"}}

2. analyze_hotels
   Parameters: none
   Shows hotel distribution by star rating and price. Use after search.
   Example: {"action": "analyze_hotels", "action_input": {}}

3. compare_hotels
   Parameters: hotel_ids (required - list of 2+ hotel IDs like ["HT0001", "HT0002"])
   Compares specific hotels. IDs come from search_hotels results.
   Example: {"action": "compare_hotels", "action_input": {"hotel_ids": ["HT0031", "HT0087"]}}

4. filter_by_price_range (only available during negotiation)
   Parameters: min_price (required), max_price (required)
   Narrows hotels to price range per night.
   Example: {"action": "filter_by_price_range", "action_input": {"min_price": 200, "max_price": 400}}

5. finish
   Parameters: result (required - dict with "selected_hotels" list and "reasoning" string)
   Submit your final selection. Call this when you've decided.
   Example: {"action": "finish", "action_input": {"result": {"selected_hotels": ["HT0031"], "reasoning": "3★ at $265/night with business center and great location"}}}

=== WORKFLOW ===
Step 1: search_hotels → Get all options with amenities
Step 2: analyze_hotels → See distribution by stars/price
Step 3: compare_hotels → Compare 2-3 top candidates
Step 4: finish → Return your selection

You should finish in 3-5 iterations. Don't overthink it.

=== DECISION PRINCIPLES ===
- Balance cost vs. location vs. quality vs. amenities
- Consider business travel needs (meeting proximity, work facilities)
- Don't default to cheapest - optimize for overall trip value
- Check required amenities if specified

=== WHEN TO CALL FINISH ===
Call finish() when you can answer: "Why is this hotel the best option?"
If you have a clear preference with reasoning, finish immediately.
DON'T wait until max iterations - decide and finish early.

=== CRITICAL RULES ===
✓ Return ONE hotel only in selected_hotels list
✓ Use exact tool names from list above
✗ Don't call same tool with same parameters twice
✗ Don't hallucinate tool names like "filter_hotels" or "get_details"

FINISH FORMAT: {"selected_hotels": ["HT0XXX"], "reasoning": "why this is best"}"""
    
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
        
        # Use search-only tools (no price filtering during initial search)
        original_tools = self.tools
        self.tools = self._register_search_tools()
        
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

OBJECTIVE: Select the single best hotel for this business trip.

CONSTRAINTS:
- City: {query.city}
- Budget: {f"Max ${query.max_price_per_night}/night" if query.max_price_per_night else "Flexible"}
- Quality: {f"Min {query.min_stars}★" if query.min_stars else "Any star rating"}
- Location: {f"Within {query.max_distance_to_center_km}km of center" if query.max_distance_to_center_km else "Flexible"}
- Required amenities: {', '.join(query.required_amenities) if query.required_amenities else "None"}

OPTIMIZATION CRITERIA:
- Balance cost vs. location vs. quality vs. amenities
- Consider business context (productivity, meeting access, comfort)
- Evaluate all star ratings (2★, 3★, 4★, 5★) on their merits
- Don't default to cheapest - optimize for trip value

AVAILABLE TOOLS: search_hotels, analyze_hotels, compare_hotels, finish

IMPORTANT: search_hotels returns ALL hotels with amenities - only call it ONCE.

Use tools in whatever sequence makes sense. When confident in your choice, return:
{{"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this hotel best meets the criteria"}}"""

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
        
        # Restore original tools
        self.tools = original_tools
        return HotelSearchResult(query=query, hotels=top_hotels, reasoning=reasoning)
    
    def refine_proposal(self, feedback: Dict[str, Any], previous_hotels: List[Dict]) -> HotelSearchResult:
        """
        CNP NEGOTIATION: Refine hotel proposal based on PolicyAgent feedback.

        Uses ReAct reasoning to autonomously respond to feedback while being guided
        toward the target price range through smart pre-loading and clear examples.

        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "location_issue"
                - target_price_min/max: suggested price range per night
                - reasoning: PolicyAgent's explanation
                - city: destination city
            previous_hotels: Hotels from previous proposal

        Returns:
            Refined HotelSearchResult with new proposal and reasoning
        """
        self.reset_state()
        
        # Use negotiation tools (includes price filtering)
        original_tools = self.tools
        self.tools = self._register_negotiation_tools()

        issue = feedback.get("issue", "general")
        city = feedback.get("city", "")
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)

        if self.verbose:
            print(f"    [HotelAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [HotelAgent] Target price range: ${target_min}-${target_max}/night")

        # Load ALL hotels - let agent use filter_by_price_range tool
        all_hotels = self.loader.search(city=city)
        
        if not all_hotels:
            return HotelSearchResult(
                query=HotelQuery(city=city),
                hotels=[],
                reasoning="No hotels available in this city."
            )
        
        # Load into state - NO pre-filtering
        self.state.add_belief("available_hotels", all_hotels)
        self.state.add_belief("search_city", city)
        
        if self.verbose:
            print(f"    [HotelAgent] Loaded {len(all_hotels)} hotels for agent to filter")

        # BUILD GOAL PROMPT - Keep it short and directive
        goal = f"""NEGOTIATION TASK: Refine hotel selection based on feedback.

CONTEXT:
- Issue: {issue}
- Target price: ${target_min}-${target_max}/night
- City: {city}
- Previous: {[h.get('hotel_id', 'N/A') for h in previous_hotels]}
- Hotels loaded: {len(all_hotels)} options ALREADY IN MEMORY

YOUR JOB: Select ONE hotel in ${target_min}-${target_max}/night range.

STEP-BY-STEP WORKFLOW (DO THIS EXACTLY):
1. {{action: filter_by_price_range, action_input: {{min_price: {target_min}, max_price: {target_max}}}}}
2. {{action: compare_hotels, action_input: {{hotel_ids: ["HT0XXX", "HT0YYY"]}}}} (pick 2-3 IDs from filter results)
3. {{action: finish, action_input: {{result: {{selected_hotels: ["HT0XXX"], reasoning: "..."}}}}}}

CRITICAL RULES:
- DO NOT call search_hotels (hotels already loaded)
- DO NOT make up tool names (only use: filter_by_price_range, compare_hotels, analyze_hotels, finish)
- DO return exactly ONE hotel in finish()
- Finish in 3-4 iterations maximum

AVAILABLE TOOLS: filter_by_price_range, analyze_hotels, compare_hotels, finish"""

        # Run ReAct reasoning loop
        result = self.run(goal)

        # Parse LLM's selection
        selected_ids = []
        llm_reasoning = ""

        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    # Extract JSON from string (handles extra text before/after)
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    selected_ids = parsed.get("selected_hotels", parsed.get("top_hotels", []))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    # Already a dict
                    selected_ids = final.get("selected_hotels", final.get("top_hotels", []))
                    llm_reasoning = final.get("reasoning", str(final))
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
                if self.verbose:
                    print(f"    [HotelAgent] Failed to parse LLM response: {e}")
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
            if self.verbose:
                print(f"    [HotelAgent] ReAct loop failed: {result.get('error')}")

        # Get current available hotels from state (may have been filtered by agent)
        available = self.state.get_belief("available_hotels", all_hotels)

        # TRUST THE LLM'S SELECTION
        if selected_ids:
            hotel_dict = {h['hotel_id']: h for h in available}
            
            # Also build name-to-id lookup for fallback matching
            name_to_id = {h.get('name', '').lower(): h['hotel_id'] for h in available}

            # Normalize IDs - handle multiple formats
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract ID from strings like "HT0031: Holiday Inn..., 3★, $265/night" or just "HT0031"
                    if item.startswith('HT') and ':' in item:
                        hotel_id = item.split(':')[0].strip()
                        normalized_ids.append(hotel_id)
                    elif item.startswith('HT'):
                        normalized_ids.append(item.strip())
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

            if selected_hotels:
                # Take only the first hotel (CNP expects single proposal)
                refined_hotels = [Hotel(**h) for h in selected_hotels[:1]]

                if self.verbose:
                    hotel = refined_hotels[0]
                    print(f"    [HotelAgent] ✓ Selected {hotel.hotel_id}: {hotel.stars}★ ${hotel.price_per_night_usd}/night")
                    print(f"    [HotelAgent] Reasoning: {llm_reasoning[:100]}{'...' if len(llm_reasoning) > 100 else ''}")

                self.log_message("policy_agent", f"Refined: {refined_hotels[0].hotel_id} - {llm_reasoning[:100]}", "negotiation")

                # Restore original tools
                self.tools = original_tools
                return HotelSearchResult(
                    query=HotelQuery(city=city),
                    hotels=refined_hotels,
                    reasoning=llm_reasoning
                )

        # FALLBACK: If LLM didn't select valid hotels, use first available
        if self.verbose:
            print(f"    [HotelAgent] ⚠ LLM selection failed, using fallback (first available)")

        fallback_hotels = [Hotel(**h) for h in available[:1]]
        
        # Restore original tools
        self.tools = original_tools
        return HotelSearchResult(
            query=HotelQuery(city=city),
            hotels=fallback_hotels,
            reasoning=f"Fallback: LLM did not select valid hotel, returning first available option."
        )
    
    def _build_reasoning_trace(self, query: HotelQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Hotel Search ReAct Trace",
                f"**City**: {query.city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
