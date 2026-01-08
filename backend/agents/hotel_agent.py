# backend/agents/hotel_agent.py
"""
Hotel Agent with ReAct Pattern and Chain-of-Thought Prompting

This agent specializes in hotel search and selection for business travel.
It uses the ReAct pattern (Thought -> Action -> Observation) to:
1. Understand the traveler's accommodation requirements
2. Search for available hotels
3. Analyze proximity to business centers
4. Select the best hotels based on business criteria

References:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
"""

from .base_agent import BaseReActAgent, AgentAction
from .models import HotelQuery, HotelSearchResult, Hotel
from data.loaders import HotelDataLoader
from typing import List, Dict, Any, Optional
import json


class HotelAgent(BaseReActAgent):
    """
    Agentic Hotel Search Agent with ReAct reasoning.
    
    This agent autonomously:
    - Searches hotel databases with various criteria
    - Evaluates proximity to business centers
    - Compares amenities and star ratings
    - Selects optimal hotels for business travelers
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="HotelAgent",
            agent_role="Hotel Booking Specialist",
            model_name=model_name,
            max_iterations=5,
            verbose=verbose
        )
        
        self.loader = HotelDataLoader()
        self.tools = self._register_tools()
        
        # Track search history
        self.search_history: List[Dict] = []
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        """Register tools available to the Hotel Agent"""
        return {
            "search_hotels": AgentAction(
                name="search_hotels",
                description="Search for hotels in a city with optional filters",
                parameters={
                    "city": "str - city code (e.g., 'SF', 'NYC')",
                    "max_price_per_night": "int (optional) - maximum price per night in USD",
                    "min_stars": "int (optional) - minimum star rating (1-5)",
                    "max_distance_km": "float (optional) - maximum distance to business center in km"
                },
                function=self._tool_search_hotels
            ),
            "get_hotel_details": AgentAction(
                name="get_hotel_details",
                description="Get detailed information about a specific hotel",
                parameters={
                    "hotel_id": "str - the hotel ID to look up"
                },
                function=self._tool_get_hotel_details
            ),
            "compare_hotels": AgentAction(
                name="compare_hotels",
                description="Compare multiple hotels on specific criteria",
                parameters={
                    "hotel_ids": "list - list of hotel IDs to compare",
                    "criteria": "str - focus on: 'price', 'location', 'quality', or 'overall'"
                },
                function=self._tool_compare_hotels
            ),
            "check_amenities": AgentAction(
                name="check_amenities",
                description="Check if hotels have specific amenities",
                parameters={
                    "hotel_ids": "list - list of hotel IDs to check",
                    "required_amenities": "list - amenities to check for (e.g., ['WiFi', 'Gym'])"
                },
                function=self._tool_check_amenities
            ),
            "analyze_area_options": AgentAction(
                name="analyze_area_options",
                description="Analyze hotel options in different business areas of a city",
                parameters={
                    "city": "str - city code"
                },
                function=self._tool_analyze_area_options
            )
        }
    
    def _get_system_prompt(self) -> str:
        """Get the domain-specific system prompt for Hotel Agent"""
        return """You are an expert Hotel Booking Specialist AI Agent for business travel.

YOUR EXPERTISE:
- Finding optimal hotels for business travelers
- Evaluating proximity to business centers and meeting locations
- Assessing hotel amenities for business needs
- Balancing quality, location, and price

REASONING APPROACH (Chain-of-Thought):
When analyzing hotels, think through:
1. LOCATION: How close is it to the business center? Will commute be an issue?
2. QUALITY: What star rating? What do reviews suggest?
3. AMENITIES: Does it have WiFi, conference rooms, gym, restaurant?
4. PRICE: What's the value proposition? Price per star?
5. CONVENIENCE: Check-in/out times, airport shuttle, business services?

BUSINESS TRAVEL PRIORITIES:
- Proximity to business center is critical (< 2km ideal)
- Must have reliable WiFi
- Prefer 3+ star ratings for comfort
- Conference room access is a plus
- Early check-in / late check-out flexibility helps
- Restaurant on-site saves time"""
    
    def _tool_search_hotels(
        self,
        city: str,
        max_price_per_night: Optional[int] = None,
        min_stars: Optional[int] = None,
        max_distance_km: Optional[float] = None
    ) -> str:
        """Tool: Search hotels database"""
        
        hotels = self.loader.search(
            city=city,
            max_price_per_night=max_price_per_night,
            min_stars=min_stars,
            max_distance_to_center_km=max_distance_km
        )
        
        if not hotels:
            return f"No hotels found in {city} matching criteria."
        
        # Update beliefs
        self.state.add_belief("search_city", city)
        self.state.add_belief("available_hotels", hotels)
        self.state.add_belief("hotel_count", len(hotels))
        
        # Format results
        result_lines = [f"Found {len(hotels)} hotels in {city}:"]
        for h in hotels[:10]:  # Show top 10
            result_lines.append(
                f"  - {h['hotel_id']}: {h['name']}, "
                f"{h['stars']}*, ${h['price_per_night_usd']}/night, "
                f"{h['distance_to_business_center_km']:.1f}km from {h['business_area']}"
            )
        
        if len(hotels) > 10:
            result_lines.append(f"  ... and {len(hotels) - 10} more")
        
        # Track search
        self.search_history.append({
            "city": city,
            "max_price": max_price_per_night,
            "results": len(hotels)
        })
        
        return "\n".join(result_lines)
    
    def _tool_get_hotel_details(self, hotel_id: str) -> str:
        """Tool: Get details about a specific hotel"""
        
        hotels = self.state.get_belief("available_hotels", [])
        
        for h in hotels:
            if h['hotel_id'] == hotel_id:
                amenities = ", ".join(h.get('amenities', []))
                return f"""Hotel Details for {hotel_id}:
- Name: {h['name']}
- City: {h['city_name']}
- Business Area: {h['business_area']}
- Tier: {h['tier']}
- Stars: {h['stars']}
- Price: ${h['price_per_night_usd']}/night
- Distance to Business Center: {h['distance_to_business_center_km']:.2f}km
- Distance to Airport: {h['distance_to_airport_km']:.2f}km
- Amenities: {amenities}
- Rooms Available: {h.get('rooms_available', 'Unknown')}
- Coordinates: {h.get('coordinates', {})}"""
        
        return f"Hotel {hotel_id} not found in recent search results."
    
    def _tool_compare_hotels(self, hotel_ids: List[str], criteria: str = "overall") -> str:
        """Tool: Compare multiple hotels"""
        
        hotels = self.state.get_belief("available_hotels", [])
        hotel_dict = {h['hotel_id']: h for h in hotels}
        
        to_compare = [hotel_dict[hid] for hid in hotel_ids if hid in hotel_dict]
        
        if not to_compare:
            return "No valid hotels to compare. Search for hotels first."
        
        comparison_lines = [f"Comparison of {len(to_compare)} hotels ({criteria} focus):"]
        comparison_lines.append("-" * 50)
        
        if criteria == "price":
            sorted_hotels = sorted(to_compare, key=lambda x: x['price_per_night_usd'])
            for i, h in enumerate(sorted_hotels, 1):
                diff = h['price_per_night_usd'] - sorted_hotels[0]['price_per_night_usd']
                comparison_lines.append(
                    f"{i}. {h['hotel_id']}: ${h['price_per_night_usd']}/night "
                    f"({'cheapest' if i == 1 else f'+${diff}'})"
                )
        
        elif criteria == "location":
            sorted_hotels = sorted(to_compare, key=lambda x: x['distance_to_business_center_km'])
            for i, h in enumerate(sorted_hotels, 1):
                comparison_lines.append(
                    f"{i}. {h['hotel_id']}: {h['distance_to_business_center_km']:.1f}km from {h['business_area']} "
                    f"({'closest' if i == 1 else ''})"
                )
        
        elif criteria == "quality":
            sorted_hotels = sorted(to_compare, key=lambda x: -x['stars'])  # Higher is better
            for i, h in enumerate(sorted_hotels, 1):
                value = h['price_per_night_usd'] / h['stars']
                comparison_lines.append(
                    f"{i}. {h['hotel_id']}: {h['stars']}* (${value:.0f}/star)"
                )
        
        else:  # overall
            def overall_score(h):
                # Lower score is better
                location_score = h['distance_to_business_center_km'] / 5  # normalize to ~0-1
                price_score = h['price_per_night_usd'] / 500  # normalize
                quality_score = (5 - h['stars']) / 5  # invert so higher stars = lower score
                
                return 0.4 * location_score + 0.3 * price_score + 0.3 * quality_score
            
            sorted_hotels = sorted(to_compare, key=overall_score)
            for i, h in enumerate(sorted_hotels, 1):
                score = overall_score(h)
                comparison_lines.append(
                    f"{i}. {h['hotel_id']}: {h['name']}, {h['stars']}*, "
                    f"${h['price_per_night_usd']}/night, {h['distance_to_business_center_km']:.1f}km "
                    f"(score: {score:.2f})"
                )
        
        return "\n".join(comparison_lines)
    
    def _tool_check_amenities(
        self,
        hotel_ids: List[str],
        required_amenities: List[str]
    ) -> str:
        """Tool: Check if hotels have specific amenities"""
        
        hotels = self.state.get_belief("available_hotels", [])
        hotel_dict = {h['hotel_id']: h for h in hotels}
        
        results = []
        for hid in hotel_ids:
            if hid in hotel_dict:
                h = hotel_dict[hid]
                hotel_amenities = set(h.get('amenities', []))
                required = set(required_amenities)
                has = required & hotel_amenities
                missing = required - hotel_amenities
                
                status = "✓ All required" if not missing else f"✗ Missing: {', '.join(missing)}"
                results.append(f"  {hid}: {status}")
        
        if not results:
            return "No valid hotels to check."
        
        return f"Amenity Check ({', '.join(required_amenities)}):\n" + "\n".join(results)
    
    def _tool_analyze_area_options(self, city: str) -> str:
        """Tool: Analyze hotel options by business area"""
        
        hotels = self.loader.search(city=city)
        
        if not hotels:
            return f"No hotels found in {city}"
        
        # Group by business area
        areas = {}
        for h in hotels:
            area = h.get('business_area', 'Unknown')
            if area not in areas:
                areas[area] = []
            areas[area].append(h)
        
        result_lines = [f"Hotel Distribution in {city}:"]
        for area, area_hotels in areas.items():
            prices = [h['price_per_night_usd'] for h in area_hotels]
            stars = [h['stars'] for h in area_hotels]
            result_lines.append(
                f"\n{area}:"
                f"\n  - Hotels: {len(area_hotels)}"
                f"\n  - Price range: ${min(prices)} - ${max(prices)}"
                f"\n  - Star range: {min(stars)} - {max(stars)}"
                f"\n  - Avg distance: {sum(h['distance_to_business_center_km'] for h in area_hotels)/len(area_hotels):.1f}km"
            )
        
        return "\n".join(result_lines)
    
    def search_hotels(self, query: HotelQuery) -> HotelSearchResult:
        """
        Main entry point for hotel search using ReAct reasoning.
        
        This method triggers the agentic ReAct loop to find the best hotels.
        """
        
        # Reset state for new search
        self.reset_state()
        
        # Build constraints description
        constraints = []
        if query.max_price_per_night:
            constraints.append(f"Max ${query.max_price_per_night}/night")
        if query.min_stars:
            constraints.append(f"Min {query.min_stars} stars")
        if query.max_distance_to_center_km:
            constraints.append(f"Within {query.max_distance_to_center_km}km of business center")
        if query.required_amenities:
            constraints.append(f"Must have: {', '.join(query.required_amenities)}")
        
        constraints_str = "; ".join(constraints) if constraints else "No specific constraints"
        
        # Build the goal description
        goal = f"""Find the top 3 best hotels for a business traveler:
- City: {query.city}
- Constraints: {constraints_str}

Steps to follow:
1. First, search for available hotels matching the basic criteria
2. Analyze the options considering location, price, and quality
3. If amenities are required, verify they are available
4. Compare the top candidates on overall value
5. Select the 3 best hotels and explain why

Return your final answer as a JSON object with:
- top_3_hotels: list of hotel IDs
- reasoning: detailed explanation of your selection"""

        # Run ReAct loop
        result = self.run(goal)
        
        # Extract hotels from result
        if result["success"]:
            try:
                final_answer = result["result"]
                if isinstance(final_answer, str):
                    if "{" in final_answer:
                        json_str = final_answer[final_answer.find("{"):final_answer.rfind("}")+1]
                        parsed = json.loads(json_str)
                        top_ids = parsed.get("top_3_hotels", [])
                        llm_reasoning = parsed.get("reasoning", "")
                    else:
                        top_ids = []
                        llm_reasoning = final_answer
                else:
                    top_ids = final_answer.get("top_3_hotels", []) if isinstance(final_answer, dict) else []
                    llm_reasoning = final_answer.get("reasoning", "") if isinstance(final_answer, dict) else str(final_answer)
                
                # Get hotels from beliefs
                available_hotels = self.state.get_belief("available_hotels", [])
                hotel_dict = {h['hotel_id']: h for h in available_hotels}
                
                # Map to Hotel objects
                top_hotels = []
                for hid in top_ids[:3]:
                    if hid in hotel_dict:
                        top_hotels.append(Hotel(**hotel_dict[hid]))
                
                # Fallback if no valid hotels found
                if not top_hotels and available_hotels:
                    top_hotels = [Hotel(**h) for h in available_hotels[:3]]
                    llm_reasoning = "Fallback selection: top 3 by default sorting."
                
            except Exception as e:
                available_hotels = self.state.get_belief("available_hotels", [])
                top_hotels = [Hotel(**h) for h in available_hotels[:3]] if available_hotels else []
                llm_reasoning = f"Selection completed with fallback: {str(e)}"
        else:
            # ReAct failed, use simple search
            hotels = self.loader.search(
                city=query.city,
                max_price_per_night=query.max_price_per_night,
                min_stars=query.min_stars,
                max_distance_to_center_km=query.max_distance_to_center_km,
                required_amenities=query.required_amenities
            )
            top_hotels = [Hotel(**h) for h in hotels[:3]] if hotels else []
            llm_reasoning = f"ReAct reasoning failed: {result.get('error', 'Unknown error')}. Used fallback search."
        
        # Build comprehensive reasoning trace
        reasoning = self._build_react_reasoning(query, result, llm_reasoning)
        
        # Log completion message
        self.log_message(
            to_agent="orchestrator",
            content=f"Found {len(top_hotels)} recommended hotels",
            msg_type="result"
        )
        
        return HotelSearchResult(
            query=query,
            hotels=top_hotels,
            reasoning=reasoning
        )
    
    def _build_react_reasoning(
        self,
        query: HotelQuery,
        react_result: Dict,
        final_reasoning: str
    ) -> str:
        """Build the full ReAct reasoning trace for transparency"""
        
        reasoning_parts = [
            f"## Hotel Search ReAct Reasoning Trace",
            f"**Agent**: {self.agent_name}",
            f"**Goal**: Find hotels in {query.city}",
            f"**Iterations**: {react_result.get('iterations', 0)}",
            f"**Success**: {react_result.get('success', False)}",
            "",
            "### Reasoning Steps:",
        ]
        
        for step in react_result.get("reasoning_trace", []):
            reasoning_parts.append(f"""
**Step {step.step_number}**:
- **Thought**: {step.thought}
- **Action**: `{step.action}({json.dumps(step.action_input)})`
- **Observation**: {step.observation[:200]}{'...' if len(step.observation) > 200 else ''}
""")
        
        reasoning_parts.append(f"""
### Final Selection:
{final_reasoning}
""")
        
        return "\n".join(reasoning_parts)
