# backend/agents/flight_agent.py
"""Flight Agent with ReAct Pattern for business travel flight search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import FlightQuery, FlightSearchResult, Flight
from data.loaders import FlightDataLoader
from typing import List, Dict, Any, Optional
import json


class FlightAgent(BaseReActAgent):
    """Agentic Flight Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "mistral-small", verbose: bool = True):
        super().__init__(
            agent_name="FlightAgent", agent_role="Flight Booking Specialist",
            model_name=model_name, max_iterations=15, verbose=verbose
        )
        self.loader = FlightDataLoader()
        self.tools = self._register_tools()
    
    def _extract_best_result_from_state(self) -> dict:
        """Extract SINGLE flight when agent fails to call finish()."""
        flights = self.state.get_belief("available_flights", [])
        if not flights:
            return {"result": "No flights found"}

        # NO BIAS - just use the first flight from search results
        # The LLM should have made a decision, this is just emergency fallback
        first_flight = flights[0]

        return {"selected_flights": [first_flight['flight_id']],
                "reasoning": "Fallback: Agent did not make a selection, returning first available option."}
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        return {
            "search_flights": AgentAction(
                name="search_flights",
                description="Search for flights between cities. Returns all flight details. Only call this ONCE - it loads all available options.",
                parameters={"from_city": "str", "to_city": "str", "max_price": "int (optional)",
                           "departure_after": "str (optional)", "departure_before": "str (optional)"},
                function=self._tool_search_flights
            ),
            "analyze_flights": AgentAction(
                name="analyze_flights",
                description="Analyze flight distribution by class (Economy, Business, First). Shows price ranges for each class. Use this to understand what's available.",
                parameters={},
                function=self._tool_analyze_flights
            ),
            "compare_flights": AgentAction(
                name="compare_flights",
                description="Compare specific flights side-by-side. IMPORTANT: You MUST provide a list of flight IDs from your search results (e.g., ['FL0001', 'FL0002']). Use this after narrowing down to a few candidates.",
                parameters={"flight_ids": "list[str] REQUIRED - You MUST provide a list of flight IDs from search results, like ['FL0001', 'FL0002', 'FL0003']. This is NOT optional.",
                           "criteria": "str (optional) - 'price', 'timing', or 'overall' (default)"},
                function=self._tool_compare_flights
            ),
            "filter_by_price_range": AgentAction(
                name="filter_by_price_range",
                description="Filter available flights to a specific price range. Use this to narrow down options when you have a target budget.",
                parameters={"min_price": "int REQUIRED - minimum price", "max_price": "int REQUIRED - maximum price"},
                function=self._tool_filter_by_price_range
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Flight Booking Specialist. Your goal is to find the best flight for business travel.

DECISION PRINCIPLES:
- Optimize for overall trip value, not just lowest price
- Consider business travel needs: productivity, rest, schedule flexibility
- Balance cost vs. time vs. comfort based on trip context
- Business/First class may offer better value than Economy for certain trips

AVAILABLE TOOLS:
- search_flights: Find flights between cities
- analyze_flights: Understand flight distribution by class and price
- compare_flights: Compare specific options side-by-side
- filter_by_price_range: Narrow down to specific budget range
- finish: Submit your final selection

IMPORTANT: 
- Select exactly ONE flight
- Provide clear reasoning for your choice
- Use tools in whatever sequence makes sense for the situation

Return format: {"selected_flights": ["SINGLE_FLIGHT_ID"], "reasoning": "detailed justification"}"""
    
    def _tool_search_flights(self, from_city: str, to_city: str, max_price: Optional[int] = None,
                             departure_after: Optional[str] = None, departure_before: Optional[str] = None,
                             **kwargs) -> str:
        """Search flights. Extra kwargs are ignored to handle LLM parameter variations."""
        flights = self.loader.search(from_city=from_city, to_city=to_city, max_price=max_price,
                                     departure_after=departure_after, departure_before=departure_before)
        if not flights:
            return f"No flights found from {from_city} to {to_city} matching criteria."

        self.state.add_belief("available_flights", flights)
        self.state.add_belief("flight_count", len(flights))

        # Group by class to show diverse options (prevents economy-first bias)
        by_class = {'First Class': [], 'Business': [], 'Economy': []}
        for f in flights:
            flight_class = f.get('class', 'Economy')
            if flight_class in by_class:
                by_class[flight_class].append(f)

        result = [f"Found {len(flights)} flights from {from_city} to {to_city}:"]
        result.append("")

        # Show samples from EACH class (prevents showing only economy)
        for class_name in ['First Class', 'Business', 'Economy']:
            class_flights = by_class[class_name]
            if class_flights:
                result.append(f"{class_name.upper()} ({len(class_flights)} available):")
                # Show up to 3 from each class, sorted by ID within class
                for f in sorted(class_flights, key=lambda x: x['flight_id'])[:3]:
                    result.append(f"  {f['flight_id']}: {f['airline']}, ${f['price_usd']}, "
                                 f"{f['departure_time']}->{f['arrival_time']}, {f['duration_hours']:.1f}h")
                result.append("")
        
        if len(flights) > 15:
            result.append(f"  ... and {len(flights) - 15} more options available")
        
        return "\n".join(result)
    
    def _tool_get_flight_details(self, flight_id: str, **kwargs) -> str:
        """Get flight details. Extra kwargs are ignored."""
        if isinstance(flight_id, list):
            return "\n\n".join(self._tool_get_flight_details(fid) for fid in flight_id[:3])
        
        flights = self.state.get_belief("available_flights", [])
        for f in flights:
            if f['flight_id'] == flight_id:
                return (f"Flight {flight_id}: {f['airline']}, {f['from_city']}→{f['to_city']}, "
                       f"{f['departure_time']}-{f['arrival_time']}, {f['duration_hours']:.2f}h, ${f['price_usd']}")
        return f"Flight {flight_id} not found."
    
    def _tool_compare_flights(self, flight_ids: List[str], criteria: str = "overall", **kwargs) -> str:
        """Compare flights. Extra kwargs are ignored."""
        flights = self.state.get_belief("available_flights", [])
        flight_dict = {f['flight_id']: f for f in flights}
        to_compare = [flight_dict[fid] for fid in flight_ids if fid in flight_dict]

        if not to_compare:
            return "No valid flights to compare."

        if criteria == "price":
            sorted_f = sorted(to_compare, key=lambda x: x['price_usd'])
            return "\n".join(f"{i+1}. {f['flight_id']}: ${f['price_usd']}" for i, f in enumerate(sorted_f))
        elif criteria == "timing":
            sorted_f = sorted(to_compare, key=lambda f: f['departure_time'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f['departure_time']}" for i, f in enumerate(sorted_f))
        else:  # overall - show all details, sorted by ID (no bias)
            sorted_f = sorted(to_compare, key=lambda f: f['flight_id'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f.get('class', 'Economy')}, ${f['price_usd']}, {f['departure_time']}"
                           for i, f in enumerate(sorted_f))

    def _tool_filter_by_price_range(self, min_price: int, max_price: int, **kwargs) -> str:
        """Filter available flights to a specific price range. Extra kwargs are ignored."""
        flights = self.state.get_belief("available_flights", [])
        return self._filter_items_by_price(
            items=flights,
            min_price=min_price,
            max_price=max_price,
            price_key='price_usd',
            item_type='flights',
            group_by_key='class'
        )

    def _format_item_summary(self, item: Dict, price_key: str) -> str:
        """Format flight summary with details."""
        return (f"{item['flight_id']}: {item['airline']}, ${item[price_key]}, "
                f"{item['departure_time']}->{item['arrival_time']} ({item['duration_hours']:.1f}h)")

    def _tool_analyze_flights(self, **kwargs) -> str:
        """Analyze available flights - NO BIAS, just facts."""
        flights = self.state.get_belief("available_flights", [])
        if not flights:
            return "No flights available. Use search_flights first."

        # Group by CLASS (not by price tier to avoid "budget" bias)
        by_class = {}
        for f in flights:
            flight_class = f.get('class', 'Economy')
            if flight_class not in by_class:
                by_class[flight_class] = []
            by_class[flight_class].append(f)

        result = [f"Flight Analysis ({len(flights)} total flights available):"]
        result.append("")

        # Show each class with price range (neutral order: First, Business, Economy)
        for flight_class in ['First Class', 'Business', 'Economy']:
            if flight_class in by_class:
                class_flights = by_class[flight_class]
                prices = [f['price_usd'] for f in class_flights]
                result.append(f"  {flight_class}: {len(class_flights)} options")
                result.append(f"    Price range: ${min(prices)} - ${max(prices)}")
                result.append("")

        result.append("  Compare specific options to make your decision.")

        return "\n".join(result)
    

    def search_flights(self, query: FlightQuery) -> FlightSearchResult:
        """Main entry point for flight search using ReAct reasoning."""
        self.reset_state()

        # Goal prompt for this specific search
        goal = f"""Find the best flight from {query.from_city} to {query.to_city}.

OBJECTIVE: Select the single best flight for this business trip.

CONSTRAINTS:
- Route: {query.from_city} → {query.to_city}
- Budget: {f"${query.max_price}" if query.max_price else "Flexible"}
- Departure window: {query.departure_after or "Flexible"} to {query.departure_before or "Flexible"}

OPTIMIZATION CRITERIA:
- Balance cost efficiency vs. time savings vs. comfort
- Consider business travel context (productivity, schedule impact)
- Evaluate all flight classes (Economy, Business, First) on their merits
- Don't default to cheapest - optimize for trip value

AVAILABLE TOOLS: search_flights, analyze_flights, compare_flights, filter_by_price_range, finish

Use tools in whatever sequence makes sense. When confident in your choice, return:
{{"selected_flights": ["FLIGHT_ID"], "reasoning": "Why this flight best meets the criteria"}}"""
        result = self.run(goal)
        
        # Parse LLM's selection
        selected_ids = []
        llm_reasoning = ""
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    selected_ids = parsed.get("selected_flights", parsed.get("top_flights", []))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    # Robust parsing for various keys the LLM might use
                    selected_ids = (
                        final.get("selected_flights") or 
                        final.get("selected_flight") or
                        final.get("top_flights") or 
                        final.get("selection") or 
                        final.get("flight_id") or
                        []
                    )
                    # Handle single string case (if LLM returns "FL123" instead of ["FL123"])
                    if isinstance(selected_ids, str):
                        selected_ids = [selected_ids]
                        
                    llm_reasoning = final.get("reasoning", str(final))
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
        
        # Get available flights from state
        available = self.state.get_belief("available_flights", [])
        
        if not available:
            # Fallback: search directly
            flights = self.loader.search(from_city=query.from_city, to_city=query.to_city)
            available = flights if flights else []
        
        # TRUST THE LLM'S SELECTION
        if selected_ids:
            flight_dict = {f['flight_id']: f for f in available}

            # Handle both formats: list of strings ["FL0177"] or list of dicts [{"flight_id": "FL0177", ...}]
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract flight ID from strings like "FL0012: Economy, $270, 5.9h, 00:00"
                    # or just "FL0012"
                    if ':' in item:
                        flight_id = item.split(':')[0].strip()
                    else:
                        flight_id = item.strip()
                    normalized_ids.append(flight_id)
                elif isinstance(item, dict) and 'flight_id' in item:
                    normalized_ids.append(item['flight_id'])

            selected_flights = [flight_dict[fid] for fid in normalized_ids if fid in flight_dict]
            
            # If LLM selected valid flights, use them (take only the first one)
            if selected_flights:
                top_flights = [Flight(**f) for f in selected_flights[:1]]  # ONLY 1 flight
                self.log_message("orchestrator", f"LLM selected best flight: {top_flights[0].flight_id}", "result")
                reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
                return FlightSearchResult(query=query, flights=top_flights, reasoning=reasoning)
        
        # FALLBACK: If LLM didn't select valid flights, use first option
        # (This should rarely happen with good prompting)
        print(f"    [FlightAgent] LLM selection failed, using fallback (first available)")

        # NO BIAS - just use first available flight
        fallback_flights = available[:1]  # ONLY 1 flight
        top_flights = [Flight(**f) for f in fallback_flights]

        self.log_message("orchestrator", f"Fallback proposal: 1 flight (first available)", "result")
        reasoning = self._build_reasoning_trace(query, result, llm_reasoning)
        return FlightSearchResult(query=query, flights=top_flights, reasoning=reasoning)
    
    def refine_proposal(self, feedback: Dict[str, Any], previous_flights: List[Dict]) -> FlightSearchResult:
        """
        CNP NEGOTIATION: Refine flight proposal based on PolicyAgent feedback.

        Uses ReAct reasoning to autonomously respond to feedback while being guided
        toward the target price range through smart pre-loading and clear examples.

        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "timing_conflict"
                - target_price_min/max: suggested price range
                - reasoning: PolicyAgent's explanation
                - from_city, to_city: route information
            previous_flights: Flights from previous proposal

        Returns:
            Refined FlightSearchResult with new proposal and reasoning
        """
        self.reset_state()

        issue = feedback.get("issue", "general")
        from_city = feedback.get("from_city", "")
        to_city = feedback.get("to_city", "")
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)

        if self.verbose:
            print(f"    [FlightAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [FlightAgent] Target price range: ${target_min}-${target_max}")

        # SMART PRE-LOADING: Load flights in expanded range to prevent confusion
        # Expanded range gives agent some exploration room while keeping focus
        all_flights = self.loader.search(from_city=from_city, to_city=to_city)
        
        # Calculate expanded range (±50% of target)
        expanded_min = max(0, int(target_min * 0.5))
        expanded_max = int(target_max * 1.5)
        
        # Filter to relevant range
        relevant_flights = [
            f for f in all_flights
            if expanded_min <= f.get('price_usd', 0) <= expanded_max
        ]
        
        # Fallback: if filter is too strict (< 5 flights), use all flights
        if len(relevant_flights) < 5:
            if self.verbose:
                print(f"    [FlightAgent] Expanded range too strict, using all {len(all_flights)} flights")
            relevant_flights = all_flights
        else:
            if self.verbose:
                print(f"    [FlightAgent] Pre-filtered to {len(relevant_flights)} flights in ${expanded_min}-${expanded_max} range")
        
        # Load into state
        self.state.add_belief("available_flights", relevant_flights)

        if not relevant_flights:
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=[],
                reasoning="No flights available for this route."
            )

        # BUILD GOAL PROMPT with clear emphasis on target range
        goal = f"""REFINEMENT TASK: Find a better flight based on orchestrator feedback.

    === FEEDBACK FROM ORCHESTRATOR ===
    Issue: {issue}
    **TARGET PRICE RANGE: ${target_min} - ${target_max}** ← CRITICAL CONSTRAINT
    Reasoning: {feedback.get('reasoning', 'Not specified')}

    === YOUR OBJECTIVE ===
    Select ONE flight in the **${target_min}-${target_max}** range that addresses: {issue}

    === AVAILABLE CONTEXT ===
    - Previous selection: {[f.get('flight_id', 'N/A') for f in previous_flights]}
    - Route: {from_city} → {to_city}
    - Flights in memory: {len(relevant_flights)} options pre-loaded in relevant range
    - **Target budget: ${target_min} - ${target_max}** ← USE THIS EXACT RANGE

    === AVAILABLE TOOLS ===
    - filter_by_price_range(min_price, max_price): Narrow to specific budget
    - analyze_flights(): See distribution by class and price
    - compare_flights(flight_ids): Compare specific options side-by-side
    - finish(result): Submit your final selection

    NOTE: Flights are pre-loaded in a relevant range. Use filter_by_price_range({target_min}, {target_max}) to narrow to exact target.

    === YOUR DECISION PROCESS ===
    Think through the feedback and decide your approach:
    - What is the core issue? (price upgrade? budget reduction? quality shift?)
    - Should I filter to exact target range first? (usually YES)
    - What flight classes will be available in the target range?
    - How many options should I compare before deciding?

    Adapt your approach - simple cases need 2-3 tools, complex cases need 4-5 tools.

    === SUGGESTED APPROACH ===

    Most refinements work best with this pattern:
    1. Filter to exact target: **filter_by_price_range({target_min}, {target_max})**
    2. See what's available: analyze_flights()
    3. Compare top candidates: compare_flights([...])
    4. Submit best option: finish(...)

    You can adapt this if you have good reason, but filtering to target range first is usually most effective.

    === DECISION EXAMPLES (Diverse Approaches) ===

    Example 1: Quality Upgrade (Previous: $380 → **TARGET: $500-$720**)
    Previous selection: Economy FL0005 at $380
    Issue: Budget under-utilized, upgrade to better class

    **First action: filter_by_price_range(500, 720)** ← Uses exact TARGET range
    Observation: "Filtered to 30 flights in $500-$720 range:
    Business (10 options): $560-$720
    Economy (20 options): $500-$580"

    Next action: analyze_flights()
    Observation: "Business: 10 options, $560-$720
    Economy: 20 options, $500-$580"

    Next action: compare_flights(["FL0004", "FL0006", "FL0008"])
    Observation: "FL0004: Business, $560
    FL0006: Business, $630
    FL0008: Business, $720"

    Final action: finish({{"selected_flights": ["FL0004"], "reasoning": "..."}})

    Selected: Business FL0004 at $560
    Reasoning: "Filtered to TARGET range $500-$720. Business class becomes available 
    at this price point. FL0004 at $560 provides significant comfort upgrade for long 
    flight while staying in lower end of target range. Productivity benefits justify 
    the premium over Economy options."

    Trade-offs: ✓ Quality upgrade achieved ✗ Higher cost than previous Economy


    Example 2: Budget Reduction (Previous: $800 → **TARGET: $200-$400**)
    Previous selection: Business FL0088 at $800
    Issue: Over budget, need to reduce cost

    **First action: filter_by_price_range(200, 400)** ← Uses exact TARGET range
    Observation: "Filtered to 25 flights in $200-$400 range:
    Economy (25 options): $260-$410"

    Next action: analyze_flights()
    Observation: "Economy: 25 options, $260-$410
    (No Business or First class in this range)"

    Next action: compare_flights(["FL0001", "FL0017", "FL0007"])
    Observation: "FL0001: Economy, $280, 04:15 departure
    FL0017: Economy, $280, 06:30 departure
    FL0007: Economy, $260, 00:45 departure"

    Final action: finish({{"selected_flights": ["FL0017"], "reasoning": "..."}})

    Selected: Economy FL0017 at $280
    Reasoning: "Filtered to TARGET range $200-$400. All options are Economy class. 
    FL0017 offers best departure time (06:30) at competitive price. Avoided ultra-cheap 
    FL0007 ($260) due to impractical 00:45 departure. The $20 premium for reasonable 
    timing is justified for business travel."

    Trade-offs: ✓ Budget compliance ✗ Downgrade from Business to Economy


    Example 3: Balanced Upgrade (Previous: $410 → **TARGET: $800-$1200**)
    Previous selection: Economy FL0002 at $410
    Issue: Significant budget available, seek quality improvement

    **First action: filter_by_price_range(800, 1200)** ← Uses exact TARGET range
    Observation: "Filtered to 15 flights in $800-$1200 range:
    First Class (5 options): $1200-$1270
    Business (10 options): $800-$1030"

    Next action: analyze_flights()
    Observation: "First Class: 5 options, $1200-$1270
    Business: 10 options, $800-$1030"

    Next action: compare_flights(["FL0006", "FL0088", "FL0014"])
    Observation: "FL0006: Business, $1030, 6.4h
    FL0088: Business, $800, 6.5h
    FL0014: First Class, $1270, 6.1h"

    Final action: finish({{"selected_flights": ["FL0006"], "reasoning": "..."}})

    Selected: Business FL0006 at $1030
    Reasoning: "Filtered to TARGET range $800-$1200. Compared Business vs First class. 
    Business class provides 90% of First class benefits (comfort, productivity) at 
    significantly lower cost. For 6-hour flight, Business offers sufficient comfort. 
    First class premium ($240 more) not justified for this trip duration."

    Trade-offs: ✓ Value optimization ✗ Not maximum possible quality

    === KEY INSIGHTS FROM EXAMPLES ===

    1. **All three examples start with filter_by_price_range({target_min}, {target_max})**
    - This is not a rigid rule, but it's the most effective pattern
    - Ensures you only see options in the target range
    - Prevents confusion about which prices to consider

    2. **Different complexity levels** (3, 4, 4 iterations)
    - Simple case (all same class): filter → analyze → finish
    - Complex case (multiple classes): filter → analyze → compare → finish
    - Adapt to situation complexity

    3. **Clear reasoning about trade-offs**
    - Always explain WHY you selected this option
    - What did you optimize for?
    - What did you sacrifice?

    === YOUR DECISION ===
    Address the orchestrator's feedback by finding the best flight in **${target_min}-${target_max}** range.

    Use tools in whatever sequence makes sense, but strongly consider filtering to target range first.

    Return when confident: {{"selected_flights": ["SINGLE_FLIGHT_ID"], "reasoning": "how this addresses feedback"}}"""

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
                    selected_ids = parsed.get("selected_flights", parsed.get("top_flights", []))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    # Already a dict
                    selected_ids = final.get("selected_flights", final.get("top_flights", []))
                    llm_reasoning = final.get("reasoning", str(final))
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
                if self.verbose:
                    print(f"    [FlightAgent] Failed to parse LLM response: {e}")
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
            if self.verbose:
                print(f"    [FlightAgent] ReAct loop failed: {result.get('error')}")

        # Get current available flights from state (may have been filtered by agent)
        available = self.state.get_belief("available_flights", relevant_flights)

        # TRUST THE LLM'S SELECTION
        if selected_ids:
            flight_dict = {f['flight_id']: f for f in available}

            # Normalize IDs - handle multiple formats
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract ID from strings like "FL0012: Economy, $270..." or just "FL0012"
                    if ':' in item:
                        flight_id = item.split(':')[0].strip()
                    else:
                        flight_id = item.strip()
                    normalized_ids.append(flight_id)
                elif isinstance(item, dict) and 'flight_id' in item:
                    normalized_ids.append(item['flight_id'])

            selected_flights = [flight_dict[fid] for fid in normalized_ids if fid in flight_dict]

            if selected_flights:
                # Take only the first flight (CNP expects single proposal)
                refined_flights = [Flight(**f) for f in selected_flights[:1]]

                if self.verbose:
                    flight = refined_flights[0]
                    print(f"    [FlightAgent] ✓ Selected {flight.flight_id}: {flight.flight_class} ${flight.price_usd}")
                    print(f"    [FlightAgent] Reasoning: {llm_reasoning[:100]}{'...' if len(llm_reasoning) > 100 else ''}")

                self.log_message("policy_agent", f"Refined: {refined_flights[0].flight_id} - {llm_reasoning[:100]}", "negotiation")

                return FlightSearchResult(
                    query=FlightQuery(from_city=from_city, to_city=to_city),
                    flights=refined_flights,
                    reasoning=llm_reasoning
                )

        # FALLBACK: If LLM didn't select valid flights, use first available
        if self.verbose:
            print(f"    [FlightAgent] ⚠ LLM selection failed, using fallback (first available)")

        fallback_flights = [Flight(**f) for f in available[:1]]
        
        return FlightSearchResult(
            query=FlightQuery(from_city=from_city, to_city=to_city),
            flights=fallback_flights,
            reasoning=f"Fallback: LLM did not select valid flight, returning first available option."
        )
    
    def _build_reasoning_trace(self, query: FlightQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Flight Search ReAct Trace",
                f"**Route**: {query.from_city} → {query.to_city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
