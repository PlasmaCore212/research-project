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
        """Extract SINGLE item when agent fails to call finish() properly."""
        items_key = "available_flights" if "Flight" in self.agent_name else "available_hotels"
        id_key = "flight_id" if "Flight" in self.agent_name else "hotel_id"
        result_key = "selected_flights" if "Flight" in self.agent_name else "selected_hotels"
        
        items = self.state.get_belief(items_key, [])
        if not items:
            return {"result": f"No {items_key} found"}
        
        # Return in expected format
        return {
            result_key: [items[0][id_key]],
            "reasoning": "Fallback: Agent reached max iterations without calling finish(). Using first available option."
        }
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        """Register all tools (for backward compatibility)."""
        return self._register_negotiation_tools()
    
    def _register_search_tools(self) -> Dict[str, AgentAction]:
        """Register tools for INITIAL SEARCH only (no price filtering)."""
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
                description="Compare 2-5 specific flights by their IDs. REQUIRED: You MUST provide flight_ids parameter as a list of strings like ['FL0001', 'FL0002', 'FL0003']. These IDs come from your search_flights results.",
                parameters={"flight_ids": "list[str] REQUIRED - You MUST provide a list of flight IDs from search results, like ['FL0001', 'FL0002', 'FL0003']. This is NOT optional.",
                           "criteria": "str (optional) - 'price', 'timing', or 'overall' (default)"},
                function=self._tool_compare_flights
            )
        }
    
    def _register_negotiation_tools(self) -> Dict[str, AgentAction]:
        """Register tools for NEGOTIATION (includes price filtering)."""
        tools = self._register_search_tools()
        tools["filter_by_price_range"] = AgentAction(
            name="filter_by_price_range",
            description="Filter available flights to a specific price range. Use this during negotiation when you have a target budget from the orchestrator.",
            parameters={"min_price": "int REQUIRED - minimum price", "max_price": "int REQUIRED - maximum price"},
            function=self._tool_filter_by_price_range
        )
        return tools
    
    def _get_system_prompt(self) -> str:
        return """You are a Flight Booking Specialist. Your job: Select ONE best flight.

=== YOUR ONLY AVAILABLE TOOLS ===
You can ONLY use these exact tool names. Any other tool name will error.

1. search_flights
   Parameters: from_city (required), to_city (required), departure_after (optional), departure_before (optional)
   Returns all available flights. Call ONCE at start.
   Example: {"action": "search_flights", "action_input": {"from_city": "NYC", "to_city": "SF"}}

2. analyze_flights
   Parameters: none
   Shows flight distribution by class (Economy/Business/First). Use after search.
   Example: {"action": "analyze_flights", "action_input": {}}

3. compare_flights
   Parameters: flight_ids (required - list of 2+ flight IDs like ["FL0001", "FL0002"])
   Compares specific flights. IDs come from search_flights results.
   Example: {"action": "compare_flights", "action_input": {"flight_ids": ["FL0001", "FL0004"]}}

4. filter_by_price_range (only available during negotiation)
   Parameters: min_price (required), max_price (required)
   Narrows flights to price range.
   Example: {"action": "filter_by_price_range", "action_input": {"min_price": 300, "max_price": 600}}

5. finish
   Parameters: result (required - dict with "selected_flights" list and "reasoning" string)
   Submit your final selection. Call this when you've decided.
   Example: {"action": "finish", "action_input": {"result": {"selected_flights": ["FL0004"], "reasoning": "Business class at $860 balances cost and comfort"}}}

=== WORKFLOW ===
Step 1: search_flights → Get all options
Step 2: analyze_flights → See distribution
Step 3: compare_flights → Compare top candidates
Step 4: finish → Return your selection

You should finish in 3-5 iterations. Don't overthink it.

=== DECISION PRINCIPLES ===
- Balance cost vs. comfort vs. timing
- Consider business travel needs (productivity, schedule)
- Don't default to cheapest - optimize for overall trip value

=== WHEN TO CALL FINISH ===
Call finish() when you can answer: "Why is this flight the best option?"
If you have a clear preference with reasoning, finish immediately.
DON'T wait until max iterations - decide and finish early.

=== CRITICAL RULES ===
✓ Return ONE flight only in selected_flights list
✓ Use exact tool names from list above
✗ Don't call same tool with same parameters twice
✗ Don't hallucinate tool names like "compare" or "get_details"

FINISH FORMAT: {"selected_flights": ["FL0XXX"], "reasoning": "why this is best"}"""
    
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

    def _validate_tool_call(self, action: str, params: Dict) -> Optional[str]:
        """Validate tool calls before execution. Returns error message if invalid."""
        
        if action == "compare_flights":
            flight_ids = params.get("flight_ids", [])
            if not flight_ids or not isinstance(flight_ids, list):
                return "ERROR: compare_flights requires 'flight_ids' parameter as a list, e.g. flight_ids=['FL0001', 'FL0002']"
            if len(flight_ids) < 2:
                return "ERROR: compare_flights needs at least 2 flight IDs to compare"
        
        return None  # Valid

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
        
        # Use search-only tools (no price filtering during initial search)
        original_tools = self.tools
        self.tools = self._register_search_tools()

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

AVAILABLE TOOLS: search_flights, analyze_flights, compare_flights, finish

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
        
        # Restore original tools
        self.tools = original_tools
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
        
        # Use negotiation tools (includes price filtering)
        original_tools = self.tools
        self.tools = self._register_negotiation_tools()

        issue = feedback.get("issue", "general")
        from_city = feedback.get("from_city", "")
        to_city = feedback.get("to_city", "")
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)

        if self.verbose:
            print(f"    [FlightAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [FlightAgent] Target price range: ${target_min}-${target_max}")

        # Load ALL flights - let agent use filter_by_price_range tool
        all_flights = self.loader.search(from_city=from_city, to_city=to_city)
        
        if not all_flights:
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=[],
                reasoning="No flights available for this route."
            )
        
        # Load into state - NO pre-filtering
        self.state.add_belief("available_flights", all_flights)
        
        if self.verbose:
            print(f"    [FlightAgent] Loaded {len(all_flights)} flights for agent to filter")

        # BUILD GOAL PROMPT - Keep it short and directive
        goal = f"""NEGOTIATION TASK: Refine flight selection based on feedback.

CONTEXT:
- Issue: {issue}
- Target price: ${target_min}-${target_max}
- Route: {from_city} → {to_city}
- Previous: {[f.get('flight_id', 'N/A') for f in previous_flights]}
- Flights loaded: {len(all_flights)} options ALREADY IN MEMORY

YOUR JOB: Select ONE flight in ${target_min}-${target_max} range.

STEP-BY-STEP WORKFLOW (DO THIS EXACTLY):
1. {{action: filter_by_price_range, action_input: {{min_price: {target_min}, max_price: {target_max}}}}}
2. {{action: compare_flights, action_input: {{flight_ids: ["FL0XXX", "FL0YYY"]}}}} (pick 2-3 IDs from filter results)
3. {{action: finish, action_input: {{result: {{selected_flights: ["FL0XXX"], reasoning: "..."}}}}}}

CRITICAL RULES:
- DO NOT call search_flights (flights already loaded)
- DO NOT make up tool names (only use: filter_by_price_range, compare_flights, analyze_flights, finish)
- DO return exactly ONE flight in finish()
- Finish in 3-4 iterations maximum

AVAILABLE TOOLS: filter_by_price_range, analyze_flights, compare_flights, finish"""

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
        available = self.state.get_belief("available_flights", all_flights)

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

                # Restore original tools
                self.tools = original_tools
                return FlightSearchResult(
                    query=FlightQuery(from_city=from_city, to_city=to_city),
                    flights=refined_flights,
                    reasoning=llm_reasoning
                )

        # FALLBACK: If LLM didn't select valid flights, use first available
        if self.verbose:
            print(f"    [FlightAgent] ⚠ LLM selection failed, using fallback (first available)")

        fallback_flights = [Flight(**f) for f in available[:1]]
        
        # Restore original tools
        self.tools = original_tools
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
