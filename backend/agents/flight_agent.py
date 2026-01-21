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
        return """You are a Flight Booking Specialist. Select the best flight for this trip.

WORKFLOW:
1. search_flights - Load all options (call ONCE)
2. analyze_flights - See class distribution and price ranges
3. compare_flights - Compare 2-4 specific candidates from DIFFERENT classes
4. finish - Return your selection

CRITICAL - NO CLASS BIAS:
- DO NOT assume economy is best for business travel
- Business class offers productivity, comfort, and time efficiency
- First class may be appropriate for executive travel
- Consider ALL classes based on trip value, not just price
- Compare options from multiple classes before deciding

IMPORTANT: Select exactly ONE flight only - not multiple options!

Return format: {"selected_flights": ["SINGLE_FLIGHT_ID"], "reasoning": "How this ONE flight addresses feedback"}"""
    
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

STEPS:
1. search_flights() - Load options
2. analyze_flights() - See ALL classes (Economy, Business, First)
3. compare_flights() - Compare options from DIFFERENT classes (e.g., one economy, one business)
4. finish() - Return your choice

DO NOT default to economy. Consider business/first class benefits: comfort, productivity, time.
Evaluate based on overall trip value, not just lowest price.

Return: {{"selected_flights": ["FLIGHT_ID"], "reasoning": "Brief explanation"}}"""
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

        TRULY AGENTIC: Uses ReAct reasoning loop to autonomously decide how to
        respond to feedback, using available tools including filter_by_price_range.

        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "timing_conflict"
                - target_price_min/max: suggested price range
                - reasoning: PolicyAgent's explanation
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

        # Load ALL flights into state (no pre-filtering)
        all_flights = self.loader.search(from_city=from_city, to_city=to_city)
        self.state.add_belief("available_flights", all_flights)

        if not all_flights:
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=[],
                reasoning="No flights available for this route."
            )

        # Create goal for ReAct reasoning - agent decides how to use its tools
        goal = f"""TASK: Refine your flight proposal based on orchestrator feedback.

=== FEEDBACK FROM ORCHESTRATOR ===
Issue: {issue}
Reasoning: {feedback.get('reasoning', 'No details provided')}
Target Price Range: ${target_min} to ${target_max}

=== CURRENT SITUATION ===
✓ {len(all_flights)} flights are ALREADY LOADED in memory
✓ You do NOT need to search again
✓ Your job: Filter → Analyze → Compare → Select ONE flight

=== YOUR AVAILABLE TOOLS ===
1. filter_by_price_range(min_price=X, max_price=Y)
2. analyze_flights()
3. compare_flights(flight_ids=["FL0001", "FL0002", ...])
4. finish(result={{"selected_flights": ["FL_ID"], "reasoning": "..."}})

⚠️ TOOLS THAT DO NOT EXIST (DO NOT USE):
- search_flights (already done!)
- filter_flights (doesn't exist!)
- search (doesn't exist!)

=== STEP-BY-STEP INSTRUCTIONS ===

STEP 1: Filter to target price range
   Tool: filter_by_price_range(min_price={target_min}, max_price={target_max})
   Why: Narrow down to flights in ${target_min}-${target_max} range
   
STEP 2: Analyze filtered results
   Tool: analyze_flights()
   Why: See what flight classes are available in your filtered range
   
STEP 3: Compare specific options
   Tool: compare_flights(flight_ids=["FL0001", "FL0004", "FL0014"])
   Why: Compare 2-4 flights from DIFFERENT classes
   Note: Use actual flight IDs from the analyze results!
   
STEP 4: Select the SINGLE best flight
   Tool: finish(result={{"selected_flights": ["SINGLE_FLIGHT_ID"], "reasoning": "..."}})
   Why: Return exactly ONE flight that best addresses the feedback
   
=== CRITICAL RULES ===
1. DO NOT call search_flights - flights are already loaded!
2. DO NOT invent tools like "filter_flights" or "search"
3. MUST call filter_by_price_range FIRST
4. Return exactly ONE flight ID, not multiple
5. Consider ALL classes in target range (Economy, Business, First)
6. Don't assume economy is best - consider value and quality

=== EXAMPLE WORKFLOW ===
Iteration 1: filter_by_price_range(min_price={target_min}, max_price={target_max})
Iteration 2: analyze_flights()
Iteration 3: compare_flights(flight_ids=["FL0004", "FL0014"])
Iteration 4: finish(result={{"selected_flights": ["FL0004"], "reasoning": "Business class offers best value"}})

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
                    selected_ids = parsed.get("selected_flights", parsed.get("top_flights", []))
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    selected_ids = final.get("selected_flights", final.get("top_flights", []))
                    llm_reasoning = final.get("reasoning", str(final))
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"

        # Get current available flights from state (may have been filtered by agent)
        available = self.state.get_belief("available_flights", all_flights)

        # TRUST THE LLM'S SELECTION
        if selected_ids:
            flight_dict = {f['flight_id']: f for f in available}

            # Normalize IDs - handle both raw IDs and description strings
            normalized_ids = []
            for item in selected_ids:
                if isinstance(item, str):
                    # Extract ID from "FL0012: Economy, $270..." or just "FL0012"
                    if ':' in item:
                        flight_id = item.split(':')[0].strip()
                    else:
                        flight_id = item.strip()
                    normalized_ids.append(flight_id)
                else:
                    normalized_ids.append(item)

            selected_flights = [flight_dict[fid] for fid in normalized_ids if fid in flight_dict]

            if selected_flights:
                refined_flights = [Flight(**f) for f in selected_flights[:1]]  # ONLY 1 flight

                if self.verbose:
                    prices = [f.price_usd for f in refined_flights]
                    print(f"    [FlightAgent] Selected {len(refined_flights)} flights (${min(prices)}-${max(prices)})")
                    print(f"    [FlightAgent] Reasoning: {llm_reasoning[:80]}...")

                self.log_message("policy_agent", f"Refined: {len(refined_flights)} flights - {llm_reasoning[:100]}", "negotiation")

                return FlightSearchResult(
                    query=FlightQuery(from_city=from_city, to_city=to_city),
                    flights=refined_flights,
                    reasoning=llm_reasoning
                )

        # FALLBACK: If LLM didn't select valid flights, use first option (no bias)
        if self.verbose:
            print(f"    [FlightAgent] LLM selection failed, using fallback (first available)")

        fallback_flights = [Flight(**f) for f in available[:1]]
        return FlightSearchResult(
            query=FlightQuery(from_city=from_city, to_city=to_city),
            flights=fallback_flights,
            reasoning=f"Fallback: LLM did not select, using first available option."
        )
    
    def _build_reasoning_trace(self, query: FlightQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Flight Search ReAct Trace",
                f"**Route**: {query.from_city} → {query.to_city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
