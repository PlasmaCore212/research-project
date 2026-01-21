# backend/agents/flight_agent.py
"""Flight Agent with ReAct Pattern for business travel flight search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import FlightQuery, FlightSearchResult, Flight
from data.loaders import FlightDataLoader
from typing import List, Dict, Any, Optional
import json


class FlightAgent(BaseReActAgent):
    """Agentic Flight Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "qwen2.5:14b", verbose: bool = True):
        super().__init__(
            agent_name="FlightAgent", agent_role="Flight Booking Specialist",
            model_name=model_name, max_iterations=10, verbose=verbose
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
                description="Search for flights between cities with optional filters",
                parameters={"from_city": "str", "to_city": "str", "max_price": "int (optional)",
                           "departure_after": "str (optional)", "departure_before": "str (optional)"},
                function=self._tool_search_flights
            ),
            "get_flight_details": AgentAction(
                name="get_flight_details",
                description="Get detailed information about a specific flight",
                parameters={"flight_id": "str"},
                function=self._tool_get_flight_details
            ),
            "compare_flights": AgentAction(
                name="compare_flights",
                description="Compare multiple flights on specific criteria. REQUIRES: flight_ids as a list of strings (e.g. ['FL0001', 'FL0002']).",
                parameters={"flight_ids": "list[str] REQUIRED - list of flight IDs to compare",
                           "criteria": "str (optional) - 'price', 'duration', 'timing', or 'overall'"},
                function=self._tool_compare_flights
            ),
            "analyze_options": AgentAction(
                name="analyze_options",
                description="Analyze all available flight options by price tiers. Use after search_flights.",
                parameters={},
                function=self._tool_analyze_options
            ),
            "analyze_price_range": AgentAction(
                name="analyze_price_range",
                description="Analyze the price distribution of available flights",
                parameters={"from_city": "str", "to_city": "str"},
                function=self._tool_analyze_price_range
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Flight Booking Specialist. Find the single best flight for this trip.

AVAILABLE TOOLS:
1. search_flights(from_city, to_city) - Search for all available flights
2. analyze_options() - Get overview by flight class
3. compare_flights(flight_ids=[...]) - Compare specific flights
4. get_flight_details(flight_id) - Get detailed info about one flight
5. finish(result={...}) - Submit your final selection

Note: search_flights returns all flights. You cannot filter by time in search_flights.
To filter, call search_flights once, then use compare_flights with specific flight IDs.

YOUR TASK:
Find ONE flight that offers good value for the trip.
You don't know the budget - use your judgment to balance cost, quality, and convenience.

Consider all available options across different classes and price points:
- Economy, Business, and First Class are all valid choices
- Cheaper is not always better, more expensive is not always better
- Consider timing, duration, comfort, and price together
- Make your own decision based on overall value

Use your tools effectively:
- search_flights to see what's available
- analyze_options to understand the range
- compare_flights to evaluate specific options
- finish when you've selected the best option

Return format: {"selected_flights": ["FLIGHT_ID"], "reasoning": "Why this is the best choice"}"""
    
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
        
        # Show flights without bias - don't separate by class
        result = [f"Found {len(flights)} flights from {from_city} to {to_city}:"]
        
        # Sort by price to show range, don't group by class
        sorted_flights = sorted(flights, key=lambda x: x['price_usd'])
        for f in sorted_flights[:15]:  # Show first 15
            result.append(f"  {f['flight_id']}: {f['airline']} {f.get('class', 'Economy')}, ${f['price_usd']}, "
                         f"{f['departure_time']}->{f['arrival_time']}, {f['duration_hours']:.1f}h")
        
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
        elif criteria == "duration":
            sorted_f = sorted(to_compare, key=lambda x: x['duration_hours'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f['duration_hours']:.1f}h" for i, f in enumerate(sorted_f))
        elif criteria == "timing":
            sorted_f = sorted(to_compare, key=lambda f: f['departure_time'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f['departure_time']}" for i, f in enumerate(sorted_f))
        else:  # overall - show all details, sorted by ID (no bias)
            sorted_f = sorted(to_compare, key=lambda f: f['flight_id'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f.get('class', 'Economy')}, ${f['price_usd']}, {f['duration_hours']:.1f}h, {f['departure_time']}"
                           for i, f in enumerate(sorted_f))
    
    def _tool_analyze_options(self, **kwargs) -> str:
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
                durations = [f.get('duration_hours', 0) for f in class_flights]
                result.append(f"  {flight_class}: {len(class_flights)} options")
                result.append(f"    Price range: ${min(prices)} - ${max(prices)}")
                result.append(f"    Duration range: {min(durations):.1f}h - {max(durations):.1f}h")
                result.append("")

        result.append("  Compare specific options to make your decision.")

        return "\n".join(result)
    
    def _tool_analyze_price_range(self, from_city: str, to_city: str, **kwargs) -> str:
        flights = self.loader.search(from_city=from_city, to_city=to_city)
        if not flights:
            return f"No flights found for {from_city} → {to_city}"
        prices = [f['price_usd'] for f in flights]
        avg = sum(prices) / len(prices)
        return f"Price Analysis {from_city}→{to_city}: Min ${min(prices)}, Max ${max(prices)}, Avg ${avg:.0f}, {len(flights)} flights"
    
    def search_flights(self, query: FlightQuery) -> FlightSearchResult:
        """Main entry point for flight search using ReAct reasoning."""
        self.reset_state()

        # Goal prompt for this specific search
        goal = f"""Find the best flight from {query.from_city} to {query.to_city}.

YOUR TASK:
Explore available options and select ONE flight that offers good overall value.
Consider: comfort, timing, convenience, and price.

All flight classes (Economy, Business, First) are valid choices - use your judgment.

Use your tools to explore before deciding:
- search_flights() to see what's available
- analyze_options() to understand the range
- compare_flights() to evaluate specific options
- finish() when you've decided on the best option

Return: {{"selected_flights": ["FLIGHT_ID"], "reasoning": "Explain your decision"}}"""
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
                    normalized_ids.append(item)
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
        
        TRULY AGENTIC: Uses LLM reasoning to decide how to respond to feedback,
        considering trade-offs and making autonomous decisions about which
        options best address the PolicyAgent's concerns.
        
        Args:
            feedback: Dict with keys like:
                - issue: "budget_exceeded" | "quality_upgrade" | "timing_conflict"
                - target_price_min/max: suggested price range (for upgrade)
                - max_price: suggested max price (if budget issue)
                - reasoning: PolicyAgent's explanation
            previous_flights: Flights from previous proposal
        
        Returns:
            Refined FlightSearchResult with new proposal and reasoning
        """
        issue = feedback.get("issue", "general")
        from_city = feedback.get("from_city", "")
        to_city = feedback.get("to_city", "")
        
        # Get PolicyAgent's target price constraints
        target_min = feedback.get("target_price_min", 0)
        target_max = feedback.get("target_price_max", 9999)
        should_re_search = feedback.get("re_search", False)
        
        if self.verbose:
            print(f"    [FlightAgent] Reasoning about feedback: {issue}")
            if target_min > 0 or target_max < 9999:
                print(f"    [FlightAgent] Target price range: ${target_min}-${target_max}")
        
        # RE-SEARCH with PolicyAgent's constraints
        # This is the key enhancement - agents actually use their tools again
        if should_re_search:
            if self.verbose:
                print(f"    [FlightAgent] Re-searching ALL flights with price filter...")
            all_flights = self.loader.search(from_city=from_city, to_city=to_city)
            
            # Filter to target price range
            if target_min > 0 or target_max < 9999:
                filtered = [f for f in all_flights 
                           if target_min <= f.get('price_usd', 0) <= target_max]
                if filtered:
                    available = filtered
                    if self.verbose:
                        print(f"    [FlightAgent] Found {len(available)} flights in ${target_min}-${target_max} range")
                else:
                    # No flights in exact range, get closest
                    available = all_flights
                    if self.verbose:
                        print(f"    [FlightAgent] No flights in exact range, using all {len(available)} flights")
            else:
                available = all_flights
        else:
            # Fallback: use cached
            all_flights = self.loader.search(from_city=from_city, to_city=to_city)
            available = all_flights if all_flights else []
        
        self.state.add_belief("available_flights", available)
        
        if not available:
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=[],
                reasoning="No flights available for this route."
            )
        
        # Build context for LLM reasoning
        # NO BIAS - show flights in neutral order (by price for consistency)
        # The LLM will decide which one to pick based on the feedback
        sorted_available = sorted(available, key=lambda x: x.get('price_usd', 0))

        flight_summary = []
        for f in sorted_available[:20]:  # Show 20 options
            flight_summary.append(
                f"{f['flight_id']}: {f['airline']} {f.get('class', 'Economy')} "
                f"${f['price_usd']} {f['departure_time']}->{f['arrival_time']} ({f['duration_hours']:.1f}h)"
            )
        
        # LLM prompt for agentic reasoning - SINGLE OPTION ONLY
        prompt = f"""You are a Flight Booking Specialist agent. The PolicyAgent has rejected your proposal.

FEEDBACK FROM POLICY AGENT:
- Issue: {issue}
- Reasoning: {feedback.get('reasoning', 'No details provided')}
- Target Price Range: ${feedback.get('target_price_min', 100)}-${feedback.get('target_price_max', 1000)}

AVAILABLE FLIGHTS ({from_city} → {to_city}):
{chr(10).join(flight_summary[:10])}

YOUR TASK:
Analyze the feedback and target price range, then select EXACTLY ONE flight that best addresses the PolicyAgent's concerns.
Consider the trade-offs between price, flight class, duration, and departure time.

Return JSON with EXACTLY ONE flight ID:
{{"selected_flight": "FL0XXX", "reasoning": "Brief explanation of why this flight best addresses the feedback"}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            # Get SINGLE flight - check both singular and plural keys
            selected_id = result.get("selected_flight") or (result.get("selected_flights", []) or [None])[0]
            reasoning = result.get("reasoning", "Selected based on feedback analysis")
            
            # Get the selected flight - ONLY ONE
            selected = [f for f in available if f['flight_id'] == selected_id][:1]
            
            # Fallback: if LLM didn't select valid flights, use first option (no bias)
            if not selected:
                if self.verbose:
                    print(f"    [FlightAgent] LLM selection empty, using fallback (first available)")
                selected = available[:1]
                reasoning = f"Fallback: LLM did not select, using first available option."
            
            # CRITICAL: Only return 1 flight
            refined_flights = [Flight(**f) for f in selected[:1]]
            
            if self.verbose:
                if refined_flights:
                    prices = [f.price_usd for f in refined_flights]
                    print(f"    [FlightAgent] Selected {len(refined_flights)} flights (${min(prices)}-${max(prices)})")
                    print(f"    [FlightAgent] Reasoning: {reasoning[:80]}...")
            
            self.log_message("policy_agent", f"Refined: {len(refined_flights)} flights - {reasoning[:100]}", "negotiation")
            
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=refined_flights,
                reasoning=reasoning
            )
            
        except Exception as e:
            if self.verbose:
                print(f"    [FlightAgent] LLM error: {e}, using fallback")

            # Fallback: first available (no bias)
            selected = available[:1]

            refined_flights = [Flight(**f) for f in selected]
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=refined_flights,
                reasoning=f"Fallback: LLM error, using first available option."
            )
    
    def _build_reasoning_trace(self, query: FlightQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Flight Search ReAct Trace",
                f"**Route**: {query.from_city} → {query.to_city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
