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
        """Extract diverse flight options across price tiers for PolicyAgent."""
        flights = self.state.get_belief("available_flights", [])
        if not flights:
            return {"result": "No flights found"}
        
        # Sort and categorize by price tier
        sorted_flights = sorted(flights, key=lambda f: f.get('price_usd', 999))
        prices = [f.get('price_usd', 0) for f in sorted_flights]
        min_p, max_p = min(prices), max(prices)
        price_range = max_p - min_p if max_p > min_p else 100
        
        def get_tier(price):
            if price <= min_p + price_range * 0.33: return 'budget'
            elif price <= min_p + price_range * 0.66: return 'mid'
            return 'premium'
        
        tiers = {'budget': [], 'mid': [], 'premium': []}
        for f in flights:
            tiers[get_tier(f.get('price_usd', 0))].append(f)
        
        # No bias - just sort by price within each tier
        for tier in tiers:
            tiers[tier].sort(key=lambda f: f.get('price_usd', 0))
        
        # Build diverse selection
        selected = []
        for tier in ['budget', 'mid', 'premium']:
            if tiers[tier]:
                selected.append(tiers[tier][0])
        for tier in ['budget', 'mid', 'premium']:
            for f in tiers[tier]:
                if f not in selected and len(selected) < 5:
                    selected.append(f)
        
        return {"top_3_flights": [f['flight_id'] for f in selected[:5]],
                "reasoning": "Diverse selection across price tiers for budget optimization."}
    
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
        return """You are a Flight Booking Specialist finding the best business travel flights.

AVAILABLE TOOLS (ONLY these exist):
• search_flights(from_city, to_city) - Get all available flights. Call this FIRST.
• compare_flights(flight_ids=["FL001","FL002"]) - Compare specific flights by ID
• get_flight_details(flight_id="FL001") - Get detailed info about ONE flight
• analyze_options() - Get price tier summary (Budget/Mid-Range/Premium)
• finish(result) - Return your final recommendations

GOAL: Find diverse flight options across price tiers and classes for business travelers.
- Include Budget (cheapest), Mid-Range, and Premium options
- Include Business/First class flights if available
- Consider flight timing and duration for business convenience

⚠️ CRITICAL RULES:
1. ONLY use tools from the list above - NO OTHER TOOLS EXIST
2. Do NOT make up tools like 'filter_flights', 'sort_flights', 'check_seat_availability'
3. If you need to filter, do it MENTALLY from search results
4. get_flight_details requires flight_id="FL001" format (a string)
5. compare_flights requires flight_ids=["FL001","FL002"] format (a list)

WORKFLOW:
1. search_flights(from_city, to_city) → Get all options
2. analyze_options() or compare_flights() → Understand the options
3. finish(result) → Return recommendations"""
    
    def _tool_search_flights(self, from_city: str, to_city: str, max_price: Optional[int] = None,
                             departure_after: Optional[str] = None, departure_before: Optional[str] = None,
                             **kwargs) -> str:
        """Search flights. Extra kwargs are ignored to handle LLM parameter variations."""
        flights = self.loader.search(from_city=from_city, to_city=to_city, max_price=max_price,
                                     departure_after=departure_after, departure_before=departure_before)
        if not flights:
            return f"No flights found from {from_city} to {to_city} matching criteria."
        
        self.state.add_belief("last_search_from", from_city)
        self.state.add_belief("last_search_to", to_city)
        self.state.add_belief("available_flights", flights)
        self.state.add_belief("flight_count", len(flights))
        
        result = [f"Found {len(flights)} flights from {from_city} to {to_city}:"]
        for f in flights[:10]:
            result.append(f"  - {f['flight_id']}: {f['airline']}, ${f['price_usd']}, "
                         f"{f['departure_time']}→{f['arrival_time']}, {f['duration_hours']:.1f}h")
        if len(flights) > 10:
            result.append(f"  ... and {len(flights) - 10} more")
        
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
            # No morning preference - just sort by departure time
            sorted_f = sorted(to_compare, key=lambda f: f['departure_time'])
            return "\n".join(f"{i+1}. {f['flight_id']}: {f['departure_time']}" for i, f in enumerate(sorted_f))
        else:  # overall - sort by price only, no other bias
            sorted_f = sorted(to_compare, key=lambda f: f['price_usd'])
            return "\n".join(f"{i+1}. {f['flight_id']}: ${f['price_usd']}, {f['duration_hours']:.1f}h, {f['departure_time']}" 
                           for i, f in enumerate(sorted_f))
    
    def _tool_analyze_options(self, **kwargs) -> str:
        """Analyze available flights by price tier and business-friendliness."""
        flights = self.state.get_belief("available_flights", [])
        if not flights:
            return "No flights available. Use search_flights first."
        
        # Categorize by price tier
        prices = [f['price_usd'] for f in flights]
        min_p, max_p = min(prices), max(prices)
        price_range = max_p - min_p if max_p > min_p else 100
        
        def get_tier(price):
            if price <= min_p + price_range * 0.33: return 'Budget'
            elif price <= min_p + price_range * 0.66: return 'Mid-Range'
            return 'Premium'
        
        tiers = {'Budget': [], 'Mid-Range': [], 'Premium': []}
        for f in flights:
            tier = get_tier(f['price_usd'])
            tiers[tier].append(f)
        
        result = [f"Flight Analysis ({len(flights)} total):"]
        for tier, tier_flights in tiers.items():
            if tier_flights:
                tier_prices = [f['price_usd'] for f in tier_flights]
                result.append(f"  {tier}: {len(tier_flights)} flights, ${min(tier_prices)}-${max(tier_prices)}")
        
        result.append(f"  Recommended: Compare top options from each tier for best value.")
        
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
        
        goal = f"""Find flights for business trip: {query.from_city} to {query.to_city}
Max price: ${query.max_price if query.max_price else 'No limit'}

STEPS:
1. search_flights (required params: from_city, to_city)
2. analyze_options (no params needed)
3. finish with results

Return JSON: {{"top_flights": [flight IDs], "reasoning": "explanation"}}"""

        result = self.run(goal)
        
        if result["success"]:
            try:
                final = result["result"]
                if isinstance(final, str) and "{" in final:
                    parsed = json.loads(final[final.find("{"):final.rfind("}")+1])
                    llm_reasoning = parsed.get("reasoning", "")
                elif isinstance(final, dict):
                    llm_reasoning = final.get("reasoning", str(final))
                else:
                    llm_reasoning = str(final)
            except Exception as e:
                llm_reasoning = f"Parse error: {e}"
        else:
            llm_reasoning = f"ReAct failed: {result.get('error', 'Unknown')}"
        
        # INITIAL PROPOSAL: Include DIVERSE options across ALL price tiers
        # Include cheapest, mid-range, and premium so PolicyAgent can reason about trade-offs
        # NO hardcoded premium bias - let agents negotiate quality vs price
        available = self.state.get_belief("available_flights", [])
        
        if not available:
            # Fallback: search directly
            flights = self.loader.search(from_city=query.from_city, to_city=query.to_city,
                                        max_price=query.max_price, departure_after=query.departure_after,
                                        departure_before=query.departure_before)
            available = flights if flights else []
        
        diverse_flights = []
        seen_ids = set()
        
        # Sort ALL flights by price to ensure cheapest are included
        all_sorted = sorted(available, key=lambda x: x.get('price_usd', 0))
        
        # Include cheapest options first (budget tier) - 3 options
        for f in all_sorted[:3]:
            if f['flight_id'] not in seen_ids:
                seen_ids.add(f['flight_id'])
                diverse_flights.append(f)
        
        # Then add mid-range for variety - 3 options
        mid_start = len(all_sorted) // 3
        mid_end = 2 * len(all_sorted) // 3
        for f in all_sorted[mid_start:mid_end][:3]:
            if f['flight_id'] not in seen_ids:
                seen_ids.add(f['flight_id'])
                diverse_flights.append(f)
        
        # Then add premium for quality-focused travelers - 2 options
        for f in all_sorted[-3:]:
            if f['flight_id'] not in seen_ids:
                seen_ids.add(f['flight_id'])
                diverse_flights.append(f)
        
        top_flights = [Flight(**f) for f in diverse_flights[:8]]
        
        self.log_message("orchestrator", f"Initial proposal: {len(top_flights)} diverse flight options (all price tiers)", "result")
        
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
        # CRITICAL: Sort by relevance to the issue before showing to LLM
        if issue == "budget_exceeded":
            # For budget issues, show CHEAPEST flights first so LLM can select them
            sorted_available = sorted(available, key=lambda x: x.get('price_usd', 999))
        elif issue == "quality_upgrade" and target_min > 0:
            # For quality upgrade with target, sort by distance from target midpoint
            target_mid = (target_min + target_max) / 2
            sorted_available = sorted(available, key=lambda x: abs(x.get('price_usd', 0) - target_mid))
        else:
            # For quality issues, show premium flights first
            sorted_available = sorted(available, key=lambda x: -x.get('price_usd', 0))
        
        flight_summary = []
        for f in sorted_available[:20]:  # Show 20 options sorted by relevance
            flight_summary.append(
                f"{f['flight_id']}: {f['airline']} {f.get('class', 'Economy')} "
                f"${f['price_usd']} {f['departure_time']}->{f['arrival_time']} ({f['duration_hours']:.1f}h)"
            )
        
        # LLM prompt for agentic reasoning
        prompt = f"""You are a Flight Booking Specialist agent. The PolicyAgent has rejected your proposal.

FEEDBACK FROM POLICY AGENT:
- Issue: {issue}
- Reasoning: {feedback.get('reasoning', 'No details provided')}
- Constraint: {feedback.get('max_price', feedback.get('min_class', 'Not specified'))}

AVAILABLE FLIGHTS ({from_city} → {to_city}):
{chr(10).join(flight_summary)}

YOUR TASK:
Analyze the feedback and select the best flights that address the PolicyAgent's concerns.
Consider trade-offs: if budget is tight, prioritize price; if quality is requested, prioritize class/timing.

For budget issues: Find the cheapest options, even if they don't meet the exact constraint.
For quality issues: Find premium options (Business/First class, morning flights).

Return JSON with your reasoning:
{{"selected_flights": ["flight_id1", "flight_id2", ...], "reasoning": "Your explanation of why these flights best address the feedback", "addresses_constraint": true/false}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            selected_ids = set(result.get("selected_flights", []))
            reasoning = result.get("reasoning", "Selected based on feedback analysis")
            
            # Get the selected flights
            selected = [f for f in available if f['flight_id'] in selected_ids]
            
            # Fallback: if LLM didn't select valid flights, use heuristic
            if not selected:
                if self.verbose:
                    print(f"    [FlightAgent] LLM selection empty, using fallback")
                if issue == "budget_exceeded":
                    selected = sorted(available, key=lambda x: x.get('price_usd', 999))[:8]
                    reasoning = "Fallback: Selecting cheapest available flights."
                elif issue == "quality_upgrade":
                    # For quality upgrade, select PREMIUM options (Business/First class)
                    premium = [f for f in available if f.get('class', 'Economy') in ['Business', 'First Class']]
                    if premium:
                        selected = sorted(premium, key=lambda x: -x.get('price_usd', 0))[:8]
                        reasoning = "Quality upgrade: Selecting Business/First class flights."
                    else:
                        # No premium class, select highest-priced economy
                        selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:8]
                        reasoning = "Quality upgrade: Selecting premium economy flights."
                else:
                    selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:8]
                    reasoning = "Fallback: Selecting premium flights."
            else:
                # IMPORTANT: For budget issues, ALWAYS include the absolute cheapest option
                if issue == "budget_exceeded":
                    cheapest = min(available, key=lambda x: x.get('price_usd', 999))
                    if cheapest not in selected:
                        selected.insert(0, cheapest)
                # For quality upgrade, include the premium options
                elif issue == "quality_upgrade":
                    premium = [f for f in available if f.get('class', 'Economy') in ['Business', 'First Class']]
                    for f in premium[:3]:
                        if f not in selected:
                            selected.insert(0, f)
            
            refined_flights = [Flight(**f) for f in selected[:8]]
            
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
            
            # Fallback to heuristic selection
            if issue == "budget_exceeded":
                selected = sorted(available, key=lambda x: x.get('price_usd', 999))[:8]
            else:
                selected = sorted(available, key=lambda x: (-x.get('stars', 0) if 'stars' in x else -x.get('price_usd', 0)))[:8]
            
            refined_flights = [Flight(**f) for f in selected]
            return FlightSearchResult(
                query=FlightQuery(from_city=from_city, to_city=to_city),
                flights=refined_flights,
                reasoning=f"Fallback selection addressing {issue}"
            )
    
    def _build_reasoning_trace(self, query: FlightQuery, result: Dict, final_reasoning: str) -> str:
        parts = [f"## Flight Search ReAct Trace",
                f"**Route**: {query.from_city} → {query.to_city}",
                f"**Iterations**: {result.get('iterations', 0)}", "### Steps:"]
        for step in result.get("reasoning_trace", []):
            parts.append(f"**Step {step.step_number}**: {step.action} → {step.observation[:150]}...")
        parts.append(f"### Final: {final_reasoning}")
        return "\n".join(parts)
