# backend/agents/flight_agent.py
"""Flight Agent with ReAct Pattern for business travel flight search."""

from .base_agent import BaseReActAgent, AgentAction
from .models import FlightQuery, FlightSearchResult, Flight
from data.loaders import FlightDataLoader
from typing import List, Dict, Any, Optional
import json


class FlightAgent(BaseReActAgent):
    """Agentic Flight Search Agent with ReAct reasoning."""
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="FlightAgent", agent_role="Flight Booking Specialist",
            model_name=model_name, max_iterations=5, verbose=verbose
        )
        self.loader = FlightDataLoader()
        self.tools = self._register_tools()
        self.search_history: List[Dict] = []
    
    def _should_stop_early(self, observation: str) -> bool:
        obs_lower = observation.lower()
        has_flights = self.state.get_belief("available_flights") is not None
        signals = ["top 3", "best flights", "recommend", "final", "selection complete", "comparison of"]
        return has_flights and any(s in obs_lower for s in signals)
    
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
        
        # Score by business-friendliness (duration + timing penalty)
        def business_score(f):
            hour = int(f.get('departure_time', '12:00').split(':')[0])
            timing_penalty = 0 if 6 <= hour <= 10 else (1 if 10 < hour <= 14 else 2)
            return f.get('duration_hours', 10) + timing_penalty
        
        for tier in tiers:
            tiers[tier].sort(key=business_score)
        
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
                description="Compare multiple flights on specific criteria",
                parameters={"flight_ids": "list", "criteria": "str - 'price', 'duration', 'timing', or 'overall'"},
                function=self._tool_compare_flights
            ),
            "analyze_options": AgentAction(
                name="analyze_options",
                description="Analyze all available flight options by price tiers and business-friendliness",
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
        return """You are an expert Flight Booking Specialist for business travel.
PRIORITIES: Morning departures (6-10am), shorter durations, balance cost with convenience.
REASONING: Consider budget, timing, duration, value, and risk for each option."""
    
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
        
        self.search_history.append({"from": from_city, "to": to_city, "max_price": max_price, "results": len(flights)})
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
            def timing_score(f):
                hour = int(f['departure_time'].split(':')[0])
                return 0 if 6 <= hour <= 10 else (1 if 10 < hour <= 14 else 2)
            sorted_f = sorted(to_compare, key=timing_score)
            return "\n".join(f"{i+1}. {f['flight_id']}: {f['departure_time']}" for i, f in enumerate(sorted_f))
        else:  # overall
            prices = [f['price_usd'] for f in to_compare]
            durations = [f['duration_hours'] for f in to_compare]
            min_p, max_p, min_d, max_d = min(prices), max(prices), min(durations), max(durations)
            
            def score(f):
                price_n = (f['price_usd'] - min_p) / (max_p - min_p + 1)
                dur_n = (f['duration_hours'] - min_d) / (max_d - min_d + 0.1)
                hour = int(f['departure_time'].split(':')[0])
                timing_n = 0 if 6 <= hour <= 10 else (0.5 if 10 < hour <= 14 else 1)
                return 0.4 * price_n + 0.3 * dur_n + 0.3 * timing_n
            
            sorted_f = sorted(to_compare, key=score)
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
        
        # Analyze morning flights (6-10am) - best for business
        morning = [f for f in flights if 6 <= int(f['departure_time'].split(':')[0]) <= 10]
        
        result = [f"Flight Analysis ({len(flights)} total):"]
        for tier, tier_flights in tiers.items():
            if tier_flights:
                tier_prices = [f['price_usd'] for f in tier_flights]
                result.append(f"  {tier}: {len(tier_flights)} flights, ${min(tier_prices)}-${max(tier_prices)}")
        
        result.append(f"  Morning departures (6-10am): {len(morning)} flights")
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
        
        goal = f"""Find best flights for business: {query.from_city} to {query.to_city}
Max price: ${query.max_price if query.max_price else 'No limit'}
Preferred departure: {query.departure_after or '06:00'} to {query.departure_before or '21:00'}

1. Search for flights 2. Analyze options 3. Compare top candidates across price tiers
Return JSON: {{"top_flights": [flight IDs from all price tiers], "reasoning": "explanation"}}"""

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
        
        # IMPORTANT: Return DIVERSE options, not just LLM's filtered picks
        # PolicyAgent needs variety to make informed budget decisions
        available = self.state.get_belief("available_flights", [])
        
        if not available:
            # Fallback: search directly
            flights = self.loader.search(from_city=query.from_city, to_city=query.to_city,
                                        max_price=query.max_price, departure_after=query.departure_after,
                                        departure_before=query.departure_before)
            available = flights if flights else []
        
        # Build diverse set: include options from each flight class
        diverse_flights = []
        seen_ids = set()
        
        # Group by class
        by_class = {'Economy': [], 'Business': [], 'First Class': []}
        for f in available:
            flight_class = f.get('class', 'Economy')
            if flight_class in by_class:
                by_class[flight_class].append(f)
        
        # Take best from each class (by timing/duration for business travel)
        def business_score(f):
            hour = int(f.get('departure_time', '12:00').split(':')[0])
            timing = 0 if 6 <= hour <= 10 else (1 if 10 < hour <= 14 else 2)
            return (timing, f.get('duration_hours', 5))
        
        for flight_class in ['Economy', 'Business', 'First Class']:
            class_flights = sorted(by_class[flight_class], key=business_score)
            for f in class_flights[:3]:  # Top 3 from each class
                if f['flight_id'] not in seen_ids:
                    seen_ids.add(f['flight_id'])
                    diverse_flights.append(f)
        
        # If we have fewer than 5, add more economy options
        if len(diverse_flights) < 5:
            for f in sorted(available, key=lambda x: x.get('price_usd', 999)):
                if f['flight_id'] not in seen_ids and len(diverse_flights) < 8:
                    seen_ids.add(f['flight_id'])
                    diverse_flights.append(f)
        
        top_flights = [Flight(**f) for f in diverse_flights[:10]]  # Up to 10 diverse options
        
        self.log_message("orchestrator", f"Found {len(top_flights)} diverse flight options", "result")
        
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
                - issue: "budget_exceeded" | "quality_insufficient" | "timing_conflict"
                - max_price: suggested max price (if budget issue)
                - min_class: suggested class upgrade (if quality issue)
                - reasoning: PolicyAgent's explanation
            previous_flights: Flights from previous proposal
        
        Returns:
            Refined FlightSearchResult with new proposal and reasoning
        """
        issue = feedback.get("issue", "general")
        from_city = feedback.get("from_city", "")
        to_city = feedback.get("to_city", "")
        
        if self.verbose:
            print(f"    [FlightAgent] Reasoning about feedback: {issue}")
        
        # Get all available flights from loader
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
        flight_summary = []
        for f in available[:15]:  # Top 15 for context
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
                else:
                    selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:8]
                    reasoning = "Fallback: Selecting premium flights."
            
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
