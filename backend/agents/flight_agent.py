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
        """Extract SINGLE best flight when agent fails to call finish()."""
        flights = self.state.get_belief("available_flights", [])
        if not flights:
            return {"result": "No flights found"}
        
        # Return best VALUE flight (balance price, duration, class) - NOT just cheapest
        def value_score(f):
            # Lower score = better value
            price_norm = f.get('price_usd', 999) / 1000  # Normalize to 0-2 range
            duration_norm = f.get('duration_hours', 10) / 10  # Normalize to 0-2 range
            class_bonus = -0.3 if f.get('class') in ['Business', 'First Class'] else 0
            return price_norm + duration_norm + class_bonus
        
        sorted_flights = sorted(flights, key=value_score)
        best_flight = sorted_flights[0]
        
        return {"selected_flights": [best_flight['flight_id']],
                "reasoning": "Fallback: Selected best value flight (price + duration + class)."}
    
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
        return """You are a Flight Booking Specialist finding the SINGLE best flight.

⚠️ CRITICAL: You can ONLY use these 5 tools. Any other tool name will FAIL:

1. search_flights(from_city="NYC", to_city="SF") - Search flights. ALWAYS call this FIRST.
2. compare_flights(flight_ids=["FL0001","FL0002"]) - Compare flights. REQUIRES a LIST of IDs.
3. get_flight_details(flight_id="FL0001") - Get ONE flight's details. REQUIRES flight_id as STRING.
4. analyze_options() - Get price tier summary. No parameters needed.
5. finish(result={...}) - Return final selection. MUST call this when done!

❌ THESE TOOLS DO NOT EXIST - DO NOT USE THEM:
- filter_flights ❌
- sort_flights ❌
- check_seat_availability ❌
- book_flight ❌

YOUR TASK: Search for flights and return EXACTLY ONE flight - your absolute best recommendation.

🚫 CRITICAL - NEVER JUST PICK THE CHEAPEST:
- DO NOT select based on price alone
- DO NOT say "cheapest option" in your reasoning
- You MUST evaluate MULTIPLE factors (price, class, timing, duration)
- "Best value" ≠ "cheapest" - it means the best BALANCE

⚠️ IMPORTANT: You do NOT know the user's budget. Make your own judgment about quality vs cost tradeoff.

⚠️ CRITICAL: Consider ALL flight classes (Economy, Business, First Class) equally!
- DO NOT assume business travelers need Business class
- Economy can be excellent value for short/medium flights
- Business/First makes sense for long flights or if budget allows

EVALUATION CRITERIA (ALL are important):
1. **Price** - Reasonable but not necessarily cheapest
2. **Flight Class** - Economy for <6h, Business for >6h is ideal but flexible
3. **Timing** - Convenient departure/arrival times (avoid red-eyes if possible)
4. **Duration** - Shorter is better
5. **Airline** - Reputable carriers

WORKFLOW:
1. search_flights(from_city="...", to_city="...") → See ALL options
2. analyze_options() → Understand price tiers
3. compare_flights(flight_ids=[...]) → Compare 3-5 diverse options from different tiers
4. finish(result={"selected_flights": ["<YOUR_ONE_BEST_PICK>"], "reasoning": "Why this is THE best VALUE considering ALL factors"})

GOAL: Find the SINGLE best VALUE option (good BALANCE of quality, price, timing, duration).

⚠️ CRITICAL: Return EXACTLY ONE flight, not multiple options!
⚠️ REMEMBER: You MUST call finish() to complete your task!"""
    
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
        
        # Separate by class to highlight Business/First class
        business_first = [f for f in flights if f.get('class') in ['Business', 'First Class']]
        economy = [f for f in flights if f.get('class', 'Economy') == 'Economy']
        
        result = [f"Found {len(flights)} flights from {from_city} to {to_city}:"]
        
        # Show Business/First class first (if any)
        if business_first:
            result.append(f"\n📍 BUSINESS/FIRST CLASS ({len(business_first)} options):")
            for f in sorted(business_first, key=lambda x: x['price_usd'])[:5]:
                result.append(f"  - {f['flight_id']}: {f['airline']} {f.get('class', 'Economy')}, ${f['price_usd']}, "
                             f"{f['departure_time']}→{f['arrival_time']}, {f['duration_hours']:.1f}h")
        
        # Then show Economy
        result.append(f"\n📍 ECONOMY ({len(economy)} options):")
        for f in sorted(economy, key=lambda x: x['price_usd'])[:8]:
            result.append(f"  - {f['flight_id']}: {f['airline']} Economy, ${f['price_usd']}, "
                         f"{f['departure_time']}→{f['arrival_time']}, {f['duration_hours']:.1f}h")
        
        if len(flights) > 13:
            result.append(f"  ... and {len(flights) - 13} more")
        
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
        else:  # overall - multi-factor value score
            def value_score(f):
                # Lower score = better overall value
                price_norm = f['price_usd'] / 1000
                duration_norm = f['duration_hours'] / 10
                class_bonus = -0.3 if f.get('class') in ['Business', 'First Class'] else 0
                return price_norm + duration_norm + class_bonus
            sorted_f = sorted(to_compare, key=value_score)
            return "\n".join(f"{i+1}. {f['flight_id']}: {f.get('class', 'Economy')}, ${f['price_usd']}, {f['duration_hours']:.1f}h, {f['departure_time']}" 
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

        # AGENTIC PROMPT: Let the agent decide what's best value
        goal = f"""Find the SINGLE best value flight: {query.from_city} to {query.to_city}

YOUR TASK: Search for flights and return EXACTLY ONE flight - your absolute best recommendation.

🚫 CRITICAL - NEVER JUST PICK THE CHEAPEST:
- DO NOT select based on price alone
- DO NOT say "cheapest option" in your reasoning
- You MUST evaluate MULTIPLE factors (price, class, timing, duration)
- "Best value" ≠ "cheapest" - it means the best BALANCE

⚠️ IMPORTANT: You do NOT know the user's budget. Make your own judgment about quality vs cost tradeoff.

⚠️ CRITICAL: Consider ALL flight classes (Economy, Business, First Class) equally!
- DO NOT assume business travelers need Business class
- Economy can be excellent value for short/medium flights (<6h)
- Business/First makes sense for long flights (>6h) or if budget allows

EVALUATION CRITERIA (ALL are important):
1. **Price** - Reasonable but not necessarily cheapest
2. **Flight Class** - Economy for <6h, Business for >6h is ideal but flexible
3. **Timing** - Convenient departure/arrival times (avoid red-eyes)
4. **Duration** - Shorter is better
5. **Airline** - Reputable carriers

WORKFLOW:
1. search_flights(from_city="{query.from_city}", to_city="{query.to_city}")
2. analyze_options() → Understand price tiers
3. compare_flights() to compare 3-5 diverse options from different tiers
4. finish() with EXACTLY ONE flight ID

Return JSON: {{"selected_flights": ["<YOUR_ONE_BEST_PICK>"], "reasoning": "Why this is THE best VALUE considering ALL factors (not just price)"}}

⚠️ CRITICAL: Return EXACTLY ONE flight, not multiple options!"""
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
        
        # FALLBACK: If LLM didn't select valid flights, use best value option
        # (This should rarely happen with good prompting)
        print(f"    [FlightAgent] LLM selection failed, using fallback (best value option)")
        
        def value_score(f):
            price_norm = f.get('price_usd', 999) / 1000
            duration_norm = f.get('duration_hours', 10) / 10
            class_bonus = -0.3 if f.get('class') in ['Business', 'First Class'] else 0
            return price_norm + duration_norm + class_bonus
        
        fallback_flights = sorted(available, key=value_score)[:1]  # ONLY 1 flight
        top_flights = [Flight(**f) for f in fallback_flights]
        
        self.log_message("orchestrator", f"Fallback proposal: 1 flight (best value)", "result")
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
            
            # Fallback: if LLM didn't select valid flights, use heuristic
            if not selected:
                if self.verbose:
                    print(f"    [FlightAgent] LLM selection empty, using fallback")
                if issue == "budget_exceeded":
                    selected = sorted(available, key=lambda x: x.get('price_usd', 999))[:1]
                    reasoning = "Fallback: Selecting cheapest available flight."
                elif issue == "quality_upgrade":
                    # For quality upgrade, select PREMIUM options (Business/First class)
                    premium = [f for f in available if f.get('class', 'Economy') in ['Business', 'First Class']]
                    if premium:
                        selected = sorted(premium, key=lambda x: -x.get('price_usd', 0))[:1]
                        reasoning = "Quality upgrade: Selecting Business/First class flight."
                    else:
                        # No premium class, select highest-priced economy
                        selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:1]
                        reasoning = "Quality upgrade: Selecting premium economy flight."
                else:
                    selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:1]
                    reasoning = "Fallback: Selecting premium flights."
            
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
            
            # Fallback to heuristic selection
            if issue == "budget_exceeded":
                selected = sorted(available, key=lambda x: x.get('price_usd', 999))[:1]
            else:
                # For quality upgrades, select most expensive option
                selected = sorted(available, key=lambda x: -x.get('price_usd', 0))[:1]
            
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
