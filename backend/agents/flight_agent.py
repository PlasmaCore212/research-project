# backend/agents/flight_agent.py
"""
Flight Agent with ReAct Pattern and Chain-of-Thought Prompting

This agent specializes in flight search and selection for business travel.
It uses the ReAct pattern (Thought -> Action -> Observation) to:
1. Understand the traveler's requirements
2. Search for available flights
3. Analyze and compare options
4. Select the best flights based on business criteria

References:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
"""

from .base_agent import BaseReActAgent, AgentAction
from .models import FlightQuery, FlightSearchResult, Flight
from data.loaders import FlightDataLoader
from typing import List, Dict, Any, Optional
import json


class FlightAgent(BaseReActAgent):
    """
    Agentic Flight Search Agent with ReAct reasoning.
    
    This agent autonomously:
    - Searches flight databases with various criteria
    - Compares flights on multiple dimensions
    - Refines searches based on observations
    - Selects optimal flights for business travelers
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="FlightAgent",
            agent_role="Flight Booking Specialist",
            model_name=model_name,
            max_iterations=5,
            verbose=verbose
        )
        
        self.loader = FlightDataLoader()
        self.tools = self._register_tools()
        
        # Track search history for learning
        self.search_history: List[Dict] = []
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        """Register tools available to the Flight Agent"""
        return {
            "search_flights": AgentAction(
                name="search_flights",
                description="Search for flights between cities with optional filters",
                parameters={
                    "from_city": "str - departure city code (e.g., 'NYC')",
                    "to_city": "str - arrival city code (e.g., 'SF')",
                    "max_price": "int (optional) - maximum price in USD",
                    "departure_after": "str (optional) - earliest departure time 'HH:MM'",
                    "departure_before": "str (optional) - latest departure time 'HH:MM'"
                },
                function=self._tool_search_flights
            ),
            "get_flight_details": AgentAction(
                name="get_flight_details",
                description="Get detailed information about a specific flight",
                parameters={
                    "flight_id": "str - the flight ID to look up"
                },
                function=self._tool_get_flight_details
            ),
            "compare_flights": AgentAction(
                name="compare_flights",
                description="Compare multiple flights on specific criteria",
                parameters={
                    "flight_ids": "list - list of flight IDs to compare",
                    "criteria": "str - what to focus on: 'price', 'duration', 'timing', or 'overall'"
                },
                function=self._tool_compare_flights
            ),
            "analyze_price_range": AgentAction(
                name="analyze_price_range",
                description="Analyze the price distribution of available flights",
                parameters={
                    "from_city": "str - departure city",
                    "to_city": "str - arrival city"
                },
                function=self._tool_analyze_price_range
            )
        }
    
    def _get_system_prompt(self) -> str:
        """Get the domain-specific system prompt for Flight Agent"""
        return """You are an expert Flight Booking Specialist AI Agent for business travel.

YOUR EXPERTISE:
- Finding optimal flights for business travelers
- Balancing price, convenience, and timing
- Understanding business travel requirements (early arrivals, avoiding red-eyes)
- Comparing flight options across multiple criteria

REASONING APPROACH (Chain-of-Thought):
When analyzing flights, think through:
1. BUDGET: What price range fits the requirements?
2. TIMING: What departure/arrival times suit business needs?
3. DURATION: How does flight length compare across options?
4. VALUE: Which flights offer the best price-to-quality ratio?
5. RISK: Are there any potential issues (tight connections, late arrivals)?

BUSINESS TRAVEL PRIORITIES:
- Morning departures (6:00-10:00) are ideal for same-day meetings
- Avoid red-eye flights when possible
- Prefer shorter flight durations
- Balance cost with convenience
- Consider arrival time relative to meeting schedules"""
    
    def _tool_search_flights(
        self,
        from_city: str,
        to_city: str,
        max_price: Optional[int] = None,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None
    ) -> str:
        """Tool: Search flights database"""
        
        flights = self.loader.search(
            from_city=from_city,
            to_city=to_city,
            max_price=max_price,
            departure_after=departure_after,
            departure_before=departure_before
        )
        
        if not flights:
            return f"No flights found from {from_city} to {to_city} matching criteria."
        
        # Update beliefs
        self.state.add_belief("last_search_from", from_city)
        self.state.add_belief("last_search_to", to_city)
        self.state.add_belief("available_flights", flights)
        self.state.add_belief("flight_count", len(flights))
        
        # Format results
        result_lines = [f"Found {len(flights)} flights from {from_city} to {to_city}:"]
        for f in flights[:10]:  # Show top 10
            result_lines.append(
                f"  - {f['flight_id']}: {f['airline']}, "
                f"${f['price_usd']}, {f['departure_time']}→{f['arrival_time']}, "
                f"{f['duration_hours']:.1f}h"
            )
        
        if len(flights) > 10:
            result_lines.append(f"  ... and {len(flights) - 10} more")
        
        # Track search
        self.search_history.append({
            "from": from_city,
            "to": to_city,
            "max_price": max_price,
            "results": len(flights)
        })
        
        return "\n".join(result_lines)
    
    def _tool_get_flight_details(self, flight_id: str) -> str:
        """Tool: Get details about a specific flight"""
        
        flights = self.state.get_belief("available_flights", [])
        
        for f in flights:
            if f['flight_id'] == flight_id:
                return f"""Flight Details for {flight_id}:
- Airline: {f['airline']}
- Route: {f['from_city']} → {f['to_city']}
- Departure: {f['departure_time']}
- Arrival: {f['arrival_time']}
- Duration: {f['duration_hours']:.2f} hours
- Price: ${f['price_usd']}
- Class: {f.get('class', 'Economy')}
- Seats Available: {f.get('seats_available', 'Unknown')}"""
        
        return f"Flight {flight_id} not found in recent search results."
    
    def _tool_compare_flights(self, flight_ids: List[str], criteria: str = "overall") -> str:
        """Tool: Compare multiple flights"""
        
        flights = self.state.get_belief("available_flights", [])
        flight_dict = {f['flight_id']: f for f in flights}
        
        to_compare = [flight_dict[fid] for fid in flight_ids if fid in flight_dict]
        
        if not to_compare:
            return "No valid flights to compare. Search for flights first."
        
        comparison_lines = [f"Comparison of {len(to_compare)} flights ({criteria} focus):"]
        comparison_lines.append("-" * 50)
        
        if criteria == "price":
            sorted_flights = sorted(to_compare, key=lambda x: x['price_usd'])
            for i, f in enumerate(sorted_flights, 1):
                price = f['price_usd']
                diff = price - sorted_flights[0]['price_usd']
                label = 'cheapest' if i == 1 else f'+${diff}'
                comparison_lines.append(
                    f"{i}. {f['flight_id']}: ${price} ({label})"
                )
        
        elif criteria == "duration":
            sorted_flights = sorted(to_compare, key=lambda x: x['duration_hours'])
            for i, f in enumerate(sorted_flights, 1):
                duration = f['duration_hours']
                diff = duration - sorted_flights[0]['duration_hours']
                label = 'fastest' if i == 1 else f'+{diff:.1f}h'
                comparison_lines.append(
                    f"{i}. {f['flight_id']}: {duration:.1f}h ({label})"
                )
        
        elif criteria == "timing":
            # Score by business-friendliness (early morning best)
            def timing_score(f):
                hour = int(f['departure_time'].split(':')[0])
                if 6 <= hour <= 10:
                    return 0  # Best
                elif 10 < hour <= 14:
                    return 1  # Good
                elif 14 < hour <= 18:
                    return 2  # OK
                else:
                    return 3  # Poor (red-eye or very early)
            
            sorted_flights = sorted(to_compare, key=timing_score)
            timing_labels = {0: "Excellent", 1: "Good", 2: "Fair", 3: "Poor"}
            for i, f in enumerate(sorted_flights, 1):
                score = timing_score(f)
                comparison_lines.append(
                    f"{i}. {f['flight_id']}: {f['departure_time']} "
                    f"({timing_labels[score]} for business)"
                )
        
        else:  # overall
            # Combined score: normalize price (40%), duration (30%), timing (30%)
            prices = [f['price_usd'] for f in to_compare]
            durations = [f['duration_hours'] for f in to_compare]
            min_price, max_price = min(prices), max(prices)
            min_dur, max_dur = min(durations), max(durations)
            
            def overall_score(f):
                # Normalize to 0-1 (lower is better)
                price_norm = (f['price_usd'] - min_price) / (max_price - min_price + 1)
                dur_norm = (f['duration_hours'] - min_dur) / (max_dur - min_dur + 0.1)
                
                hour = int(f['departure_time'].split(':')[0])
                timing_norm = 0 if 6 <= hour <= 10 else (0.5 if 10 < hour <= 14 else 1)
                
                return 0.4 * price_norm + 0.3 * dur_norm + 0.3 * timing_norm
            
            sorted_flights = sorted(to_compare, key=overall_score)
            for i, f in enumerate(sorted_flights, 1):
                score = overall_score(f)
                comparison_lines.append(
                    f"{i}. {f['flight_id']}: ${f['price_usd']}, {f['duration_hours']:.1f}h, "
                    f"dep {f['departure_time']} (score: {score:.2f})"
                )
        
        return "\n".join(comparison_lines)
    
    def _tool_analyze_price_range(self, from_city: str, to_city: str) -> str:
        """Tool: Analyze price distribution for route"""
        
        flights = self.loader.search(from_city=from_city, to_city=to_city)
        
        if not flights:
            return f"No flights found for {from_city} → {to_city}"
        
        prices = [f['price_usd'] for f in flights]
        
        return f"""Price Analysis for {from_city} → {to_city}:
- Minimum: ${min(prices)}
- Maximum: ${max(prices)}
- Average: ${sum(prices) / len(prices):.0f}
- Total flights: {len(flights)}
- Budget options (<${sum(prices) / len(prices):.0f}): {len([p for p in prices if p < sum(prices) / len(prices)])}
- Premium options (>${sum(prices) / len(prices):.0f}): {len([p for p in prices if p >= sum(prices) / len(prices)])}"""
    
    def search_flights(self, query: FlightQuery) -> FlightSearchResult:
        """
        Main entry point for flight search using ReAct reasoning.
        
        This method triggers the agentic ReAct loop to find the best flights.
        """
        
        # Reset state for new search
        self.reset_state()
        
        # Build the goal description
        goal = f"""Find the top 3 best flights for a business traveler:
- Route: {query.from_city} to {query.to_city}
- Maximum price: ${query.max_price if query.max_price else 'No limit'}
- Preferred departure time: {query.departure_after or '06:00'} to {query.departure_before or '21:00'}
- Class preference: {query.class_preference}

Steps to follow:
1. First, search for available flights matching the criteria
2. Analyze the options considering price, duration, and timing
3. Compare the top candidates
4. Select the 3 best flights and explain why

Return your final answer as a JSON object with:
- top_3_flights: list of flight IDs
- reasoning: detailed explanation of your selection"""

        # Run ReAct loop
        result = self.run(goal)
        
        # Extract flights from result
        if result["success"]:
            try:
                # Parse the result
                final_answer = result["result"]
                if isinstance(final_answer, str):
                    # Try to extract JSON from the result
                    if "{" in final_answer:
                        json_str = final_answer[final_answer.find("{"):final_answer.rfind("}")+1]
                        parsed = json.loads(json_str)
                        top_ids = parsed.get("top_3_flights", [])
                        llm_reasoning = parsed.get("reasoning", "")
                    else:
                        top_ids = []
                        llm_reasoning = final_answer
                else:
                    top_ids = final_answer.get("top_3_flights", []) if isinstance(final_answer, dict) else []
                    llm_reasoning = final_answer.get("reasoning", "") if isinstance(final_answer, dict) else str(final_answer)
                
                # Get flights from beliefs
                available_flights = self.state.get_belief("available_flights", [])
                flight_dict = {f['flight_id']: f for f in available_flights}
                
                # Map to Flight objects
                top_flights = []
                for fid in top_ids[:3]:
                    if fid in flight_dict:
                        top_flights.append(Flight(**flight_dict[fid]))
                
                # Fallback if no valid flights found
                if not top_flights and available_flights:
                    top_flights = [Flight(**f) for f in available_flights[:3]]
                    llm_reasoning = "Fallback selection: top 3 by default sorting."
                
            except Exception as e:
                # Fallback
                available_flights = self.state.get_belief("available_flights", [])
                top_flights = [Flight(**f) for f in available_flights[:3]] if available_flights else []
                llm_reasoning = f"Selection completed with fallback: {str(e)}"
        else:
            # ReAct failed, use simple search
            flights = self.loader.search(
                from_city=query.from_city,
                to_city=query.to_city,
                max_price=query.max_price,
                departure_after=query.departure_after,
                departure_before=query.departure_before
            )
            top_flights = [Flight(**f) for f in flights[:3]] if flights else []
            llm_reasoning = f"ReAct reasoning failed: {result.get('error', 'Unknown error')}. Used fallback search."
        
        # Build comprehensive reasoning trace
        reasoning = self._build_react_reasoning(query, result, llm_reasoning)
        
        # Log completion message
        self.log_message(
            to_agent="orchestrator",
            content=f"Found {len(top_flights)} recommended flights",
            msg_type="result"
        )
        
        return FlightSearchResult(
            query=query,
            flights=top_flights,
            reasoning=reasoning
        )
    
    def _build_react_reasoning(
        self,
        query: FlightQuery,
        react_result: Dict,
        final_reasoning: str
    ) -> str:
        """Build the full ReAct reasoning trace for transparency"""
        
        # Header
        reasoning_parts = [
            f"## Flight Search ReAct Reasoning Trace",
            f"**Agent**: {self.agent_name}",
            f"**Goal**: Find flights from {query.from_city} to {query.to_city}",
            f"**Iterations**: {react_result.get('iterations', 0)}",
            f"**Success**: {react_result.get('success', False)}",
            "",
            "### Reasoning Steps:",
        ]
        
        # Add each ReAct step
        for step in react_result.get("reasoning_trace", []):
            reasoning_parts.append(f"""
**Step {step.step_number}**:
- **Thought**: {step.thought}
- **Action**: `{step.action}({json.dumps(step.action_input)})`
- **Observation**: {step.observation[:200]}{'...' if len(step.observation) > 200 else ''}
""")
        
        # Final conclusion
        reasoning_parts.append(f"""
### Final Selection:
{final_reasoning}
""")
        
        return "\n".join(reasoning_parts)
