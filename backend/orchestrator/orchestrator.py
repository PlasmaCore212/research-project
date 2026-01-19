# backend/orchestrator/orchestrator.py
"""
Trip Orchestrator with ReAct Pattern and Agent Memory

This orchestrator coordinates all specialized agents (Flight, Hotel, Time)
using principles from:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2023)
- BDI Architecture (Rao & Georgeff, 1995)
- Contract Net Protocol (Smith, 1980)
- Chain-of-Thought Prompting (Wei et al., 2022)

The orchestrator:
1. Maintains shared state accessible to all agents
2. Coordinates agent communication
3. Tracks message exchanges for metrics
4. Makes high-level decisions about trip planning
"""

from langchain_ollama import OllamaLLM
from agents.models import Flight, Hotel
from data.loaders import FlightDataLoader, HotelDataLoader
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AgentMessage:
    """Represents a message between agents (for FIPA-ACL style communication)"""
    performative: str  # inform, request, propose, accept, reject
    sender: str
    receiver: str
    content: Any
    conversation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratorMemory:
    """
    Shared memory for the orchestrator (BDI-inspired).
    
    This maintains beliefs about the current state of trip planning,
    tracks all agent interactions, and stores reasoning history.
    """
    # Current planning state
    beliefs: Dict[str, Any] = field(default_factory=dict)
    
    # All messages exchanged between agents
    message_history: List[AgentMessage] = field(default_factory=list)
    
    # Reasoning trace for explainability
    reasoning_steps: List[Dict] = field(default_factory=list)
    
    # Metrics for research
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_messages": 0,
        "iterations": 0,
        "agent_calls": {},
        "start_time": None,
        "end_time": None
    })
    
    def add_belief(self, key: str, value: Any):
        """Update a belief"""
        self.beliefs[key] = value
    
    def get_belief(self, key: str, default: Any = None) -> Any:
        """Get a belief value"""
        return self.beliefs.get(key, default)
    
    def log_message(self, msg: AgentMessage):
        """Log an agent message"""
        self.message_history.append(msg)
        self.metrics["total_messages"] += 1
    
    def log_reasoning(self, step: Dict):
        """Log a reasoning step"""
        self.reasoning_steps.append({
            **step,
            "timestamp": datetime.now().isoformat()
        })
    
    def increment_agent_call(self, agent_name: str):
        """Track agent calls for metrics"""
        if agent_name not in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][agent_name] = 0
        self.metrics["agent_calls"][agent_name] += 1
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of metrics for research"""
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            duration = (end - start).total_seconds()
        else:
            duration = 0
        
        return {
            "total_messages": self.metrics["total_messages"],
            "iterations": self.metrics["iterations"],
            "agent_calls": self.metrics["agent_calls"],
            "duration_seconds": duration,
            "reasoning_steps": len(self.reasoning_steps)
        }


class TripOrchestrator:
    """
    Central coordinator for the multi-agent trip planning system.
    
    This orchestrator uses Chain-of-Thought reasoning for:
    - Selecting the best flight + hotel combinations
    - Adjusting budgets based on budget constraints
    - Coordinating feedback loops between agents
    """
    
    def __init__(self, model_name: str = "qwen2.5:14b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        
        # Data loaders for querying market data
        self.flight_loader = FlightDataLoader()
        self.hotel_loader = HotelDataLoader()
        
        # Initialize orchestrator memory
        self.memory = OrchestratorMemory()
        self.memory.metrics["start_time"] = datetime.now().isoformat()
        
        # Conversation ID for message tracking
        self.conversation_id = f"trip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _log(self, message: str):
        """Log a message if verbose mode is on"""
        if self.verbose:
            print(f"[Orchestrator] {message}")
    
    def _send_message(
        self,
        to_agent: str,
        content: Any,
        performative: str = "inform"
    ) -> AgentMessage:
        """Send a message to an agent (FIPA-ACL style)"""
        msg = AgentMessage(
            performative=performative,
            sender="orchestrator",
            receiver=to_agent,
            content=content,
            conversation_id=self.conversation_id
        )
        self.memory.log_message(msg)
        return msg
    
    def select_bookings(self, state: Dict) -> Dict:
        """
        Select the best flight and hotel combination using Chain-of-Thought.
        
        This method analyzes available options and selects the optimal combination
        based on price, quality, and budget constraints.
        """
        
        self.memory.increment_agent_call("orchestrator_select")
        
        flights = state.get("flight_options", [])
        hotels = state.get("hotel_options", [])
        budget = state.get("total_budget", 2000)
        nights = state.get("nights", 1)
        
        if not flights or not hotels:
            self._log("No options to select from")
            return state
        
        # Log reasoning step
        self.memory.log_reasoning({
            "step": "select_bookings",
            "thought": f"Analyzing {len(flights)} flights and {len(hotels)} hotels",
            "context": {"budget": budget, "nights": nights}
        })
        
        # Prepare data for LLM
        flights_data = [
            {
                "id": f.flight_id,
                "airline": f.airline,
                "price": f.price_usd,
                "departure": f.departure_time,
                "arrival": f.arrival_time,
                "duration": f.duration_hours
            }
            for f in flights[:5]
        ]
        
        hotels_data = [
            {
                "id": h.hotel_id,
                "name": h.name,
                "price_per_night": h.price_per_night_usd,
                "stars": h.stars,
                "distance_km": h.distance_to_business_center_km
            }
            for h in hotels[:5]
        ]
        
        # Chain-of-Thought prompt
        prompt = f"""You are the Trip Planning Orchestrator. Select the BEST flight and hotel combination.

BUDGET CONSTRAINT: ${budget} total for flight + ({nights} nights × hotel)

AVAILABLE FLIGHTS:
{json.dumps(flights_data, indent=2)}

AVAILABLE HOTELS:
{json.dumps(hotels_data, indent=2)}

THINK STEP BY STEP:

Step 1 - Calculate Maximum Hotel Budget:
- Total budget: ${budget}
- After cheapest flight (${{min flight price}}), remaining for hotel: ${{remaining}}
- Per night budget: ${{remaining}} / {nights} nights = ${{per_night}}

Step 2 - Evaluate Each Flight:
- Consider price, timing, and duration
- Shorter flights are generally better

Step 3 - Evaluate Each Hotel:
- Consider price per night, stars, and distance to business center
- Closer to business center is better
- Higher stars = better quality

Step 4 - Find Best Combinations:
- List viable combinations that fit budget
- Calculate total cost for each: flight + (hotel × {nights})

Step 5 - Select Optimal:
- Choose the combination with best VALUE (quality relative to cost)
- Prefer higher hotel stars if budget allows
- Prefer closer to business center

Return ONLY this JSON (no other text):
{{
    "reasoning_steps": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ...",
        "Step 4: ...",
        "Step 5: ..."
    ],
    "flight_id": "selected flight ID",
    "hotel_id": "selected hotel ID",
    "total_cost": calculated_total,
    "justification": "Why this is the best combination"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            selected_flight_id = result.get("flight_id")
            selected_hotel_id = result.get("hotel_id")
            justification = result.get("justification", "")
            reasoning_steps = result.get("reasoning_steps", [])
            
            # Log the reasoning
            self.memory.log_reasoning({
                "step": "selection_complete",
                "thought": justification,
                "reasoning_steps": reasoning_steps,
                "selection": {"flight": selected_flight_id, "hotel": selected_hotel_id}
            })
            
            # Find the actual objects
            selected_flight = next(
                (f for f in flights if f.flight_id == selected_flight_id),
                flights[0]  # Fallback
            )
            selected_hotel = next(
                (h for h in hotels if h.hotel_id == selected_hotel_id),
                hotels[0]  # Fallback
            )
            
            # Send messages (for tracking)
            self._send_message(
                "flight_agent",
                f"Selected flight {selected_flight.flight_id}",
                "inform"
            )
            self._send_message(
                "hotel_agent",
                f"Selected hotel {selected_hotel.hotel_id}",
                "inform"
            )
            
            # Update state (if messages key exists)
            if "messages" in state:
                state["messages"].append({
                    "from": "orchestrator",
                    "to": "agents",
                    "content": f"Selected {selected_flight.flight_id} + {selected_hotel.hotel_id}: {justification}"
                })
            
            self._log(f"Selected: {selected_flight.flight_id} + {selected_hotel.hotel_id}")
            
            return {
                **state,
                "selected_flight": selected_flight,
                "selected_hotel": selected_hotel
            }
            
        except Exception as e:
            self._log(f"Selection failed: {e}, using fallback")

            # Fallback: select first options
            return {
                **state,
                "selected_flight": flights[0],
                "selected_hotel": hotels[0]
            }

    def handle_policy_result(self, validation_result: Dict, state: Dict) -> Dict:
        """
        Interpret PolicyAgent's validation and decide what to do next.

        Args:
            validation_result: Dict from PolicyAgent with:
                {"is_valid": bool, "violations": [...], "total_cost": float, "budget_remaining": float}
            state: Current TripPlanningState

        Returns:
            Dict with: {"action": "negotiate"/"accept"/"fail", "reasoning": str, "next_node": str}
        """
        self.memory.increment_agent_call("orchestrator_policy_handler")

        is_valid = validation_result.get("is_valid", False)
        violations = validation_result.get("violations", [])
        total_cost = validation_result.get("total_cost", 0)
        budget = state.get("budget", 2000)
        budget_remaining = validation_result.get("budget_remaining", 0)

        # Calculate budget utilization percentage
        utilization = (total_cost / budget * 100) if budget > 0 else 0

        # Log reasoning
        self.memory.log_reasoning({
            "step": "handle_policy_result",
            "thought": f"Analyzing policy validation: valid={is_valid}, utilization={utilization:.1f}%",
            "context": {"total_cost": total_cost, "budget": budget, "violations": len(violations)}
        })

        # Get selected flight and hotel details for quality assessment
        # ALWAYS do comprehensive review - don't blindly accept based on budget alone
        selected_flight = state.get("selected_flight", {})
        selected_hotel = state.get("selected_hotel", {})
        
        # Handle None values (safety check)
        if selected_flight is None:
            selected_flight = {}
        if selected_hotel is None:
            selected_hotel = {}
        
        # Extract flight details
        flight_cost = selected_flight.get("price_usd", 0)
        flight_class = selected_flight.get("class", "Economy")
        flight_airline = selected_flight.get("airline", "Unknown")
        flight_departure = selected_flight.get("departure_time", "N/A")
        flight_arrival = selected_flight.get("arrival_time", "N/A")
        flight_duration = selected_flight.get("duration_hours", 0)
        
        # Extract hotel details
        hotel_name = selected_hotel.get("name", "Unknown")
        hotel_stars = selected_hotel.get("stars", 3)
        hotel_per_night = selected_hotel.get("price_per_night_usd", 0)
        hotel_distance = selected_hotel.get("distance_to_meeting_km", 
                                           selected_hotel.get("distance_to_business_center_km", "N/A"))
        hotel_amenities = selected_hotel.get("amenities", [])
        
        # Use LLM to analyze quality, value, and appropriateness
        # This runs for ALL selections - no quick accept bypass
        prompt = f"""Analyze this trip planning result for a BUSINESS trip:

SELECTED OPTIONS:
- Flight: {flight_airline} {flight_class} - ${flight_cost}
  - Departure: {flight_departure} → Arrival: {flight_arrival}
  - Duration: {flight_duration:.1f}h
- Hotel: {hotel_name} ({hotel_stars}★) - ${hotel_per_night}/night
  - Location: {hotel_distance}km from meeting/center
  - Amenities: {', '.join(hotel_amenities[:4]) if hotel_amenities else 'None listed'}

BUDGET ANALYSIS:
- Total cost: ${total_cost}
- Budget: ${budget}
- Utilization: {utilization:.1f}%
- Remaining: ${budget_remaining}

QUALITY ASSESSMENT:
- Flight class: {flight_class} (Economy/Business/First)
- Flight duration: {flight_duration:.1f}h
- Hotel stars: {hotel_stars}★ (business standard is 3-4★)
- Violations: {json.dumps(violations, indent=2) if violations else 'None'}

YOUR TASK:
Decide if this selection should be ACCEPTED or if we should NEGOTIATE for better options.

Consider ALL factors:
1. **Quality**: Is this appropriate for business travel?
   - Hotels: 3-4★ is standard, 5★ is premium, <3★ is budget
   - Flights: Business/First for long flights (>6h), Economy acceptable for short
2. **Value**: Are we getting good quality for the price?
   - Is budget being used wisely?
   - Could we get notably better quality within budget?
3. **Budget Utilization**: 
   - IDEAL: 85-95% (maximizes value without waste)
   - ACCEPTABLE: 75-100%
   - UNDER-UTILIZING: <75% (missing upgrade opportunities)
4. **Convenience**: 
   - Flight timing (avoid red-eyes if possible)
   - Hotel location (closer to meeting is better)

Decision criteria:
- ACCEPT if: Good quality + appropriate for business + 75-100% budget utilization
- NEGOTIATE (quality_upgrade) if: Valid but low quality OR under-utilizing budget (<75%)
- NEGOTIATE (cost_reduction) if: Over budget (remaining < 0)
- FAIL if: Way over budget (>120% utilization)

Return ONLY this JSON (no other text):
{{
    "action": "accept" or "negotiate" or "fail",
    "reasoning": "Brief explanation considering quality, value, and convenience (not just budget)",
    "priority": "quality_upgrade" or "cost_reduction" or "none"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)

            action = result.get("action", "accept")
            reasoning = result.get("reasoning", "")
            priority = result.get("priority", "none")

            # Determine next node based on action
            next_node_map = {
                "accept": "check_time",
                "negotiate": "negotiation",
                "fail": "finalize"  # Will finalize with failure status
            }

            decision = {
                "action": action,
                "reasoning": reasoning,
                "priority": priority,
                "next_node": next_node_map.get(action, "check_time")
            }

            self.memory.log_reasoning({
                "step": "policy_decision",
                "thought": reasoning,
                "decision": decision
            })

            return decision

        except Exception as e:
            self._log(f"Policy decision failed: {e}, using fallback")
            # Fallback: accept if valid, negotiate otherwise
            if is_valid:
                return {"action": "accept", "reasoning": "Fallback: valid", "next_node": "check_time"}
            else:
                return {"action": "negotiate", "reasoning": "Fallback: invalid", "next_node": "negotiation"}

    def allocate_budget_dynamically(self, state: Dict, violations: List[Dict]) -> Dict:
        """
        Decide how to split/reallocate budget between flight and hotel.

        Args:
            state: Current state with flight/hotel costs
            violations: List of violations from PolicyAgent

        Returns:
            Dict with budget allocation decisions:
            {
                "target_agent": "flight"/"hotel"/"both",
                "flight_adjustment": <amount>,
                "hotel_adjustment": <amount>,
                "flight_new_max": <new max price>,
                "hotel_new_max": <new max per night>,
                "reasoning": str
            }
        """
        self.memory.increment_agent_call("orchestrator_budget_allocator")

        budget = state.get("budget", 2000)

        # Get current costs
        selected_flight = state.get("selected_flight", {})
        selected_hotel = state.get("selected_hotel", {})

        if isinstance(selected_flight, dict):
            flight_cost = selected_flight.get("price_usd", 0)
        else:
            flight_cost = getattr(selected_flight, "price_usd", 0)

        if isinstance(selected_hotel, dict):
            hotel_cost_per_night = selected_hotel.get("price_per_night_usd", 0)
        else:
            hotel_cost_per_night = getattr(selected_hotel, "price_per_night_usd", 0)

        nights = state.get("nights", 1)
        hotel_total_cost = hotel_cost_per_night * nights

        total_cost = flight_cost + hotel_total_cost

        # Calculate percentages
        flight_pct = (flight_cost / total_cost * 100) if total_cost > 0 else 0
        hotel_pct = (hotel_total_cost / total_cost * 100) if total_cost > 0 else 0

        # Build violation description
        violation_desc = json.dumps(violations, indent=2) if violations else "None"

        prompt = f"""Budget allocation decision:
- Total budget: ${budget}
- Current flight cost: ${flight_cost} ({flight_pct:.1f}% of total)
- Current hotel cost: ${hotel_total_cost} ({hotel_pct:.1f}% of total, ${hotel_cost_per_night}/night × {nights} nights)
- Violations: {violation_desc}

Analyze:
1. Which component is causing the issue?
2. Which has more room to adjust?
3. For upgrades: which component would benefit most from extra spending?
4. For reductions: which component can be reduced without sacrificing critical quality?

Return ONLY this JSON (no other text):
{{
    "target_agent": "flight" or "hotel" or "both",
    "flight_adjustment": <positive for increase, negative for decrease>,
    "hotel_adjustment": <positive for increase, negative for decrease>,
    "flight_new_max": <new maximum flight price>,
    "hotel_new_max": <new maximum hotel price per night>,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)

            self.memory.log_reasoning({
                "step": "budget_allocation",
                "thought": result.get("reasoning", ""),
                "allocation": result
            })

            return result

        except Exception as e:
            self._log(f"Budget allocation failed: {e}, using fallback")
            # Fallback: equal adjustment
            return {
                "target_agent": "both",
                "flight_adjustment": 0,
                "hotel_adjustment": 0,
                "flight_new_max": int(budget * 0.6),
                "hotel_new_max": int(budget * 0.4 / nights),
                "reasoning": "Fallback: equal split"
            }

    def decide_negotiation_target(self, state: Dict) -> Dict:
        """
        Create specific feedback for agents during negotiation.

        Args:
            state: Full state with current options, budget, violations

        Returns:
            Dict with feedback for each agent:
            {
                "flight_feedback": {...},
                "hotel_feedback": {...},
                "reasoning": str
            }
        """
        self.memory.increment_agent_call("orchestrator_negotiation_target")

        budget = state.get("budget", 2000)
        nights = state.get("nights", 1)
        
        # Get current costs
        selected_flight = state.get("selected_flight", {})
        selected_hotel = state.get("selected_hotel", {})
        
        if self.verbose:
            print(f"  [Orchestrator] Reading from state:")
            print(f"    selected_flight: {selected_flight.get('flight_id', 'N/A') if isinstance(selected_flight, dict) else getattr(selected_flight, 'flight_id', 'N/A')}")
            print(f"    selected_hotel: {selected_hotel.get('hotel_id', 'N/A') if isinstance(selected_hotel, dict) else getattr(selected_hotel, 'hotel_id', 'N/A')}")
        
        if isinstance(selected_flight, dict):
            current_flight_cost = selected_flight.get("price_usd", 0)
        else:
            current_flight_cost = getattr(selected_flight, "price_usd", 0)
        
        if isinstance(selected_hotel, dict):
            current_hotel_per_night = selected_hotel.get("price_per_night_usd", 0)
        else:
            current_hotel_per_night = getattr(selected_hotel, "price_per_night_usd", 0)
        
        current_hotel_total = current_hotel_per_night * nights
        current_total = current_flight_cost + current_hotel_total
        
        if self.verbose:
            print(f"    Current costs: Flight=${current_flight_cost}, Hotel=${current_hotel_per_night}/night x {nights} = ${current_hotel_total}, Total=${current_total}")

        # Determine issue type
        budget_remaining = state.get("compliance_status", {}).get("budget_remaining", 0)
        issue_type = "budget_exceeded" if budget_remaining < 0 else "quality_upgrade"
        
        # MATH: Calculate current budget utilization
        current_utilization = (current_total / budget * 100) if budget > 0 else 0
        target_utilization = 0.90  # Target 90% budget usage
        target_utilization_pct = 90.0
        
        # MATH: Query market data to determine realistic maximums
        from_city = state.get("origin", "")
        to_city = state.get("destination", "")
        
        # Get all available flights and hotels
        all_flights = self.flight_loader.search(from_city=from_city, to_city=to_city)
        all_hotels = self.hotel_loader.search(city=to_city)
        
        # Calculate market maximums (pure math)
        if all_flights:
            flight_prices = [f.get('price_usd', 0) for f in all_flights]
            max_flight_price = max(flight_prices)
            min_flight_price = min(flight_prices)
        else:
            max_flight_price = 1500  # Fallback
            min_flight_price = 100
        
        if all_hotels:
            hotel_prices = [h.get('price_per_night_usd', 0) for h in all_hotels]
            max_hotel_price = max(hotel_prices)
            min_hotel_price = min(hotel_prices)
        else:
            max_hotel_price = 800  # Fallback
            min_hotel_price = 50
        
        # Calculate absolute market maximum cost (math only)
        market_max_total = max_flight_price + (max_hotel_price * nights)
        
        # Calculate realistic achievable target (math only)
        realistic_target = min(budget * target_utilization, market_max_total)
        realistic_utilization = (realistic_target / budget * 100) if budget > 0 else 0
        
        # Check if we're already near market maximum (math only)
        distance_to_max = market_max_total - current_total
        near_max_threshold = market_max_total * 0.05  # Within 5% of maximum
        at_market_max = distance_to_max < near_max_threshold
        
        # Get negotiation context
        metrics = state.get("metrics", {})
        negotiation_round = metrics.get("negotiation_rounds", 0)
        
        # Calculate market distribution for better range setting
        flight_prices = [f.get('price_usd', 0) for f in all_flights]
        hotel_prices = [h.get('price_per_night_usd', 0) for h in all_hotels]
        
        # Percentiles for market understanding
        import statistics
        flight_p25 = statistics.quantiles(flight_prices, n=4)[0] if len(flight_prices) > 1 else min_flight_price
        flight_p50 = statistics.median(flight_prices) if len(flight_prices) > 0 else (min_flight_price + max_flight_price) / 2
        flight_p75 = statistics.quantiles(flight_prices, n=4)[2] if len(flight_prices) > 1 else max_flight_price
        
        hotel_p25 = statistics.quantiles(hotel_prices, n=4)[0] if len(hotel_prices) > 1 else min_hotel_price
        hotel_p50 = statistics.median(hotel_prices) if len(hotel_prices) > 0 else (min_hotel_price + max_hotel_price) / 2
        hotel_p75 = statistics.quantiles(hotel_prices, n=4)[2] if len(hotel_prices) > 1 else max_hotel_price
        
        # LLM REASONING: Analyze situation and decide target prices
        prompt = f"""You are the Trip Orchestrator coordinating a business trip booking system.

CURRENT SITUATION:
- Budget: ${budget}
- Current Total Cost: ${current_total} ({current_utilization:.1f}% utilization)
- Current Flight: ${current_flight_cost}
- Current Hotel: ${current_hotel_per_night}/night × {nights} nights = ${current_hotel_total}
- Issue: {issue_type}
- Negotiation Round: {negotiation_round}/7

MARKET ANALYSIS:
- Available flights: {len(all_flights)} options
  - Range: ${min_flight_price}-${max_flight_price}
  - Distribution: 25th%=${flight_p25:.0f}, Median=${flight_p50:.0f}, 75th%=${flight_p75:.0f}
- Available hotels: {len(all_hotels)} options  
  - Range: ${min_hotel_price}-${max_hotel_price}/night
  - Distribution: 25th%=${hotel_p25:.0f}, Median=${hotel_p50:.0f}, 75th%=${hotel_p75:.0f}
- Market Maximum: ${market_max_total} (flight ${max_flight_price} + hotel ${max_hotel_price}×{nights})
- Distance to Market Max: ${distance_to_max}
- At/near market max: {"YES" if at_market_max else "NO"}

TARGET:
- Ideal: {target_utilization_pct:.0f}% budget utilization (${budget * target_utilization:.0f})
- Realistic: ${realistic_target:.0f} ({realistic_utilization:.1f}% utilization)

AGENT BEHAVIOR:
- Each agent returns EXACTLY ONE option within your target range
- **CRITICAL**: Agents can ONLY select from options that exist in the market!
- Your ranges MUST overlap with the distribution shown above

YOUR TASK:
Set target price ranges that agents can actually find options in.

For QUALITY_UPGRADE:
- Check if upgrading is possible (current < market max)
- If YES: Set ranges between current and higher percentiles
  - Example: If current flight=$350, median=$500 → Set $380-$550
- If NO: Recommend accepting (already at market max)
- **VERIFY ranges overlap with market distribution!**

For BUDGET_EXCEEDED:
- Set ranges between min and current to reduce costs
  - Example: If current hotel=$400, min=$200 → Set $200-$350

EXAMPLES OF GOOD RANGES:
- Current flight $350, want upgrade → ${int(flight_p50*0.9)}-${int(flight_p75*1.1)} (overlaps median-75th%)
- Current hotel $200, want upgrade → ${int(hotel_p50*0.9)}-${int(hotel_p75*1.1)}/night (overlaps median-75th%)
- Over budget, need cheaper → Use min to current*0.8

Return JSON:
{{
  "reasoning": "Brief strategy considering market distribution",
  "should_accept_current": true/false,
  "flight_target_min": 150,
  "flight_target_max": 500,
  "hotel_target_min": 100,
  "hotel_target_max": 300
}}

CRITICAL: Ensure your ranges OVERLAP with available options (check the distribution)!"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            reasoning = result.get("reasoning", "Allocating budget based on market analysis")
            should_accept = result.get("should_accept_current", False)
            
            # If LLM says to accept, set narrow ranges around current prices
            if should_accept:
                if self.verbose:
                    print(f"  [Orchestrator] LLM decision: Accept current (at/near market max)")
                flight_min = int(current_flight_cost * 0.95)
                flight_max = int(current_flight_cost * 1.05)
                hotel_min = int(current_hotel_per_night * 0.95)
                hotel_max = int(current_hotel_per_night * 1.05)
            else:
                flight_min = max(min_flight_price, min(result.get("flight_target_min", 150), max_flight_price))
                flight_max = max(flight_min, min(result.get("flight_target_max", 500), max_flight_price))
                hotel_min = max(min_hotel_price, min(result.get("hotel_target_min", 100), max_hotel_price))
                hotel_max = max(hotel_min, min(result.get("hotel_target_max", 300), max_hotel_price))
            
        except Exception as e:
            if self.verbose:
                print(f"  [Orchestrator] LLM reasoning failed: {e}, using fallback")
            
            # Fallback to simple progressive ranges
            reasoning = "Fallback: Using progressive price increases"
            should_accept = at_market_max  # Use math-based decision
            
            if should_accept:
                flight_min = int(current_flight_cost * 0.95)
                flight_max = int(current_flight_cost * 1.05)
                hotel_min = int(current_hotel_per_night * 0.95)
                hotel_max = int(current_hotel_per_night * 1.05)
            elif issue_type == "quality_upgrade":
                flight_min = int(current_flight_cost * 1.1)
                flight_max = min(int(current_flight_cost * 1.5), max_flight_price)
                hotel_min = int(current_hotel_per_night * 1.2)
                hotel_max = min(int(current_hotel_per_night * 2.0), max_hotel_price)
            else:
                flight_min = min_flight_price
                flight_max = int(current_flight_cost * 0.8)
                hotel_min = min_hotel_price
                hotel_max = int(current_hotel_per_night * 0.8)
        
        # Build feedback structure
        feedback = {
            "reasoning": reasoning,
            "at_market_max": at_market_max or should_accept
        }
        
        feedback["flight_feedback"] = {
            "issue": issue_type,
            "target_price_min": flight_min,
            "target_price_max": flight_max,
            "from_city": from_city,
            "to_city": to_city,
            "reasoning": f"Target ${flight_min}-${flight_max}",
            "re_search": True
        }
        
        feedback["hotel_feedback"] = {
            "issue": issue_type,
            "target_price_min": hotel_min,
            "target_price_max": hotel_max,
            "city": to_city,
            "reasoning": f"Target ${hotel_min}-${hotel_max}/night",
            "re_search": True
        }

        self.memory.log_reasoning({
            "step": "negotiation_target",
            "thought": f"Target ${realistic_target:.0f} ({realistic_utilization:.0f}% of budget), market max: ${market_max_total}",
            "feedback": feedback
        })

        return feedback

    def decide_next_node(self, current_phase: str, state: Dict) -> str:
        """
        Decide routing based on current workflow phase.

        Args:
            current_phase: String like "after_policy", "after_time", "negotiation_check"
            state: Full state

        Returns:
            String node name: "negotiation", "check_time", "select_options", "finalize", etc.
        """
        self.memory.increment_agent_call("orchestrator_router")

        if current_phase == "after_policy":
            # After policy check routing
            # PRIORITY 1: Respect the LLM's comprehensive quality review
            policy_decision = state.get("policy_decision", {})
            llm_action = policy_decision.get("action", None)
            llm_reasoning = policy_decision.get("reasoning", "")
            
            compliance = state.get("compliance_status", {})
            budget_remaining = compliance.get("budget_remaining", 0)
            is_valid = compliance.get("is_valid", False)
            total_cost = compliance.get("total_cost", 0)
            budget = state.get("budget", 2000)

            # Calculate utilization
            utilization = (total_cost / budget * 100) if budget > 0 else 0

            # Check negotiation state
            metrics = state.get("metrics", {})
            negotiation_round = metrics.get("negotiation_rounds", 0)
            max_negotiation_rounds = 7

            # Enhanced stagnation detection
            prev_cost = state.get("previous_total_cost", 0)
            max_cost_reached = metrics.get("max_cost_reached", 0)
            
            # Update max cost if current is higher
            if total_cost > max_cost_reached:
                metrics["max_cost_reached"] = total_cost
                max_cost_reached = total_cost
            
            # Detect stagnation or regression
            cost_change = abs(total_cost - prev_cost) if prev_cost > 0 else float('inf')
            cost_decreased = total_cost < prev_cost if prev_cost > 0 else False
            cost_oscillating = total_cost < max_cost_reached and negotiation_round > 1
            
            # Stagnation: cost changed less than $50 OR decreased OR oscillating
            stagnated = (cost_change < 50 and negotiation_round > 1) or cost_decreased or cost_oscillating

            # CRITICAL OVERRIDES (safety checks that override LLM)
            if negotiation_round >= max_negotiation_rounds:
                self._log("Max negotiation rounds reached, moving to time check")
                return "check_time"

            if stagnated:
                if cost_decreased:
                    self._log(f"Negotiation regressed (${total_cost} < ${prev_cost}), accepting current")
                elif cost_oscillating:
                    self._log(f"Costs oscillating (current ${total_cost} < max ${max_cost_reached}), accepting current")
                else:
                    self._log(f"Negotiation stagnated (change: ${cost_change}), moving to time check")
                return "check_time"

            # RESPECT LLM DECISION if available
            if self.verbose:
                print(f"  [Orchestrator Routing] LLM action: {llm_action}, reasoning: {llm_reasoning[:50] if llm_reasoning else 'N/A'}")
            
            if llm_action:
                if llm_action == "negotiate":
                    self._log(f"LLM decided to negotiate: {llm_reasoning[:80]}")
                    return "negotiation"
                elif llm_action == "fail":
                    self._log(f"LLM decided to fail: {llm_reasoning[:80]}")
                    return "finalize"
                else:  # accept
                    self._log(f"LLM decided to accept: {llm_reasoning[:80]}")
                    return "check_time"
            
            # FALLBACK (if LLM decision not available - shouldn't happen)
            if self.verbose:
                print(f"  [Orchestrator Routing] WARNING: No LLM decision found, using fallback")
            
            if budget_remaining < 0:
                self._log(f"Over budget by ${-budget_remaining}, negotiating")
                return "negotiation"

            if is_valid and utilization < 75:
                self._log(f"Under-utilizing budget ({utilization:.1f}%), upgrading quality")
                return "negotiation"

            # Default: move to time check
            return "check_time"

        elif current_phase == "after_time":
            # After time check routing - enhanced to handle timeline conflicts
            time_conflicts = state.get("time_conflicts", [])
            time_feedback_count = state.get("time_feedback_count", 0)
            max_time_feedback = state.get("max_time_feedback_rounds", 2)

            # Check severity
            severe_conflicts = [c for c in time_conflicts if c.get("severity") == "error"]
            
            if self.verbose:
                print(f"  [Orchestrator] Time routing: {len(time_conflicts)} conflicts, {len(severe_conflicts)} severe")
                print(f"  [Orchestrator] Time feedback count: {time_feedback_count}/{max_time_feedback}")

            if severe_conflicts and time_feedback_count < max_time_feedback:
                self._log(f"Severe time conflicts detected, sending feedback (round {time_feedback_count + 1})")
                return "time_policy_feedback"

            # Otherwise, proceed to selection
            return "select_options"

        elif current_phase == "after_time_feedback":
            # After time feedback, always go back to check time
            return "check_time"

        elif current_phase == "negotiation_check":
            # After negotiation, always verify with policy
            return "check_policy"

        else:
            # Unknown phase, default to check_policy
            self._log(f"Unknown phase: {current_phase}, defaulting to check_policy")
            return "check_policy"

