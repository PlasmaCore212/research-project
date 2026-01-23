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
    
    def __init__(self, model_name: str = "mistral-small", verbose: bool = True):
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
        
        # Selection prompt
        prompt = f"""You are selecting a flight and hotel combination for a business trip.

BUDGET: ${budget} total (flight + {nights} nights hotel)

AVAILABLE FLIGHTS:
{json.dumps(flights_data, indent=2)}

AVAILABLE HOTELS:
{json.dumps(hotels_data, indent=2)}

YOUR TASK:
Analyze the options and select one flight and one hotel.

Consider:
- Budget constraints
- Quality, timing, location
- Tradeoffs between factors

Return JSON:
{{
    "reasoning_steps": ["Step 1: ...", "Step 2: ...", ...],
    "flight_id": "selected flight ID",
    "hotel_id": "selected hotel ID",
    "total_cost": calculated_total,
    "justification": "Your reasoning"
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

    def handle_policy_result(self, validation_result: Dict, state: Dict, 
                            selected_flight: Dict = None, selected_hotel: Dict = None) -> Dict:
        """
        Interpret PolicyAgent's validation and decide what to do next.

        Args:
            validation_result: Dict from PolicyAgent with:
                {"is_valid": bool, "violations": [...], "total_cost": float, "budget_remaining": float}
            state: Current TripPlanningState
            selected_flight: Selected flight dict (passed directly to avoid state timing issues)
            selected_hotel: Selected hotel dict (passed directly to avoid state timing issues)

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
        # Use parameters if provided, otherwise fall back to state (for negotiation rounds)
        if selected_flight is None:
            selected_flight = state.get("selected_flight", {})
        if selected_hotel is None:
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

        # Calculate nights and hotel total from state
        from orchestrator.helpers import calculate_nights
        nights = calculate_nights(state)
        hotel_total_cost = hotel_per_night * nights

        # Check if negotiation already ran
        metrics = state.get("metrics", {})
        negotiation_rounds = metrics.get("negotiation_rounds", 0)
        
        # HARD STOP: Force acceptance after max negotiation rounds (7)
        # This prevents infinite loops where LLM keeps saying "negotiate"
        MAX_NEGOTIATION_ROUNDS = 7
        if negotiation_rounds >= MAX_NEGOTIATION_ROUNDS:
            self._log(f"Max negotiation rounds ({negotiation_rounds}) reached - using best option seen")

            # Check if we have a better option from a previous round
            best_option = state.get("best_option_seen", {})
            if best_option and best_option.get("is_valid"):
                # Use the best valid option from any round
                self._log(f"Best valid option found in round {best_option.get('round', 0)}: ${best_option.get('total_cost', 0)} ({best_option.get('utilization', 0):.1f}% utilization)")
                return {
                    "action": "use_best",
                    "reasoning": f"Maximum negotiation rounds ({negotiation_rounds}/{MAX_NEGOTIATION_ROUNDS}) reached. Using best valid option from round {best_option.get('round', 0)} with {best_option.get('utilization', 0):.1f}% budget utilization.",
                    "priority": "none",
                    "next_node": "check_time",
                    "use_best_option": True
                }
            else:
                # No valid option found in any round - use current as fallback
                self._log(f"No valid option found in any round - accepting current as fallback")
                return {
                    "action": "accept",
                    "reasoning": f"Maximum negotiation rounds ({negotiation_rounds}/{MAX_NEGOTIATION_ROUNDS}) reached. No valid option found in any round, accepting current booking as fallback.",
                    "priority": "none",
                    "next_node": "check_time"
                }
        
        # Use LLM to analyze quality, value, and appropriateness
        # Pre-calculate additional metrics for the LLM
        flight_pct = (flight_cost / total_cost * 100) if total_cost > 0 else 0
        hotel_pct = (hotel_total_cost / total_cost * 100) if total_cost > 0 else 0
        
        prompt = f"""Evaluate this business trip booking.

BOOKING:
Flight: {flight_airline} {flight_class} - ${flight_cost}
Hotel: {hotel_name} ({hotel_stars}★) - ${hotel_per_night}/night × {nights} = ${hotel_total_cost}

BUDGET:
Total cost: ${total_cost}
Budget: ${budget}
Utilization: {utilization:.1f}%
Remaining: ${budget_remaining}
Violations: {json.dumps(violations) if violations else 'None'}

Decide:
- "accept": Booking is appropriate
- "negotiate": Changes needed
- "fail": Cannot work

If negotiating, specify priority:
- "quality_upgrade": Budget under-utilized
- "cost_reduction": Over budget or too expensive

CRITICAL: Must use at least 75% of budget - don't accept if utilization < 75%!

Return JSON:
{{
    "action": "accept/negotiate/fail",
    "reasoning": "Brief analysis",
    "priority": "quality_upgrade/cost_reduction/none"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)

            action = result.get("action", "accept")
            reasoning = result.get("reasoning", "")
            priority = result.get("priority", "none")

            # HARDCODED RULE: Minimum budget utilization threshold
            MIN_BUDGET_UTILIZATION = 75.0
            
            if action == "accept" and utilization < MIN_BUDGET_UTILIZATION:
                if self.verbose:
                    print(f"  [Orchestrator] ⚠️ Budget utilization too low ({utilization:.1f}% < {MIN_BUDGET_UTILIZATION}%)")
                    print(f"  [Orchestrator] Overriding 'accept' → 'negotiate' for better value")
                action = "negotiate"
                priority = "quality_upgrade"
                reasoning = f"Budget under-utilized ({utilization:.1f}% < 75%). {reasoning}"

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
        Create specific feedback for agents during negotiation with TRADEOFF support.

        Enables sophisticated budget optimization by independently deciding a strategy
        for each component (flight and hotel). Supports tradeoffs like:
        - Downgrade flight, upgrade hotel
        - Upgrade flight, maintain hotel
        - Maintain flight, upgrade hotel

        Each component gets a strategy: "increase", "decrease", or "maintain"
        Only components with "increase" or "decrease" receive feedback.

        Args:
            state: Full state with current options, budget, violations

        Returns:
            Dict with feedback for agents that need to change:
            {
                "flight_feedback": {...},   # Only if flight_strategy != "maintain"
                "hotel_feedback": {...},    # Only if hotel_strategy != "maintain"
                "reasoning": str,
                "flight_strategy": "increase"/"decrease"/"maintain",
                "hotel_strategy": "increase"/"decrease"/"maintain"
            }
        """
        self.memory.increment_agent_call("orchestrator_negotiation_target")

        from orchestrator.helpers import calculate_nights

        budget = state.get("budget", 2000)
        nights = calculate_nights(state)
        
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
        
        if issue_type == "quality_upgrade":
            target_utilization = 0.90  # 90% for upgrades
        elif issue_type == "budget_exceeded":
            target_utilization = 0.85  # 85% to reduce cost
        else:
            target_utilization = 0.80  # 80% for balanced scenarios

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

        # Get flight and hotel details for quality assessment
        selected_flight = state.get("selected_flight", {})
        selected_hotel = state.get("selected_hotel", {})
        flight_class = selected_flight.get("class", "Economy") if isinstance(selected_flight, dict) else getattr(selected_flight, "class", "Economy")
        flight_duration = selected_flight.get("duration_hours", 0) if isinstance(selected_flight, dict) else getattr(selected_flight, "duration_hours", 0)
        hotel_stars = selected_hotel.get("stars", 3) if isinstance(selected_hotel, dict) else getattr(selected_hotel, "stars", 3)

        # Calculate how close we are to market min/max
        flight_distance_from_min = current_flight_cost - min_flight_price
        flight_distance_from_max = max_flight_price - current_flight_cost
        hotel_distance_from_min = current_hotel_per_night - min_hotel_price
        hotel_distance_from_max = max_hotel_price - current_hotel_per_night

        # LLM REASONING: Decide STRATEGY for each component independently
        # Pre-calculate comprehensive data for the LLM to reason about
        
        # Calculate budget distribution between flight and hotel
        flight_budget_pct = (current_flight_cost / current_total * 100) if current_total > 0 else 0
        hotel_budget_pct = (current_hotel_total / current_total * 100) if current_total > 0 else 0
        
        # Calculate how far each component is from its market range
        flight_position_in_market = ((current_flight_cost - min_flight_price) / (max_flight_price - min_flight_price) * 100) if max_flight_price > min_flight_price else 50
        hotel_position_in_market = ((current_hotel_per_night - min_hotel_price) / (max_hotel_price - min_hotel_price) * 100) if max_hotel_price > min_hotel_price else 50
        
        # Calculate remaining budget and what it could buy
        remaining_budget = budget - current_total

        # LLM-based quality balance assessment (no hardcoded scores)
        balance_assessment = self._assess_quality_balance(
            flight_class=flight_class,
            flight_duration=flight_duration,
            flight_cost=current_flight_cost,
            hotel_stars=hotel_stars,
            hotel_cost=current_hotel_per_night,
            nights=nights,
            budget=budget,
            current_total=current_total
        )

        # Extract LLM's assessment
        is_imbalanced = balance_assessment.get("is_imbalanced", False)
        balance_reasoning = balance_assessment.get("reasoning", "Quality appears balanced")
        quality_imbalance = balance_assessment.get("severity", 0)  # For metrics only
        
        # ADVANCED PROMPT ENGINEERING: Predict outcomes of each strategy
        # This helps LLM understand trade-offs before deciding
        
        # Estimate what would happen with different strategies
        estimated_business_flight = min(max_flight_price, current_flight_cost * 3)  # Rough estimate
        estimated_mid_hotel = (min_hotel_price + max_hotel_price) / 2
        
        prompt = f"""You are a business trip booking coordinator. Analyze this booking and decide on the best strategy.

=== CURRENT BOOKING ===
Budget: ${budget}
Current Cost: ${current_total} ({current_utilization:.1f}% utilization)
Remaining: ${remaining_budget}

Flight: ${current_flight_cost} ({flight_class}, {flight_duration:.1f}h)
  Market position: {flight_position_in_market:.0f}% of range (${min_flight_price}-${max_flight_price})
  Adjustment room: {flight_position_in_market:.0f}% to decrease, {100 - flight_position_in_market:.0f}% to increase

Hotel: ${current_hotel_per_night}/night × {nights} nights = ${current_hotel_total} ({hotel_stars}★)
  Market position: {hotel_position_in_market:.0f}% of range (${min_hotel_price}-${max_hotel_price}/night)
  Adjustment room: {hotel_position_in_market:.0f}% to decrease, {100 - hotel_position_in_market:.0f}% to increase

Quality Balance: {"⚠️ IMBALANCED" if is_imbalanced else "✓ Balanced"}
Assessment: {balance_reasoning}

=== CONSTRAINTS ===
✓ MUST use ≥75% of budget (currently: {current_utilization:.1f}%)
✓ Components near min/max (position <10% or >90%) cannot be adjusted further
✓ Round {negotiation_round}/7 - prefer simpler strategies in early rounds

=== DECISION FRAMEWORK ===

Think through this systematically:

1. What is the primary issue?
   - Is budget utilization the problem? (< 75%)
   - Is quality imbalance the problem? (gap > 30)
   - Are both issues present?

2. What are the constraints?
   - Can flight be adjusted? (position: {flight_position_in_market:.0f}%)
   - Can hotel be adjusted? (position: {hotel_position_in_market:.0f}%)
   - What's the remaining budget? (${remaining_budget})

3. How do the trade-offs compare?
   - Which predicted outcome best satisfies BOTH constraints?
   - If no outcome satisfies both, which constraint is more critical?
     * For business trips: Budget utilization ≥75% is MANDATORY
     * Quality balance is DESIRABLE but not mandatory
a
4. What is the best strategy?
   - Choose the strategy that maximizes budget utilization while keeping quality gap reasonable
   - Avoid strategies that fix one problem but create another

=== FEW-SHOT EXAMPLES ===

Example 1: High utilization (95%), imbalanced quality (gap: +60)
  Reasoning: "Budget utilization is excellent. Quality imbalance exists but fixing it would drop utilization below 75%. Priority: maintain high utilization."
  Decision: maintain both → Accept current booking

Example 2: Low utilization (40%), balanced quality (gap: 5)
  Reasoning: "Budget severely under-utilized. Quality is balanced. Need to upgrade both components proportionally to reach 75%+ utilization."
  Decision: increase both → Target 80% utilization

Example 3: Good utilization (80%), imbalanced quality (gap: +50)
  Reasoning: "Utilization is good. Quality imbalanced (5★ hotel, Economy flight). Can upgrade flight moderately without dropping below 75%. Hotel stays same."
  Decision: increase flight, maintain hotel → Slight increase to 90% utilization acceptable

Example 4: Over budget (110%), both components mid-range
  Reasoning: "Over budget by 10%. Both flight and hotel are at 50% of their market ranges. Need to reduce both proportionally to reach 95% utilization."
  Decision: decrease both → Target 95% utilization

Example 5: Under budget (60%), flight at minimum (5%), hotel mid-range (50%)
  Reasoning: "Severely under-utilizing budget. Flight already at minimum (can't decrease further). Hotel has room to increase. Upgrade hotel significantly."
  Decision: maintain flight, increase hotel → Target 80% utilization

=== YOUR TASK ===

Follow the decision framework above. Think step-by-step.

Return JSON:
{{
  "reasoning": "Step-by-step analysis following the framework above",
  "should_accept_current": true/false,
  "flight_strategy": "maintain/increase/decrease",
  "hotel_strategy": "maintain/increase/decrease",
  "flight_target_min": <number - total flight price>,
  "flight_target_max": <number - total flight price>,
  "hotel_target_min": <number - PRICE PER NIGHT (will be × {nights} nights for total)>,
  "hotel_target_max": <number - PRICE PER NIGHT (will be × {nights} nights for total)>
}}

CRITICAL VALIDATION RULES:
- hotel_target_min and hotel_target_max are PER NIGHT prices
- Total hotel cost will be: hotel_target_max × {nights} nights
- MUST satisfy: flight_target_max + (hotel_target_max × {nights}) ≤ ${budget * 1.1}
- Example: If hotel_target_max = $400/night and nights = 3, total hotel = $1200

Double-check your math before returning!"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)

            reasoning = result.get("reasoning", "Allocating budget based on market analysis")
            should_accept = result.get("should_accept_current", False)
            flight_strategy = result.get("flight_strategy", "maintain")  # "increase", "decrease", or "maintain"
            hotel_strategy = result.get("hotel_strategy", "maintain")

            if self.verbose:
                print(f"  [Orchestrator] LLM decision: Flight={flight_strategy}, Hotel={hotel_strategy}, Accept={should_accept}")

            # HARDCODED RULE: Minimum budget utilization threshold
            MIN_BUDGET_UTILIZATION = 80.0  # Must use at least 75% of budget
            
            if should_accept and current_utilization < MIN_BUDGET_UTILIZATION:
                if self.verbose:
                    print(f"  [Orchestrator] ⚠️ Budget utilization too low ({current_utilization:.1f}% < {MIN_BUDGET_UTILIZATION}%)")
                    print(f"  [Orchestrator] Overriding 'accept' → continuing negotiation for better value")
                should_accept = False
                # If LLM wanted to accept but budget is low, upgrade both
                if flight_strategy == "maintain" and hotel_strategy == "maintain":
                    flight_strategy = "increase"
                    hotel_strategy = "increase"

            # If LLM says to accept, maintain both
            if should_accept:
                if self.verbose:
                    print(f"  [Orchestrator] Accepting current booking (utilization: {current_utilization:.1f}%)")
                flight_strategy = "maintain"
                hotel_strategy = "maintain"

            # Use LLM's target ranges with validation
            # If strategy is "maintain", LLM should provide current prices as targets
            flight_min = result.get("flight_target_min", int(current_flight_cost))
            flight_max = result.get("flight_target_max", int(current_flight_cost))
            hotel_min = result.get("hotel_target_min", int(current_hotel_per_night))
            hotel_max = result.get("hotel_target_max", int(current_hotel_per_night))

            # SMART VALIDATION: Override impossible strategies
            # If already at/near minimum, can't decrease further
            if flight_strategy == "decrease" and flight_position_in_market < 10:
                if self.verbose:
                    print(f"  [Orchestrator] Flight already at minimum ({flight_position_in_market:.0f}% position) - changing 'decrease' to 'maintain'")
                flight_strategy = "maintain"
                flight_min = int(current_flight_cost)
                flight_max = int(current_flight_cost)
            
            if hotel_strategy == "decrease" and hotel_position_in_market < 10:
                if self.verbose:
                    print(f"  [Orchestrator] Hotel already at minimum ({hotel_position_in_market:.0f}% position) - changing 'decrease' to 'maintain'")
                hotel_strategy = "maintain"
                hotel_min = int(current_hotel_per_night)
                hotel_max = int(current_hotel_per_night)
            
            # If already at/near maximum, can't increase further
            if flight_strategy == "increase" and flight_position_in_market > 90:
                if self.verbose:
                    print(f"  [Orchestrator] Flight already at maximum ({flight_position_in_market:.0f}% position) - changing 'increase' to 'maintain'")
                flight_strategy = "maintain"
                flight_min = int(current_flight_cost)
                flight_max = int(current_flight_cost)
            
            if hotel_strategy == "increase" and hotel_position_in_market > 90:
                if self.verbose:
                    print(f"  [Orchestrator] Hotel already at maximum ({hotel_position_in_market:.0f}% position) - changing 'increase' to 'maintain'")
                hotel_strategy = "maintain"
                hotel_min = int(current_hotel_per_night)
                hotel_max = int(current_hotel_per_night)

            # Validate target ranges to prevent unrealistic values
            # For "decrease" strategy, don't go below 30% of current price (prevents "insanely cheap")
            # For "increase" strategy, don't go above 200% of current price
            if flight_strategy == "decrease":
                flight_min = max(int(current_flight_cost * 0.3), flight_min)
                flight_max = max(int(current_flight_cost * 0.3), flight_max)
            elif flight_strategy == "increase":
                flight_max = min(int(current_flight_cost * 2.0), flight_max)

            if hotel_strategy == "decrease":
                hotel_min = max(int(current_hotel_per_night * 0.3), hotel_min)
                hotel_max = max(int(current_hotel_per_night * 0.3), hotel_max)
            elif hotel_strategy == "increase":
                hotel_max = min(int(current_hotel_per_night * 2.0), hotel_max)

            # Clamp to market bounds
            flight_min = max(min_flight_price, min(flight_min, max_flight_price))
            flight_max = max(flight_min, min(flight_max, max_flight_price))
            hotel_min = max(min_hotel_price, min(hotel_min, max_hotel_price))
            hotel_max = max(hotel_min, min(hotel_max, max_hotel_price))
            
            # CRITICAL: Validate that target ranges fit within budget
            # This prevents the LLM from suggesting hotel prices that result in over-budget totals
            flight_min, flight_max, hotel_min, hotel_max = self._validate_target_ranges(
                flight_min, flight_max, hotel_min, hotel_max, nights, budget
            )
            
            
            # LLM decides target ranges autonomously - no hardcoded enforcement
            
        except Exception as e:
            if self.verbose:
                print(f"  [Orchestrator] LLM reasoning failed: {e}, using fallback")

            # Simple fallback logic
            reasoning = "Fallback: Simple budget-based strategy"
            should_accept = at_market_max

            if should_accept:
                # Near market maximum - maintain both
                flight_strategy = "maintain"
                hotel_strategy = "maintain"
                flight_min = int(current_flight_cost * 0.95)
                flight_max = int(current_flight_cost * 1.05)
                hotel_min = int(current_hotel_per_night * 0.95)
                hotel_max = int(current_hotel_per_night * 1.05)
            elif current_utilization < 75:
                # Under-utilizing budget - consider upgrades
                flight_strategy = "increase"
                hotel_strategy = "increase"
                flight_min = int(current_flight_cost)
                flight_max = int(flight_p75)
                hotel_min = int(current_hotel_per_night)
                hotel_max = int(hotel_p75)
            elif current_utilization > 100:
                # Over budget - reduce costs
                flight_strategy = "decrease"
                hotel_strategy = "decrease"
                flight_min = int(flight_p25)
                flight_max = int(current_flight_cost * 0.9)
                hotel_min = int(hotel_p25)
                hotel_max = int(current_hotel_per_night * 0.9)
            else:
                # Balanced - maintain both
                flight_strategy = "maintain"
                hotel_strategy = "maintain"
                flight_min = int(current_flight_cost * 0.95)
                flight_max = int(current_flight_cost * 1.05)
                hotel_min = int(current_hotel_per_night * 0.95)
                hotel_max = int(current_hotel_per_night * 1.05)

        # Determine issue type for each component based on strategy
        def get_issue_type(strategy: str) -> str:
            if strategy == "increase":
                return "quality_upgrade"
            elif strategy == "decrease":
                return "budget_exceeded"
            else:
                return "maintain"

        # Build feedback structure - ONLY include feedback for agents that should change
        # If should_accept is True, set at_market_max to True to signal termination
        feedback = {
            "reasoning": reasoning,
            "at_market_max": at_market_max or should_accept,
            "flight_strategy": flight_strategy,
            "hotel_strategy": hotel_strategy,
            "should_accept": should_accept  # Add explicit flag for termination
        }

        # Only add flight_feedback if flight should increase or decrease (not maintain)
        if flight_strategy in ["increase", "decrease"]:
            feedback["flight_feedback"] = {
                "issue": get_issue_type(flight_strategy),
                "target_price_min": flight_min,
                "target_price_max": flight_max,
                "from_city": from_city,
                "to_city": to_city,
                "reasoning": f"Strategy: {flight_strategy} → Target ${flight_min}-${flight_max}",
                "re_search": True
            }

        # Only add hotel_feedback if hotel should increase or decrease (not maintain)
        if hotel_strategy in ["increase", "decrease"]:
            feedback["hotel_feedback"] = {
                "issue": get_issue_type(hotel_strategy),
                "target_price_min": hotel_min,
                "target_price_max": hotel_max,
                "city": to_city,
                "reasoning": f"Strategy: {hotel_strategy} → Target ${hotel_min}-${hotel_max}/night",
                "re_search": True
            }

        self.memory.log_reasoning({
            "step": "negotiation_target",
            "thought": f"Target ${realistic_target:.0f} ({realistic_utilization:.0f}% of budget), market max: ${market_max_total}",
            "feedback": feedback
        })

        return feedback
    
    def _validate_target_ranges(self, flight_min: int, flight_max: int, 
                                hotel_min: int, hotel_max: int, 
                                nights: int, budget: float) -> tuple:
        """
        Validate and adjust target ranges to ensure budget feasibility.
        
        CRITICAL: hotel_min and hotel_max are PER NIGHT prices.
        Total hotel cost = hotel_max * nights
        
        Args:
            flight_min, flight_max: Flight price range (total)
            hotel_min, hotel_max: Hotel price range (PER NIGHT)
            nights: Number of nights
            budget: Total budget
            
        Returns:
            Tuple of (flight_min, flight_max, hotel_min, hotel_max) after validation
        """
        # Calculate worst-case total cost
        max_total = flight_max + (hotel_max * nights)
        
        if max_total > budget * 1.1:  # Allow 10% overage for flexibility
            if self.verbose:
                print(f"  [Orchestrator] ⚠️  Target ranges exceed budget:")
                print(f"    Flight: ${flight_max}")
                print(f"    Hotel: ${hotel_max}/night × {nights} nights = ${hotel_max * nights}")
                print(f"    Total: ${max_total} > Budget: ${budget}")
            
            # Proportionally reduce to fit budget
            scale_factor = (budget * 0.95) / max_total  # Target 95% to leave margin
            flight_max = int(flight_max * scale_factor)
            flight_min = int(flight_min * scale_factor)
            hotel_max = int(hotel_max * scale_factor)
            hotel_min = int(hotel_min * scale_factor)
            
            if self.verbose:
                print(f"  [Orchestrator] ✓ Adjusted ranges:")
                print(f"    Flight: ${flight_min}-${flight_max}")
                print(f"    Hotel: ${hotel_min}-${hotel_max}/night × {nights} = ${hotel_min * nights}-${hotel_max * nights}")
                print(f"    New max total: ${flight_max + (hotel_max * nights)}")
        
        return flight_min, flight_max, hotel_min, hotel_max

    def _assess_quality_balance(
        self,
        flight_class: str,
        flight_duration: float,
        flight_cost: float,
        hotel_stars: int,
        hotel_cost: float,
        nights: int,
        budget: float,
        current_total: float
    ) -> Dict[str, Any]:
        """
        Use LLM to assess quality balance contextually without hardcoded scores.
        
        Returns:
            {
                "is_imbalanced": bool,
                "severity": int (0-100, for metrics),
                "reasoning": str,
                "recommendation": "accept"/"rebalance_flight"/"rebalance_hotel"/"both"
            }
        """
        
        utilization = (current_total / budget * 100) if budget > 0 else 0
        
        prompt = f"""Assess the quality balance of this business trip booking.

    BOOKING:
    Flight: {flight_class} class, {flight_duration:.1f}h duration, ${flight_cost}
    Hotel: {hotel_stars}★, ${hotel_cost}/night × {nights} nights = ${hotel_cost * nights}

    CONTEXT:
    Total: ${current_total} ({utilization:.1f}% of ${budget} budget)

    TASK:
    Evaluate if flight and hotel are reasonably balanced in quality for business travel.
    Consider:
    - Is Economy appropriate for this flight duration?
    - Do hotel stars match the trip's budget tier?
    - Are they roughly equivalent in comfort/quality?

    A quality imbalance means one component is significantly better/worse than the other
    (e.g., Economy on 8h flight + 5★ luxury hotel, or First Class + 2★ motel).

    Balanced examples:
    - Economy 2h flight + 3★ hotel (budget trip)
    - Business 6h flight + 4★ hotel (standard business)
    - First Class long-haul + 5★ hotel (premium)

    Return JSON:
    {{
        "is_imbalanced": true/false,
        "severity": <0-100>,
        "reasoning": "Brief explanation",
        "recommendation": "accept"/"rebalance_flight"/"rebalance_hotel"/"both"
    }}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            if self.verbose:
                print(f"  [Orchestrator] Quality assessment failed: {e}, assuming balanced")
            return {
                "is_imbalanced": False,
                "severity": 0,
                "reasoning": "Fallback: assuming balanced",
                "recommendation": "accept"
            }

    def decide_next_node(self, current_phase: str, state: Dict) -> str:
        """
        Decide routing based on current workflow phase.

        Args:
            current_phase: String like "after_time", "after_time_feedback", "negotiation_check"
            state: Full state

        Returns:
            String node name: "select_options", "check_time", "check_policy", etc.
        
        NOTE: Policy routing is handled by should_route_after_policy() in routing.py,
              NOT by this method. This method only handles time-related routing.
        """
        self.memory.increment_agent_call("orchestrator_router")

        if current_phase == "after_policy":
            # This should NEVER be called - policy routing is handled by should_route_after_policy()
            self._log("ERROR: decide_next_node called with 'after_policy' - this is a bug!")
            self._log("Policy routing should use should_route_after_policy() in routing.py")
            # Return a safe default
            return "check_time"

        elif current_phase == "after_time":
            # After time check routing - enhanced to handle timeline conflicts
            # FIX: Read conflicts from the correct state key
            time_constraints = state.get("time_constraints", {})
            time_conflicts = time_constraints.get("conflicts", [])
            metrics = state.get("metrics", {})
            time_feedback_count = metrics.get("time_feedback_count", 0)
            max_time_feedback = state.get("max_time_feedback_rounds", 2)

            # Check severity - any conflict is severe for meeting reachability
            severe_conflicts = [c for c in time_conflicts if c.get("severity") in ["error", "critical"]]
            # Also check for unreachable meetings (even without explicit severity)
            unreachable_conflicts = [c for c in time_conflicts if "UNREACHABLE" in str(c.get("type", "")).upper()]
            all_severe = severe_conflicts + unreachable_conflicts

            if self.verbose:
                print(f"  [Orchestrator] Time routing: {len(time_conflicts)} conflicts, {len(all_severe)} severe")
                print(f"  [Orchestrator] Time feedback count: {time_feedback_count}/{max_time_feedback}")

            if all_severe and time_feedback_count < max_time_feedback:
                self._log(f"Severe time conflicts detected, requesting earlier flight (round {time_feedback_count + 1})")
                return "time_policy_feedback"

            # Otherwise, proceed to finalization
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

