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
from typing import List, Dict, Any, Optional
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
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        
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
    
    def _receive_message(
        self,
        from_agent: str,
        content: Any,
        performative: str = "inform"
    ) -> AgentMessage:
        """Receive a message from an agent"""
        msg = AgentMessage(
            performative=performative,
            sender=from_agent,
            receiver="orchestrator",
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
- Morning departures (6:00-10:00) are preferred for business

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
            
            # Update state
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
    
    def adjust_budgets(self, state: Dict) -> Dict:
        """
        Adjust budgets based on budget constraints using Chain-of-Thought reasoning.
        
        This method analyzes what went wrong and proposes new budget allocations
        that should help find options within budget.
        """
        
        self.memory.increment_agent_call("orchestrator_adjust")
        self.memory.metrics["iterations"] += 1
        
        # Get violations from compliance status
        compliance = state.get("compliance_status", {})
        violations = compliance.get("violations", [])
        
        current_flight_budget = state.get("max_flight_budget", 800)
        current_hotel_budget = state.get("max_hotel_budget", 300)
        total_budget = state.get("total_budget", 2000)
        iteration = state.get("iteration", 0)
        
        # Categorize violations
        flight_violations = [v for v in violations if isinstance(v, dict) and "flight" in str(v.get("message", "")).lower()]
        hotel_violations = [v for v in violations if isinstance(v, dict) and "hotel" in str(v.get("message", "")).lower()]
        
        # Log reasoning
        self.memory.log_reasoning({
            "step": f"adjust_budgets_iteration_{iteration + 1}",
            "thought": f"Found {len(flight_violations)} flight and {len(hotel_violations)} hotel issues",
            "violations": [v.get("message", str(v)) if isinstance(v, dict) else str(v) for v in violations]
        })
        
        # Chain-of-Thought prompt for budget adjustment
        prompt = f"""You are the Trip Planning Orchestrator. Budget constraints require adjustment.

CURRENT STATE:
- Iteration: {iteration + 1}
- Flight budget: ${current_flight_budget}
- Hotel budget: ${current_hotel_budget}/night
- Total budget: ${total_budget}

ISSUES FOUND:
{json.dumps(violations, indent=2)}

THINK STEP BY STEP:

Step 1 - Analyze Issues:
- Flight issues: {len(flight_violations)}
- Hotel issues: {len(hotel_violations)}
- What specific limits were exceeded?

Step 2 - Identify Root Cause:
- Are we searching with too high budgets?
- Or are there simply no options in this price range?

Step 3 - Calculate New Budgets:
- If FLIGHT issues: reduce flight budget by 15-20%
- If HOTEL issues: reduce hotel budget by 15-20%
- Ensure: new_flight + new_hotel <= total_budget

Step 4 - Verify Reasonableness:
- Flight budget should be at least $200
- Hotel budget should be at least $100/night
- New budgets should still allow finding options

Return ONLY this JSON:
{{
    "reasoning_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ..."],
    "new_flight_budget": adjusted_amount,
    "new_hotel_budget": adjusted_amount,
    "adjustment_rationale": "Why these specific adjustments"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            new_flight = result.get("new_flight_budget", current_flight_budget * 0.85)
            new_hotel = result.get("new_hotel_budget", current_hotel_budget * 0.85)
            rationale = result.get("adjustment_rationale", "Budget reduced to find compliant options")
            
            # Ensure minimums
            new_flight = max(200, int(new_flight))
            new_hotel = max(100, int(new_hotel))
            
            # Log
            self.memory.log_reasoning({
                "step": "budget_adjusted",
                "thought": rationale,
                "old_budgets": {"flight": current_flight_budget, "hotel": current_hotel_budget},
                "new_budgets": {"flight": new_flight, "hotel": new_hotel}
            })
            
            # Send feedback messages
            self._send_message(
                "flight_agent",
                f"Adjust search: new max ${new_flight}",
                "request"
            )
            self._send_message(
                "hotel_agent",
                f"Adjust search: new max ${new_hotel}/night",
                "request"
            )
            
            state["messages"].append({
                "from": "orchestrator",
                "to": "agents",
                "content": f"Budget adjustment: Flight ${current_flight_budget}→${new_flight}, Hotel ${current_hotel_budget}→${new_hotel}. {rationale}"
            })
            
            self._log(f"Adjusted budgets: Flight ${new_flight}, Hotel ${new_hotel}")
            
            return {
                **state,
                "max_flight_budget": new_flight,
                "max_hotel_budget": new_hotel,
                "iteration": iteration + 1
            }
            
        except Exception as e:
            self._log(f"Adjustment failed: {e}, using fallback reduction")
            
            # Fallback: simple 15% reduction
            return {
                **state,
                "max_flight_budget": max(200, int(current_flight_budget * 0.85)),
                "max_hotel_budget": max(100, int(current_hotel_budget * 0.85)),
                "iteration": iteration + 1
            }
    
    def finalize_trip(self, state: Dict) -> Dict:
        """
        Finalize the trip planning and generate summary.
        """
        
        self.memory.metrics["end_time"] = datetime.now().isoformat()
        
        # Get final metrics
        metrics = self.memory.get_metrics_summary()
        
        # Log final state
        self.memory.log_reasoning({
            "step": "finalize",
            "thought": "Trip planning complete",
            "metrics": metrics
        })
        
        state["messages"].append({
            "from": "orchestrator",
            "to": "user",
            "content": f"Trip planning complete. Metrics: {metrics}"
        })
        
        return {
            **state,
            "planning_metrics": metrics,
            "is_complete": True
        }
    
    def get_reasoning_trace(self) -> str:
        """Get a human-readable reasoning trace for the entire planning session"""
        
        trace_parts = [
            "# Trip Planning Reasoning Trace",
            f"Conversation ID: {self.conversation_id}",
            "",
            "## Reasoning Steps:"
        ]
        
        for step in self.memory.reasoning_steps:
            trace_parts.append(f"\n### {step.get('step', 'Unknown Step')}")
            trace_parts.append(f"**Thought**: {step.get('thought', 'N/A')}")
            if 'reasoning_steps' in step:
                trace_parts.append("**Chain-of-Thought**:")
                for rs in step['reasoning_steps']:
                    trace_parts.append(f"  - {rs}")
        
        trace_parts.append("\n## Message History:")
        for msg in self.memory.message_history[-10:]:  # Last 10 messages
            trace_parts.append(f"  [{msg.performative}] {msg.sender} → {msg.receiver}: {str(msg.content)[:50]}...")
        
        trace_parts.append("\n## Metrics:")
        metrics = self.memory.get_metrics_summary()
        for key, value in metrics.items():
            trace_parts.append(f"  - {key}: {value}")
        
        return "\n".join(trace_parts)
