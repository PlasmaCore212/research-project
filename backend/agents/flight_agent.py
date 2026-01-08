from langchain_community.llms import ollama
from .models import FlightQuery, FlightSearchResult, Flight
from data.loaders import FlightDataLoader
from typing import List, Dict
import json
import re


class FlightAgent:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.llm = ollama.Ollama(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        self.loader = FlightDataLoader()

    def search_flights(self, query: FlightQuery) -> FlightSearchResult:
        """Execute search and get LLM recommendation based on actual results"""
        
        # Step 1: Search database first
        flights = self.loader.search(
            from_city=query.from_city,
            to_city=query.to_city,
            max_price=query.max_price,
            departure_after=query.departure_after,
            departure_before=query.departure_before
        )
        
        if not flights:
            return FlightSearchResult(
                query=query,
                flights=[],
                reasoning="No flights found matching the specified criteria."
            )
        
        # Step 2: Get LLM to rank the actual flights
        candidates = flights[:10]  # Top 10 for LLM to consider
        ranking_prompt = self._create_ranking_prompt(query, candidates)
        
        try:
            response = self.llm.invoke(ranking_prompt)
            rankings = json.loads(response)
            
            # Extract recommended flight IDs
            top_ids = rankings.get("top_3_flight_ids", [])[:3]
            
            # Map IDs to actual flight objects
            flight_dict = {f['flight_id']: f for f in flights}
            top_flights = [
                Flight(**flight_dict[fid]) 
                for fid in top_ids 
                if fid in flight_dict
            ]
            
            # Fallback if LLM didn't provide valid IDs
            if len(top_flights) < 3:
                top_flights = [Flight(**f) for f in candidates[:3]]
            
            reasoning = self._build_reasoning(query, candidates, rankings.get("reasoning", ""), top_flights)
            
        except (json.JSONDecodeError, KeyError):
            # Fallback: return top 3 by price
            top_flights = [Flight(**f) for f in candidates[:3]]
            reasoning = self._build_reasoning(query, candidates, "Analysis failed, sorted by price", top_flights)
        
        return FlightSearchResult(
            query=query,
            flights=top_flights,
            reasoning=reasoning
        )

    def _create_ranking_prompt(self, query: FlightQuery, flights: List[Dict]) -> str:
        """Create prompt asking LLM to rank existing flights"""
        
        flights_list = [
            {
                "flight_id": f['flight_id'],
                "airline": f['airline'],
                "departure_time": f['departure_time'],
                "arrival_time": f['arrival_time'],
                "duration_hours": round(f['duration_hours'], 2),
                "price_usd": f['price_usd']
            }
            for f in flights
        ]
        
        time_window = self._format_time_window(query.departure_after, query.departure_before)
        
        return f"""You are a Flight Booking Specialist. Rank these flights for a business traveler.

Route: {query.from_city} → {query.to_city}
Max Price: ${query.max_price if query.max_price else 'unlimited'}
Time Window: {time_window}
Class: {query.class_preference}

Available flights:
{json.dumps(flights_list, indent=2)}

Select the best 3 flights considering:
- Price-to-value ratio
- Departure time suitability for business travel
- Duration efficiency

Return ONLY this JSON format (no other text):
{{
  "top_3_flight_ids": ["FL0001", "FL0002", "FL0003"],
  "reasoning": "Brief explanation of why these 3 are best"
}}

The flight_id values MUST be from the list above. Do not invent new IDs."""

    def _build_reasoning(
        self, 
        query: FlightQuery,
        candidates: List[Dict],
        llm_reasoning: str,
        selected: List[Flight]
    ) -> str:
        """Build the ReAct-style reasoning chain"""
        
        time_window = self._format_time_window(query.departure_after, query.departure_before)
        
        # Format candidate list
        candidates_str = "\n".join([
            f"- {f['flight_id']} ({f['airline']}): ${f['price_usd']}, "
            f"{f['departure_time']} → {f['arrival_time']}, {f['duration_hours']:.2f}h"
            for f in candidates
        ])
        
        # Format selected flights
        selected_str = "\n".join([
            f"{i+1}. {f.flight_id} - {f.airline}: ${f.price_usd}, "
            f"{f.departure_time} → {f.arrival_time}, {f.duration_hours:.2f}h"
            for i, f in enumerate(selected)
        ])
        
        return f"""**Thought**: User needs flights from {query.from_city} to {query.to_city}
- Max price: ${query.max_price if query.max_price else 'No limit'}
- Time window: {time_window}
- Class: {query.class_preference}

**Action**: Searched flight database

**Observation**: Found {len(candidates)} matching flights:
{candidates_str}

**Analysis**: {llm_reasoning}

**Final Answer**: Top 3 recommendations:
{selected_str}"""

    def _format_time_window(self, after: str, before: str) -> str:
        """Format time window for display"""
        if after and before:
            return f"{after} - {before}"
        elif after:
            return f"after {after}"
        elif before:
            return f"before {before}"
        else:
            return "any time"