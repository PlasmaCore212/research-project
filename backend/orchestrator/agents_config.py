"""
Agent instances configuration for the Trip Planning workflow.

This module initializes and exports all agent instances used across the workflow.
Centralizing agent creation ensures consistent model configuration.
"""

from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.time_agent import TimeManagementAgent
from agents.policy_agent import PolicyComplianceAgent
from orchestrator.orchestrator import TripOrchestrator
from data.loaders import FlightDataLoader, HotelDataLoader

# Model configuration - using qwen2.5:14b for better tool-calling accuracy
MODEL_NAME = "qwen2.5:14b"

# Agent instances (shared across workflow nodes)
flight_agent = FlightAgent(model_name=MODEL_NAME)
hotel_agent = HotelAgent(model_name=MODEL_NAME)
time_agent = TimeManagementAgent(model_name=MODEL_NAME)
policy_agent = PolicyComplianceAgent(model_name=MODEL_NAME)
orchestrator = TripOrchestrator(model_name=MODEL_NAME)

# Data loaders
flight_loader = FlightDataLoader()
hotel_loader = HotelDataLoader()
