# backend/test_routing.py
from utils.routing import RoutingService

routing = RoutingService()

# JFK Airport coordinates
jfk_lat = 40.6413
jfk_lon = -73.7781

# Manhattan Financial District
manhattan_lat = 40.7074
manhattan_lon = -74.0113

distance, duration = routing.get_route(
    jfk_lon, jfk_lat,      # start
    manhattan_lon, manhattan_lat  # end
)

print(f"JFK → Manhattan Financial District:")
print(f"  Distance: {distance:.1f} km")
print(f"  Duration: {duration:.0f} minutes")
print(f"  With buffer: {duration + 15:.0f} minutes")    