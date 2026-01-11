#!/usr/bin/env python3
"""
Flight Data Visualization Script

Generates charts showing:
1. Distribution of flights by airline
2. Distribution of flights by departure time (2-hour buckets)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import FlightDataLoader
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def analyze_flights():
    """Analyze and visualize flight data."""
    loader = FlightDataLoader()
    
    # Get all flights (search with broad criteria)
    all_flights = loader.flights  # Access the raw data directly
    print(f"Total flights in database: {len(all_flights)}")
    
    # 1. Analyze by airline
    airlines = [f['airline'] for f in all_flights]
    airline_counts = Counter(airlines)
    
    # 2. Analyze by departure hour (2-hour buckets)
    hour_buckets = {
        '00-02': 0, '02-04': 0, '04-06': 0, '06-08': 0,
        '08-10': 0, '10-12': 0, '12-14': 0, '14-16': 0,
        '16-18': 0, '18-20': 0, '20-22': 0, '22-24': 0
    }
    
    for f in all_flights:
        dep_time = f.get('departure_time', '12:00')
        try:
            hour = int(dep_time.split(':')[0])
            bucket_start = (hour // 2) * 2
            bucket_key = f"{bucket_start:02d}-{bucket_start+2:02d}"
            hour_buckets[bucket_key] += 1
        except:
            pass
    
    # 3. Analyze by route
    routes = [f"{f['from_city']}→{f['to_city']}" for f in all_flights]
    route_counts = Counter(routes)
    
    # 4. Analyze by price tier
    prices = [f.get('price_usd', 0) for f in all_flights]
    min_p, max_p = min(prices), max(prices)
    
    budget = len([p for p in prices if p < 300])
    mid = len([p for p in prices if 300 <= p < 700])
    premium = len([p for p in prices if p >= 700])
    
    # 5. Analyze by class
    classes = [f.get('class', 'Economy') for f in all_flights]
    class_counts = Counter(classes)
    
    # Print summary
    print("\n" + "="*60)
    print("FLIGHT DATA ANALYSIS")
    print("="*60)
    
    print("\n📊 BY AIRLINE:")
    for airline, count in sorted(airline_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"  {airline:20} {count:3d} {bar}")
    
    print("\n⏰ BY DEPARTURE TIME (2-hour buckets):")
    for bucket, count in hour_buckets.items():
        bar = "█" * (count // 3)
        print(f"  {bucket}: {count:3d} {bar}")
    
    print("\n🛫 TOP ROUTES:")
    for route, count in route_counts.most_common(10):
        print(f"  {route:15} {count:3d}")
    
    print("\n💰 BY PRICE TIER:")
    print(f"  Budget (<$300):     {budget:3d} ({100*budget/len(prices):.0f}%)")
    print(f"  Mid ($300-$700):    {mid:3d} ({100*mid/len(prices):.0f}%)")
    print(f"  Premium (>$700):    {premium:3d} ({100*premium/len(prices):.0f}%)")
    
    print("\n✈️  BY CLASS:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:15} {count:3d} ({100*count/len(all_flights):.0f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Flight Data Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Flights by Airline
    ax1 = axes[0, 0]
    sorted_airlines = sorted(airline_counts.items(), key=lambda x: -x[1])
    airlines_names = [a[0] for a in sorted_airlines]
    airlines_vals = [a[1] for a in sorted_airlines]
    colors1 = plt.cm.Set3(np.linspace(0, 1, len(airlines_names)))
    bars1 = ax1.barh(airlines_names, airlines_vals, color=colors1)
    ax1.set_xlabel('Number of Flights')
    ax1.set_title('Flights by Airline')
    ax1.invert_yaxis()
    for bar, val in zip(bars1, airlines_vals):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
    
    # Plot 2: Flights by Departure Time
    ax2 = axes[0, 1]
    buckets_names = list(hour_buckets.keys())
    buckets_vals = list(hour_buckets.values())
    colors2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(buckets_names)))
    bars2 = ax2.bar(buckets_names, buckets_vals, color=colors2)
    ax2.set_xlabel('Departure Time (hours)')
    ax2.set_ylabel('Number of Flights')
    ax2.set_title('Flights by Departure Time (2-hour buckets)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, buckets_vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center', fontsize=8)
    
    # Plot 3: Flights by Class
    ax3 = axes[1, 0]
    class_labels = list(class_counts.keys())
    class_vals = list(class_counts.values())
    colors3 = ['#3498db', '#e74c3c', '#2ecc71'][:len(class_labels)]
    wedges, texts, autotexts = ax3.pie(class_vals, labels=class_labels, autopct='%1.1f%%', 
                                        colors=colors3, startangle=90)
    ax3.set_title('Flights by Class')
    
    # Plot 4: Price Distribution
    ax4 = axes[1, 1]
    ax4.hist(prices, bins=20, color='#9b59b6', edgecolor='white', alpha=0.7)
    ax4.axvline(x=300, color='green', linestyle='--', label='Budget/Mid boundary ($300)')
    ax4.axvline(x=700, color='red', linestyle='--', label='Mid/Premium boundary ($700)')
    ax4.set_xlabel('Price (USD)')
    ax4.set_ylabel('Number of Flights')
    ax4.set_title('Flight Price Distribution')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), 'flight_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Chart saved to: {output_path}")
    
    plt.show()
    
    return {
        'total_flights': len(all_flights),
        'airlines': dict(airline_counts),
        'hour_distribution': hour_buckets,
        'price_tiers': {'budget': budget, 'mid': mid, 'premium': premium},
        'classes': dict(class_counts)
    }

if __name__ == "__main__":
    analyze_flights()
