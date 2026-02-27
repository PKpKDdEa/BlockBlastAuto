"""
Oracle Performance Validation Tool.
Simulates games from replays and compares your solver against the oracle ratio.
"""
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))
from oracle_feedback import OracleFeedback

def run_replay_analysis():
    oracle = OracleFeedback()
    if not oracle.replays:
        print("No replays found in game_replays.json. Run the bot first with F14 enabled.")
        return
        
    print(f"Analyzing {len(oracle.replays)} moves...")
    ratio = oracle.calculate_performance_ratio()
    
    print("-" * 30)
    print(f"Oracle Match Ratio: {ratio*100:.1f}%")
    print("-" * 30)
    
    if ratio >= 0.9:
        print("STATUS: PERFORMANCE OPTIMAL (Matching Oracle Experts)")
    elif ratio >= 0.7:
        print("STATUS: SUB-OPTIMAL (Deviation detected)")
    else:
        print("STATUS: CRITICAL (Emergency Reset Recommended)")

if __name__ == "__main__":
    run_replay_analysis()
