"""
Optimizer for tuning heuristic weights using a simple evolutionary strategy.
"""
import json
import os
import random
from typing import Dict, List
from config import config

class WeightOptimizer:
    """Manages weight populations and evolution."""
    
    def __init__(self, history_file: str = "tuning_history.json"):
        self.history_file = history_file
        self.history = self.load_history()
        
    def load_history(self) -> List[Dict]:
        """Load past game results."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """Save results to disk."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4)
            
    def record_game(self, score: int, moves: int, weights: Dict[str, float]):
        """Record game statistics with used weights."""
        self.history.append({
            "score": score,
            "moves": moves,
            "weights": weights.copy()
        })
        self.save_history()
        
    def evolve_weights(self) -> Dict[str, float]:
        """
        Suggest new weights based on past performance.
        Uses top-performing game as base and adds small mutations.
        """
        if not self.history:
            return self.get_current_weights()
            
        # Sort by score
        sorted_history = sorted(self.history, key=lambda x: x['score'], reverse=True)
        # Take the best weights
        best = sorted_history[0]['weights']
        
        # Mutate
        new_weights = {}
        for k, v in best.items():
            mutation = random.uniform(0.9, 1.1)  # +/- 10%
            new_weights[k] = v * mutation
            
        return new_weights

    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights from config."""
        return {
            "WEIGHT_EMPTY_CELLS": config.WEIGHT_EMPTY_CELLS,
            "WEIGHT_HOLES_PENALTY": config.WEIGHT_HOLES_PENALTY,
            "WEIGHT_BUMPINESS": config.WEIGHT_BUMPINESS,
            "WEIGHT_NEAR_COMPLETE": config.WEIGHT_NEAR_COMPLETE,
            "WEIGHT_STREAK_BONUS": config.WEIGHT_STREAK_BONUS
        }

    def apply_weights(self, weights: Dict[str, float]):
        """Apply weights to config."""
        config.WEIGHT_EMPTY_CELLS = weights.get("WEIGHT_EMPTY_CELLS", config.WEIGHT_EMPTY_CELLS)
        config.WEIGHT_HOLES_PENALTY = weights.get("WEIGHT_HOLES_PENALTY", config.WEIGHT_HOLES_PENALTY)
        config.WEIGHT_BUMPINESS = weights.get("WEIGHT_BUMPINESS", config.WEIGHT_BUMPINESS)
        config.WEIGHT_NEAR_COMPLETE = weights.get("WEIGHT_NEAR_COMPLETE", config.WEIGHT_NEAR_COMPLETE)
        config.WEIGHT_STREAK_BONUS = weights.get("WEIGHT_STREAK_BONUS", config.WEIGHT_STREAK_BONUS)
        config.save()

    def reset_training_data(self):
        """Reset all training logs and weights to factory defaults."""
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        self.history = []
        
        # Reset to defaults
        default_weights = {
            "WEIGHT_EMPTY_CELLS": 2.0,
            "WEIGHT_HOLES_PENALTY": -15.0,
            "WEIGHT_BUMPINESS": -0.5,
            "WEIGHT_NEAR_COMPLETE": 3.0,
            "WEIGHT_STREAK_BONUS": 20.0
        }
        self.apply_weights(default_weights)
        print("\n[OPTIMIZER] Training data reset to defaults. Population cleared.")

    def oracle_weight_adjustment(self, score_ratio: float):
        """
        Adjust weights based on comparison with the Oracle.
        Rules:
        - ratio >= 1.05: +2.0x reinforcement (reduce mutation)
        - ratio >= 0.9: +1.0x (normal evolution)
        - ratio < 0.9: -1.5x mutation away
        - ratio < 0.7: Emergency reset
        """
        if score_ratio < 0.7:
             print("\n[ORACLE] Performance critical (ratio < 0.7). Emergency Reset!")
             self.reset_training_data()
             return

        current = self.get_current_weights()
        adjustment = 1.0
        
        if score_ratio >= 1.05:
            # High performance - lock in weights
            adjustment = 0.5 # Less mutation next turn
            print(f"\n[ORACLE] Expert match detected (ratio: {score_ratio:.2f}). Reinforcing strategy.")
        elif score_ratio < 0.9:
            # Poor performance - shake things up
            adjustment = 2.0 # More mutation next turn
            print(f"\n[ORACLE] Sub-optimal match (ratio: {score_ratio:.2f}). Mutating strategy.")
            
        # We don't direct apply here, we influence the next evolve_weights call (optional expansion)
        # For now, just print the guidance.

if __name__ == "__main__":
    # Test optimizer logic
    optimizer = WeightOptimizer()
    print("Suggesting new weights...")
    new_w = optimizer.evolve_weights()
    print(json.dumps(new_w, indent=4))
