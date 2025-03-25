from typing import Dict
import numpy as np

class BusinessRules:
    @staticmethod
    def apply_rules(recommendations: Dict[str, float], customer: dict) -> Dict[str, float]:
        """Apply domain-specific boosting rules"""
        boosted = recommendations.copy()
        
        # Example rules
        if customer['amount_mean'] > 10000:
            boosted['Investment'] *= 1.5
        if customer['Age'] < 30:
            boosted['Ethereum'] *= 1.8
        if 'travel' in customer['Interests'].lower():
            boosted['Luxury Train Ride'] *= 2.0
            
        # Normalize
        total = sum(boosted.values())
        return {k: v/total for k, v in boosted.items()}