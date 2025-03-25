from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing
import pandas as pd
import numpy as np

class BiasAuditor:
    def __init__(self):
        self.protected_attributes = ['Gender', 'Age', 'Location']
        self.privileged_groups = [{'Gender': 'Male'}, {'Age': 30}]
        self.unprivileged_groups = [{'Gender': 'Female'}, {'Age': 50}]
    
    def audit(self, data: pd.DataFrame):
        """Run comprehensive bias audit on dataset"""
        dataset = BinaryLabelDataset(
            df=data,
            label_names=['preferred_category'],
            protected_attribute_names=self.protected_attributes
        )
        
        # Split into train/test for fairness metrics
        train, test = dataset.split([0.7], shuffle=True)
        
        # Check statistical parity
        metric = BinaryLabelDatasetMetric(
            dataset,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        # Add classification metrics
        classified = dataset.copy()
        classified.labels = np.random.randint(2, size=dataset.labels.shape)
        
        class_metric = ClassificationMetric(
            dataset,
            classified,
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups
        )
        
        self.report = {
            "disparate_impact": metric.disparate_impact(),
            "statistical_parity": metric.statistical_parity_difference(),
            "equal_opp_diff": class_metric.equal_opportunity_difference(),
            "average_odds_diff": class_metric.average_odds_difference(),
            "consistency": metric.consistency()
        }
        return self.report
    
    def check(self, customer_data: pd.DataFrame) -> dict:
        """Check for bias in individual recommendations"""
        return {
            "gender_fairness": self._check_attribute(customer_data, 'Gender'),
            "age_fairness": self._check_attribute(customer_data, 'Age'),
            "location_fairness": self._check_attribute(customer_data, 'Location')
        }
    
    def _check_attribute(self, data: pd.DataFrame, attr: str) -> float:
        """Calculate fairness score for an attribute"""
        if attr not in data.columns:
            return 0.0
            
        group_mean = data.groupby(attr)['amount_mean'].mean()
        if len(group_mean) < 2:
            return 0.0
            
        return float(group_mean.std() / group_mean.mean())