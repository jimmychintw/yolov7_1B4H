"""
MultiHead utilities for YOLOv7
Maintains compatibility with original YOLOv7 structure
"""

import yaml
import torch
import numpy as np
from pathlib import Path


class MultiHeadConfig:
    """
    Configuration handler for MultiHead detection
    Loads and manages head assignments from yaml config
    """
    
    def __init__(self, config_path='data/coco-multihead.yaml'):
        """
        Initialize MultiHead configuration
        
        Args:
            config_path: Path to configuration yaml file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)
        
        # Basic COCO settings
        self.nc = self.data['nc']  # number of classes
        self.names = self.data['names']  # class names
        
        # MultiHead settings
        multihead = self.data.get('multihead', {})
        self.enabled = multihead.get('enabled', False)
        
        if not self.enabled:
            raise ValueError("MultiHead not enabled in configuration")
        
        self.n_heads = multihead['n_heads']
        self.strategy = multihead['strategy']
        self.shared_reg_obj = multihead.get('shared_reg_obj', True)
        self.head_assignments = multihead['head_assignments']
        self.normalize_weights = multihead.get('normalize_weights', True)
        
        # Build mapping dictionaries
        self._build_mappings()
        
        # Normalize head weights if requested
        if self.normalize_weights:
            self._normalize_head_weights()
    
    def _build_mappings(self):
        """Build utility mappings for fast lookups"""
        # Class to head mapping
        self.class_to_head = {}
        for head_id in range(self.n_heads):
            for class_id in self.head_assignments[head_id]['classes']:
                self.class_to_head[class_id] = head_id
        
        # Validate all classes are assigned
        assert len(self.class_to_head) == self.nc, \
            f"Not all classes assigned: {len(self.class_to_head)} != {self.nc}"
        
        # Head to classes mapping (for convenience)
        self.head_to_classes = {
            i: set(self.head_assignments[i]['classes']) 
            for i in range(self.n_heads)
        }
    
    def _normalize_head_weights(self):
        """Normalize head weights to sum to 1.0"""
        total_weight = sum(
            self.head_assignments[i].get('weight', 1.0) 
            for i in range(self.n_heads)
        )
        
        for i in range(self.n_heads):
            original = self.head_assignments[i].get('weight', 1.0)
            self.head_assignments[i]['weight'] = original / total_weight
    
    def get_head_mask(self, head_id, device='cpu'):
        """
        Get boolean mask for classes assigned to a specific head
        
        Args:
            head_id: Head index (0 to n_heads-1)
            device: Torch device for the mask
        
        Returns:
            Boolean tensor of shape (nc,) where True indicates classes 
            assigned to this head
        """
        if head_id >= self.n_heads:
            raise ValueError(f"Invalid head_id {head_id}, max is {self.n_heads-1}")
        
        mask = torch.zeros(self.nc, dtype=torch.bool, device=device)
        classes = self.head_assignments[head_id]['classes']
        mask[classes] = True
        
        return mask
    
    def get_head_for_class(self, class_id):
        """
        Get which head is responsible for a specific class
        
        Args:
            class_id: Class index (0 to nc-1)
        
        Returns:
            Head index responsible for this class
        """
        if class_id not in self.class_to_head:
            raise ValueError(f"Class {class_id} not found in assignments")
        return self.class_to_head[class_id]
    
    def get_classes_for_head(self, head_id):
        """
        Get list of classes assigned to a specific head
        
        Args:
            head_id: Head index (0 to n_heads-1)
        
        Returns:
            List of class indices assigned to this head
        """
        if head_id >= self.n_heads:
            raise ValueError(f"Invalid head_id {head_id}, max is {self.n_heads-1}")
        return list(self.head_to_classes[head_id])
    
    def get_head_weights(self):
        """
        Get normalized weights for all heads
        
        Returns:
            List of weights (normalized to sum to 1.0)
        """
        return [
            self.head_assignments[i]['weight'] 
            for i in range(self.n_heads)
        ]
    
    def get_head_name(self, head_id):
        """Get descriptive name for a head"""
        return self.head_assignments[head_id].get('name', f'head_{head_id}')
    
    def __repr__(self):
        """String representation"""
        return (
            f"MultiHeadConfig(n_heads={self.n_heads}, "
            f"strategy={self.strategy}, "
            f"nc={self.nc})"
        )


# Utility function for standalone testing
def validate_config(config_path='data/coco-multihead.yaml'):
    """Validate MultiHead configuration file"""
    config = MultiHeadConfig(config_path)
    
    print(f"Configuration loaded: {config}")
    print(f"Head weights: {config.get_head_weights()}")
    
    # Check class coverage
    all_classes = set()
    for head_id in range(config.n_heads):
        classes = set(config.get_classes_for_head(head_id))
        all_classes.update(classes)
        print(f"Head {head_id} ({config.get_head_name(head_id)}): {len(classes)} classes")
    
    assert len(all_classes) == config.nc, "Not all classes covered!"
    print(f"✓ All {config.nc} classes covered")
    
    return True


if __name__ == "__main__":
    # Standalone validation
    validate_config()