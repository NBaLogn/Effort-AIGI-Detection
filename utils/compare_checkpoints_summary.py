#!/usr/bin/env python3
"""
Script to compare two PyTorch checkpoint files and show a concise summary.
"""

import torch
import os
from pathlib import Path

def compare_checkpoints(ckpt1_path, ckpt2_path):
    """Compare two PyTorch checkpoint files."""
    
    print(f"Loading checkpoint 1: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location='cpu')
    
    print(f"Loading checkpoint 2: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location='cpu')
    
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("="*80)
    
    # Compare top-level keys
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())
    
    print("\n1. TOP-LEVEL KEYS:")
    print(f"   Checkpoint 1 keys: {sorted(list(keys1))}")
    print(f"   Checkpoint 2 keys: {sorted(list(keys2))}")
    print(f"   Number of keys - Ckpt1: {len(keys1)}, Ckpt2: {len(keys2)}")
    
    if keys1 == keys2:
        print("   ✓ Both checkpoints have the same top-level keys")
    else:
        print("   ✗ Different top-level keys!")
        print(f"   Only in checkpoint 1: {keys1 - keys2}")
        print(f"   Only in checkpoint 2: {keys2 - keys1}")
    
    # Compare model state dict
    print("\n2. MODEL PARAMETERS COMPARISON:")
    print("-" * 80)
    
    # Check if checkpoints are direct state dicts or wrapped
    if 'model_state_dict' in keys1 and 'model_state_dict' in keys2:
        model1 = ckpt1['model_state_dict']
        model2 = ckpt2['model_state_dict']
    elif len(keys1) > 10 and 'backbone' in str(list(keys1)[0]):  # Direct state dict
        model1 = ckpt1
        model2 = ckpt2
    else:
        model1 = None
        model2 = None
    
    if model1 is not None and model2 is not None:
        if model1.keys() != model2.keys():
            print("   ✗ Model state dicts have different keys!")
            print(f"   Ckpt1 has {len(model1)} parameters, Ckpt2 has {len(model2)} parameters")
        else:
            all_match = True
            total_params = 0
            diff_params = 0
            diff_param_names = []
            diff_by_type = {}
            
            print(f"   Comparing {len(model1)} parameters...")
            for param_name in sorted(model1.keys()):
                total_params += 1
                if not torch.equal(model1[param_name], model2[param_name]):
                    all_match = False
                    diff_params += 1
                    if len(diff_param_names) < 20:  # Show first 20 differing params
                        diff_param_names.append(param_name)
                    
                    # Categorize differences
                    if 'residual' in param_name.lower():
                        param_type = 'residual'
                    elif 'weight_main' in param_name.lower():
                        param_type = 'weight_main'
                    elif 'head' in param_name.lower():
                        param_type = 'head'
                    elif 'bias' in param_name.lower():
                        param_type = 'bias'
                    elif 'layer_norm' in param_name.lower():
                        param_type = 'layer_norm'
                    else:
                        param_type = 'other'
                    
                    diff_by_type[param_type] = diff_by_type.get(param_type, 0) + 1
            
            if all_match:
                print(f"   ✓ Model state dicts are IDENTICAL ({total_params} parameters)")
            else:
                print(f"   ✗ Model state dicts DIFFER ({diff_params}/{total_params} parameters differ)")
                print(f"\n   Differences by parameter type:")
                for ptype, count in sorted(diff_by_type.items()):
                    print(f"      {ptype}: {count} parameters")
                if diff_param_names:
                    print(f"\n   First few differing parameters:")
                    for name in diff_param_names[:10]:
                        print(f"      - {name}")
    
    # File sizes
    size1 = os.path.getsize(ckpt1_path) / (1024 * 1024)  # MB
    size2 = os.path.getsize(ckpt2_path) / (1024 * 1024)  # MB
    print(f"\n3. FILE SIZES:")
    print(f"   Checkpoint 1: {size1:.2f} MB")
    print(f"   Checkpoint 2: {size2:.2f} MB")
    print(f"   Difference: {abs(size1 - size2):.2f} MB")

if __name__ == "__main__":
    base_dir = Path("/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/logs")
    
    ckpt1 = base_dir / "batchFacesAll" / "test" / "combined_test" / "ckpt_best.pth"
    ckpt2 = base_dir / "batchFaces2000" / "test" / "combined_test" / "batchFaces2000.pth"
    
    compare_checkpoints(ckpt1, ckpt2)

