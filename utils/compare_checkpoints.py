#!/usr/bin/env python3
"""
Script to compare two PyTorch checkpoint files.
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
    print("CHECKPOINT COMPARISON")
    print("="*80)
    
    # Compare top-level keys
    print("\n1. TOP-LEVEL KEYS:")
    print(f"   Checkpoint 1 keys: {list(ckpt1.keys())}")
    print(f"   Checkpoint 2 keys: {list(ckpt2.keys())}")
    
    keys1 = set(ckpt1.keys())
    keys2 = set(ckpt2.keys())
    
    if keys1 == keys2:
        print("   ✓ Both checkpoints have the same top-level keys")
    else:
        print("   ✗ Different top-level keys!")
        print(f"   Only in checkpoint 1: {keys1 - keys2}")
        print(f"   Only in checkpoint 2: {keys2 - keys1}")
    
    # Compare each key
    print("\n2. DETAILED COMPARISON:")
    print("-" * 80)
    
    common_keys = keys1 & keys2
    for key in sorted(common_keys):
        val1 = ckpt1[key]
        val2 = ckpt2[key]
        
        # Check type
        type1 = type(val1).__name__
        type2 = type(val2).__name__
        
        if type1 != type2:
            print(f"\n   Key: '{key}'")
            print(f"      ✗ Different types: {type1} vs {type2}")
            continue
        
        # Handle model_state_dict specially (too verbose otherwise)
        if key == 'model_state_dict':
            print(f"\n   Key: '{key}' (model parameters - detailed comparison in summary)")
            continue
        
        print(f"\n   Key: '{key}'")
        print(f"      Type: {type1}")
        
        # Handle different types
        if isinstance(val1, dict):
            print(f"      Dict keys - Ckpt1: {len(val1)} keys, Ckpt2: {len(val2)} keys")
            if val1.keys() != val2.keys():
                print(f"      ✗ Different dict keys!")
            else:
                print(f"      ✓ Same dict keys")
        
        elif isinstance(val1, torch.Tensor):
            shape1 = val1.shape
            shape2 = val2.shape
            print(f"      Shape - Ckpt1: {shape1}, Ckpt2: {shape2}")
            
            if shape1 != shape2:
                print(f"      ✗ Different shapes!")
            else:
                # Check if values are identical
                if torch.equal(val1, val2):
                    print(f"      ✓ Tensors are identical")
                else:
                    diff = torch.abs(val1 - val2)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    print(f"      ✗ Tensors differ!")
                    print(f"         Max difference: {max_diff:.6e}")
                    print(f"         Mean difference: {mean_diff:.6e}")
                    print(f"         Ckpt1 stats - Min: {val1.min().item():.6f}, Max: {val1.max().item():.6f}, Mean: {val1.mean().item():.6f}")
                    print(f"         Ckpt2 stats - Min: {val2.min().item():.6f}, Max: {val2.max().item():.6f}, Mean: {val2.mean().item():.6f}")
        
        elif isinstance(val1, (int, float, str, bool)):
            if val1 == val2:
                print(f"      ✓ Values match: {val1}")
            else:
                print(f"      ✗ Values differ: {val1} vs {val2}")
        
        elif isinstance(val1, list):
            if len(val1) != len(val2):
                print(f"      ✗ Different list lengths: {len(val1)} vs {len(val2)}")
            else:
                if val1 == val2:
                    print(f"      ✓ Lists are identical")
                else:
                    print(f"      ✗ Lists differ")
                    print(f"         Length: {len(val1)}")
    
    # Check if models are identical
    print("\n3. OVERALL SUMMARY:")
    print("-" * 80)
    
    if 'model_state_dict' in common_keys:
        model1 = ckpt1['model_state_dict']
        model2 = ckpt2['model_state_dict']
        
        if model1.keys() != model2.keys():
            print("   ✗ Model state dicts have different keys!")
            print(f"   Ckpt1 has {len(model1)} parameters, Ckpt2 has {len(model2)} parameters")
            only_in_1 = set(model1.keys()) - set(model2.keys())
            only_in_2 = set(model2.keys()) - set(model1.keys())
            if only_in_1:
                print(f"   Only in checkpoint 1: {list(only_in_1)[:5]}..." if len(only_in_1) > 5 else f"   Only in checkpoint 1: {list(only_in_1)}")
            if only_in_2:
                print(f"   Only in checkpoint 2: {list(only_in_2)[:5]}..." if len(only_in_2) > 5 else f"   Only in checkpoint 2: {list(only_in_2)}")
        else:
            all_match = True
            total_params = 0
            diff_params = 0
            diff_param_names = []
            
            print(f"   Comparing {len(model1)} parameters...")
            for param_name in sorted(model1.keys()):
                total_params += 1
                if not torch.equal(model1[param_name], model2[param_name]):
                    all_match = False
                    diff_params += 1
                    if len(diff_param_names) < 10:  # Show first 10 differing params
                        diff_param_names.append(param_name)
            
            if all_match:
                print(f"   ✓ Model state dicts are IDENTICAL ({total_params} parameters)")
            else:
                print(f"   ✗ Model state dicts DIFFER ({diff_params}/{total_params} parameters differ)")
                if diff_param_names:
                    print(f"   First few differing parameters: {diff_param_names}")
    
    # File sizes
    size1 = os.path.getsize(ckpt1_path) / (1024 * 1024)  # MB
    size2 = os.path.getsize(ckpt2_path) / (1024 * 1024)  # MB
    print(f"\n   File sizes:")
    print(f"      Checkpoint 1: {size1:.2f} MB")
    print(f"      Checkpoint 2: {size2:.2f} MB")
    print(f"      Difference: {abs(size1 - size2):.2f} MB")

if __name__ == "__main__":
    base_dir = Path("/Users/logan/Developer/WORK/DEEPFAKE_DETECTION/Effort-AIGI-Detection/DeepfakeBench/training/logs")
    
    ckpt1 = base_dir / "batchFacesAll" / "test" / "combined_test" / "ckpt_best.pth"
    ckpt2 = base_dir / "batchFaces2000" / "test" / "combined_test" / "batchFaces2000.pth"
    
    compare_checkpoints(ckpt1, ckpt2)

