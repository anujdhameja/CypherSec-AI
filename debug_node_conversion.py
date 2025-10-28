#!/usr/bin/env python3
"""
Debug the node conversion process to see where code field is lost
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')
import src.utils.functions.cpg as cpg
import configs

print("="*80)
print("DEBUGGING NODE CONVERSION PROCESS")
print("="*80)

# Load CPG data
cpg_files = list(Path('data/cpg').glob('*_cpg.pkl'))
cpg_df = pd.read_pickle(cpg_files[0])
sample_row = cpg_df.iloc[0]

print(f"Raw CPG structure:")
print(f"  CPG keys: {list(sample_row['cpg'].keys())}")

# Check raw nodes before parsing
raw_cpg = sample_row['cpg']
if 'nodes' in raw_cpg:
    raw_nodes = raw_cpg['nodes']
    print(f"  Raw nodes count: {len(raw_nodes)}")
    
    if len(raw_nodes) > 0:
        first_raw_node = raw_nodes[0]
        print(f"\n🔍 First raw node (before parsing):")
        print(f"  Type: {type(first_raw_node)}")
        print(f"  Keys: {list(first_raw_node.keys()) if isinstance(first_raw_node, dict) else 'Not a dict'}")
        
        if isinstance(first_raw_node, dict):
            if 'code' in first_raw_node:
                print(f"  ✓ Raw node HAS 'code': '{first_raw_node['code'][:100]}...'")
            else:
                print(f"  ❌ Raw node missing 'code' field")

# Now parse to nodes and see what happens
print(f"\n🔄 Parsing CPG to nodes...")
context = configs.Embed()
parsed_nodes = cpg.parse_to_nodes(raw_cpg, context.nodes_dim)

print(f"\n🔍 After parsing:")
print(f"  Parsed nodes count: {len(parsed_nodes)}")

if len(parsed_nodes) > 0:
    first_parsed_node = parsed_nodes[0]
    print(f"\n🔍 First parsed node (after parsing):")
    print(f"  Type: {type(first_parsed_node)}")
    print(f"  Keys: {list(first_parsed_node.keys()) if isinstance(first_parsed_node, dict) else 'Not a dict'}")
    
    if isinstance(first_parsed_node, dict):
        if 'code' in first_parsed_node:
            print(f"  ✓ Parsed node HAS 'code': '{first_parsed_node['code'][:100]}...'")
        else:
            print(f"  ❌ Parsed node missing 'code' field")
            
        # Check what fields are available
        print(f"\n  Available fields in parsed node:")
        for key, value in first_parsed_node.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"    {key}: '{value[:50]}...' (type: {type(value)})")
            else:
                print(f"    {key}: {value} (type: {type(value)})")

# Test safe_get_node_field on the parsed node
if len(parsed_nodes) > 0:
    from src.prepare.embeddings import safe_get_node_field
    
    print(f"\n🔍 Testing safe_get_node_field:")
    test_fields = ['code', 'label', 'name', 'type', 'value']
    
    for field in test_fields:
        try:
            result = safe_get_node_field(first_parsed_node, field)
            print(f"  {field}: '{result}' (type: {type(result)})")
        except Exception as e:
            print(f"  {field}: ERROR - {e}")

print("="*80)