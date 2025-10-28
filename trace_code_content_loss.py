"""
Trace where code content is lost in the CPG pipeline.
Check each stage to see when code text disappears.
"""

import json
from pathlib import Path

print("="*80)
print("TRACING CODE CONTENT THROUGH PIPELINE")
print("="*80)

# Stage 1: Check the raw CPG JSON from Joern
print("\n" + "="*80)
print("STAGE 1: RAW CPG JSON (direct from Joern)")
print("="*80)

cpg_json_path = Path('data/cpg/0_cpg.json')
if cpg_json_path.exists():
    print(f"\n✓ Found CPG JSON: {cpg_json_path}")
    with open(cpg_json_path) as f:
        lines = f.readlines()[:30]  # First 30 lines
    print(f"✓ CPG JSON file has {len(lines)} lines")
    
    print(f"\nFirst 5 JSON objects:")
    for i, line in enumerate(lines[:5]):
        try:
            obj = json.loads(line)
            print(f"\n  Object {i}:")
            print(f"    Keys: {list(obj.keys())}")
            if 'code' in obj:
                print(f"    code: '{obj['code']}'")
            if 'label' in obj:
                print(f"    label: '{obj['label']}'")
            if 'properties' in obj:
                print(f"    properties keys: {list(obj['properties'].keys())}")
        except:
            print(f"  Object {i}: Could not parse")
else:
    print(f"❌ CPG JSON not found at {cpg_json_path}")

# Stage 2: Check the CPG pickle (after Joern extraction, before parsing)
print("\n" + "="*80)
print("STAGE 2: CPG PICKLE (after Joern, before parsing)")
print("="*80)

cpg_pkl_path = Path('data/cpg/0_cpg.pkl')
if cpg_pkl_path.exists():
    import pandas as pd
    print(f"\n✓ Found CPG pickle: {cpg_pkl_path}")
    df = pd.read_pickle(cpg_pkl_path)
    print(f"✓ CPG pickle has {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Check first sample
    if len(df) > 0:
        first_sample = df.iloc[0]
        print(f"\nFirst sample:")
        print(f"  target: {first_sample.get('target')}")
        print(f"  func length: {len(first_sample.get('func', ''))}")
        
        cpg = first_sample.get('cpg')
        if isinstance(cpg, dict):
            print(f"  CPG type: dict")
            print(f"  CPG keys: {list(cpg.keys())}")
            
            if 'nodes' in cpg:
                nodes = cpg['nodes']
                print(f"  Number of nodes: {len(nodes)}")
                
                if len(nodes) > 0:
                    first_node = nodes[0]
                    print(f"\n  First node:")
                    print(f"    Keys: {list(first_node.keys())}")
                    print(f"    Values: {first_node}")
                    
                    # Check for code in different places
                    if 'code' in first_node:
                        print(f"    ✓ Has 'code' field: {first_node['code']}")
                    if 'properties' in first_node:
                        props = first_node['properties']
                        print(f"    Has 'properties': {type(props)}")
                        if isinstance(props, dict):
                            if 'code' in props:
                                print(f"      properties['code']: {props['code']}")
else:
    print(f"❌ CPG pickle not found at {cpg_pkl_path}")

# Stage 3: Check if code is in the raw JSON from Joern
print("\n" + "="*80)
print("STAGE 3: ANALYZING RAW JOERN OUTPUT")
print("="*80)

if cpg_json_path.exists():
    print(f"\n✓ Checking actual Joern JSON structure...")
    with open(cpg_json_path) as f:
        # Parse first 100 lines to find one with actual code
        for i, line in enumerate(f):
            if i > 100:
                break
            try:
                obj = json.loads(line)
                if 'code' in obj and obj['code'] and obj['code'] != 'None':
                    print(f"\n✓ Found object with code at line {i}:")
                    print(f"  code: '{obj['code']}'")
                    print(f"  label: '{obj.get('label')}'")
                    print(f"  properties: {obj.get('properties', {})}")
                    break
            except:
                continue
        else:
            print("\n❌ No objects with actual code found in first 100 lines")
            print("   Checking if ALL objects are structure-only...")
            
            f.seek(0)
            code_count = 0
            total_count = 0
            for line in f:
                if total_count > 500:
                    break
                try:
                    obj = json.loads(line)
                    total_count += 1
                    if 'code' in obj and obj['code'] and obj['code'] != 'None':
                        code_count += 1
                except:
                    pass
            
            print(f"\n   Sample check: {code_count}/{total_count} objects have code")

# Stage 4: Check extract_funcs.sc (the Joern query)
print("\n" + "="*80)
print("STAGE 4: JOERN QUERY (extract_funcs.sc)")
print("="*80)

extract_script = Path('joern/joern-cli/extract_funcs.sc')
if extract_script.exists():
    print(f"\n✓ Found Joern extract script: {extract_script}")
    with open(extract_script) as f:
        content = f.read()
    print(f"Script length: {len(content)} bytes")
    print(f"\nScript content (first 500 chars):")
    print(content[:500])
    
    # Check if script exports code
    if '.code' in content or '"code"' in content:
        print(f"\n✓ Script appears to export code field")
    else:
        print(f"\n❌ Script does NOT appear to export code field")
else:
    print(f"❌ Extract script not found at {extract_script}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
To fix the zero features issue, we need to identify:

1. Does raw Joern JSON have code? (Check STAGE 3 output)
   - If YES: Something is removing it during parsing
   - If NO: Joern query (extract_funcs.sc) isn't extracting code

2. Check STAGE 4: Does extract_funcs.sc export .code field?
   - If not, that's why code is missing

3. Expected fix location:
   - If in Joern query: Update extract_funcs.sc to export code
   - If in parsing: Fix the CPG parsing logic to preserve code

Run this script and share the output. It will show where code content is lost.
""")