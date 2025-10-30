#!/usr/bin/env python3
"""
Create a test multi-language dataset to test the smart batching system
"""

import json

def create_test_multilang_dataset():
    """Create a sorted multi-language dataset for testing smart batching"""
    
    # Create a dataset with multiple languages, sorted by language
    dataset = []
    
    # C++ functions (indices 0-3)
    cpp_functions = [
        {
            "project": "FFmpeg",
            "commit_id": "cpp-001",
            "language": "cpp",
            "func": "int add(int a, int b) { return a + b; }",
            "target": 0
        },
        {
            "project": "FFmpeg", 
            "commit_id": "cpp-002",
            "language": "cpp",
            "func": "void unsafe_copy(char* dest, char* src) { strcpy(dest, src); }",
            "target": 1
        },
        {
            "project": "FFmpeg",
            "commit_id": "cpp-003", 
            "language": "cpp",
            "func": "int multiply(int x, int y) { return x * y; }",
            "target": 0
        },
        {
            "project": "FFmpeg",
            "commit_id": "cpp-004",
            "language": "cpp", 
            "func": "char* get_buffer() { char buf[10]; return buf; }",
            "target": 1
        }
    ]
    
    # Java functions (indices 4-6)
    java_functions = [
        {
            "project": "FFmpeg",
            "commit_id": "java-001",
            "language": "java",
            "func": "public class Math { public static int fibonacci(int n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); } }",
            "target": 0
        },
        {
            "project": "FFmpeg",
            "commit_id": "java-002", 
            "language": "java",
            "func": "import java.sql.*; public class DB { public void query(String input) { String sql = \"SELECT * FROM users WHERE name = '\" + input + \"'\"; } }",
            "target": 1
        },
        {
            "project": "FFmpeg",
            "commit_id": "java-003",
            "language": "java",
            "func": "public class Utils { public static boolean isPrime(int n) { for(int i = 2; i < n; i++) if(n % i == 0) return false; return true; } }",
            "target": 0
        }
    ]
    
    # Python functions (indices 7-9)
    python_functions = [
        {
            "project": "FFmpeg",
            "commit_id": "python-001",
            "language": "python", 
            "func": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
            "target": 0
        },
        {
            "project": "FFmpeg",
            "commit_id": "python-002",
            "language": "python",
            "func": "import subprocess\ndef execute_cmd(user_input):\n    cmd = f\"ls {user_input}\"\n    subprocess.call(cmd, shell=True)",
            "target": 1
        },
        {
            "project": "FFmpeg", 
            "commit_id": "python-003",
            "language": "python",
            "func": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            "target": 0
        }
    ]
    
    # Combine all functions (language-sorted)
    dataset = cpp_functions + java_functions + python_functions
    
    # Save the dataset
    output_file = 'data/raw/test_multilang_sorted.json'
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Created test multi-language dataset: {output_file}")
    print(f"📊 Dataset structure:")
    print(f"   C++: 4 functions (indices 0-3)")
    print(f"   Java: 3 functions (indices 4-6)")  
    print(f"   Python: 3 functions (indices 7-9)")
    print(f"   Total: {len(dataset)} functions")
    
    # Show expected batching with slice_size=3
    print(f"\n🎯 Expected smart batching (slice_size=3):")
    print(f"   Batch 0: C++ functions [0,1,2] (3 samples)")
    print(f"   Batch 1: C++ function [3] + Java functions [4,5] (3 samples)")  
    print(f"   Batch 2: Java function [6] + Python functions [7,8] (3 samples)")
    print(f"   Batch 3: Python function [9] (1 sample)")
    
    return output_file

if __name__ == "__main__":
    create_test_multilang_dataset()