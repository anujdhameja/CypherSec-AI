#!/usr/bin/env python3
"""
Create example vulnerability datasets with proper mapping format
This creates sample datasets while the full collection script downloads real data
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

def create_example_datasets():
    """Create example datasets in the required format"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Devign dataset mapping (assuming we already have Devign data)
    devign_mapped = [
        {
            "id": "devign_001",
            "source": "devign",
            "file": "examples/buffer_overflow_1.c",
            "function": "vulnerable_strcpy",
            "code": """void vulnerable_strcpy(char *input) {
    char buffer[100];
    strcpy(buffer, input);  // CWE-119: Buffer overflow
    printf("Buffer: %s\\n", buffer);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-119",
            "cve_ids": ["CVE-2015-0001"],
            "severity": "HIGH"
        },
        {
            "id": "devign_002", 
            "source": "devign",
            "file": "examples/safe_strcpy.c",
            "function": "safe_strcpy",
            "code": """void safe_strcpy(char *input) {
    char buffer[100];
    strncpy(buffer, input, sizeof(buffer) - 1);  // Safe: bounds checking
    buffer[sizeof(buffer) - 1] = '\\0';
    printf("Buffer: %s\\n", buffer);
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-119",
            "cve_ids": [],
            "severity": "LOW"
        }
    ]
    
    # Juliet Test Suite examples
    juliet_mapped = [
        {
            "id": "juliet_001",
            "source": "juliet",
            "file": "testcases/CWE119_Buffer_Overflow/bad.c",
            "function": "CWE119_bad",
            "code": """void CWE119_bad() {
    char data[100];
    char source[100];
    strcpy(data, source);  // FLAW: potential buffer overflow
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-119",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "juliet_002",
            "source": "juliet", 
            "file": "testcases/CWE89_SQL_Injection/bad.c",
            "function": "CWE89_bad",
            "code": """void CWE89_bad(char *user_input) {
    char query[256];
    sprintf(query, "SELECT * FROM users WHERE name='%s'", user_input);
    // FLAW: SQL injection vulnerability
    execute_sql(query);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-89",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "juliet_003",
            "source": "juliet",
            "file": "testcases/CWE89_SQL_Injection/good.c", 
            "function": "CWE89_good",
            "code": """void CWE89_good(char *user_input) {
    char query[256];
    // FIX: Use parameterized query
    prepare_statement("SELECT * FROM users WHERE name=?", user_input);
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-89",
            "cve_ids": [],
            "severity": "LOW"
        }
    ]
    
    # GitHub vulnerability examples
    github_mapped = [
        {
            "id": "github_001",
            "source": "github",
            "file": "vulnerable-code/use_after_free.c",
            "function": "vulnerable_free",
            "code": """void vulnerable_free() {
    char *ptr = malloc(100);
    free(ptr);
    strcpy(ptr, "data");  // CWE-416: Use after free
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-416",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "github_002",
            "source": "github",
            "file": "vulnerable-code/integer_overflow.c",
            "function": "vulnerable_add",
            "code": """int vulnerable_add(int a, int b) {
    return a + b;  // CWE-190: Integer overflow possible
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-190", 
            "cve_ids": [],
            "severity": "MEDIUM"
        },
        {
            "id": "github_003",
            "source": "github",
            "file": "secure-code/safe_add.c",
            "function": "safe_add",
            "code": """int safe_add(int a, int b) {
    if (a > INT_MAX - b) {
        return -1;  // Overflow detection
    }
    return a + b;
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-190",
            "cve_ids": [],
            "severity": "LOW"
        }
    ]
    
    # OWASP examples
    owasp_mapped = [
        {
            "id": "owasp_001",
            "source": "owasp",
            "file": "webgoat/path_traversal.c",
            "function": "read_user_file",
            "code": """void read_user_file(char *filename) {
    char path[256];
    sprintf(path, "/var/www/files/%s", filename);
    // CWE-22: Path traversal vulnerability
    FILE *fp = fopen(path, "r");
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-22",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "owasp_002",
            "source": "owasp",
            "file": "webgoat/command_injection.c", 
            "function": "execute_user_command",
            "code": """void execute_user_command(char *cmd) {
    char command[256];
    sprintf(command, "ls %s", cmd);
    system(command);  // CWE-78: Command injection
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-78",
            "cve_ids": [],
            "severity": "HIGH"
        }
    ]
    
    # Save individual datasets
    datasets = {
        "dataset_devign_mapped.json": devign_mapped,
        "dataset_juliet_mapped.json": juliet_mapped,
        "dataset_github_mapped.json": github_mapped,
        "dataset_owasp_mapped.json": owasp_mapped
    }
    
    for filename, data in datasets.items():
        filepath = base_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Created {filename} with {len(data)} entries")
    
    # Create combined dataset
    all_data = []
    for data in datasets.values():
        all_data.extend(data)
    
    # Remove duplicates based on code hash
    seen_hashes = set()
    unique_data = []
    for entry in all_data:
        code_hash = hashlib.md5(entry["code"].encode()).hexdigest()
        if code_hash not in seen_hashes:
            seen_hashes.add(code_hash)
            unique_data.append(entry)
    
    # Save combined dataset
    combined_path = base_path / "dataset_combined.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    # Generate statistics
    vulnerable_count = sum(1 for entry in unique_data if entry["is_vulnerable"])
    safe_count = len(unique_data) - vulnerable_count
    
    # CWE distribution
    cwe_counts = {}
    for entry in unique_data:
        cwe_id = entry.get("cwe_id", "Unknown")
        cwe_counts[cwe_id] = cwe_counts.get(cwe_id, 0) + 1
    
    # Source distribution  
    source_counts = {}
    for entry in unique_data:
        source = entry.get("source", "Unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    stats = {
        "total_entries": len(unique_data),
        "vulnerable_entries": vulnerable_count,
        "safe_entries": safe_count,
        "duplicates_removed": len(all_data) - len(unique_data),
        "source_distribution": source_counts,
        "cwe_distribution": dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)),
        "severity_distribution": {
            "HIGH": sum(1 for e in unique_data if e.get("severity") == "HIGH"),
            "MEDIUM": sum(1 for e in unique_data if e.get("severity") == "MEDIUM"), 
            "LOW": sum(1 for e in unique_data if e.get("severity") == "LOW")
        },
        "generated_at": datetime.now().isoformat()
    }
    
    stats_path = base_path / "dataset_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset Summary:")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Vulnerable: {vulnerable_count}, Safe: {safe_count}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    print(f"\nCWE distribution:")
    for cwe, count in list(cwe_counts.items())[:5]:
        print(f"  {cwe}: {count}")
    
    return stats

if __name__ == "__main__":
    create_example_datasets()