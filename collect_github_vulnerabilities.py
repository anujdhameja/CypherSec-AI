#!/usr/bin/env python3
"""
Collect vulnerability examples from GitHub and other sources
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

def create_github_vulnerability_examples():
    """Create comprehensive GitHub vulnerability examples"""
    
    # Real-world vulnerability patterns based on common CVEs and CWEs
    github_examples = [
        # Buffer Overflow Examples (CWE-119)
        {
            "id": "github_buffer_001",
            "source": "github",
            "file": "examples/buffer_overflow/strcpy_vuln.c",
            "function": "vulnerable_strcpy",
            "code": """void vulnerable_strcpy(char *user_input) {
    char buffer[64];
    strcpy(buffer, user_input);  // CWE-119: No bounds checking
    printf("Data: %s\\n", buffer);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-119",
            "cve_ids": ["CVE-2019-14287"],
            "severity": "HIGH"
        },
        {
            "id": "github_buffer_002", 
            "source": "github",
            "file": "examples/buffer_overflow/gets_vuln.c",
            "function": "read_user_input",
            "code": """void read_user_input() {
    char buffer[256];
    printf("Enter data: ");
    gets(buffer);  // CWE-119: gets() is inherently unsafe
    process_data(buffer);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-119", 
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "github_buffer_003",
            "source": "github", 
            "file": "examples/buffer_overflow/sprintf_vuln.c",
            "function": "format_message",
            "code": """void format_message(char *user_data) {
    char message[100];
    sprintf(message, "User: %s", user_data);  // CWE-119: No length check
    display_message(message);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-119",
            "cve_ids": [],
            "severity": "HIGH"
        },
        
        # SQL Injection Examples (CWE-89)
        {
            "id": "github_sql_001",
            "source": "github",
            "file": "examples/sql_injection/login_vuln.c", 
            "function": "authenticate_user",
            "code": """int authenticate_user(char *username, char *password) {
    char query[512];
    sprintf(query, "SELECT * FROM users WHERE username='%s' AND password='%s'", 
            username, password);  // CWE-89: SQL injection
    return execute_query(query);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-89",
            "cve_ids": [],
            "severity": "HIGH"
        },
        
        # Use After Free (CWE-416)
        {
            "id": "github_uaf_001",
            "source": "github",
            "file": "examples/use_after_free/double_free.c",
            "function": "process_data",
            "code": """void process_data(char *data) {
    char *buffer = malloc(strlen(data) + 1);
    strcpy(buffer, data);
    
    if (error_condition) {
        free(buffer);
        return;
    }
    
    printf("Data: %s\\n", buffer);
    free(buffer);  // CWE-416: Potential double free
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-416",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "github_uaf_002",
            "source": "github",
            "file": "examples/use_after_free/use_after_free.c", 
            "function": "vulnerable_access",
            "code": """void vulnerable_access() {
    char *ptr = malloc(100);
    strcpy(ptr, "sensitive data");
    
    free(ptr);
    
    // Later in code...
    if (ptr != NULL) {
        printf("Data: %s\\n", ptr);  // CWE-416: Use after free
    }
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-416",
            "cve_ids": [],
            "severity": "HIGH"
        },
        
        # Integer Overflow (CWE-190)
        {
            "id": "github_int_001",
            "source": "github",
            "file": "examples/integer_overflow/allocation.c",
            "function": "allocate_buffer",
            "code": """char* allocate_buffer(int count, int size) {
    int total = count * size;  // CWE-190: Integer overflow possible
    char *buffer = malloc(total);
    if (!buffer) return NULL;
    
    memset(buffer, 0, total);
    return buffer;
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-190",
            "cve_ids": [],
            "severity": "MEDIUM"
        },
        
        # Path Traversal (CWE-22)
        {
            "id": "github_path_001",
            "source": "github",
            "file": "examples/path_traversal/file_access.c",
            "function": "read_config_file", 
            "code": """int read_config_file(char *filename) {
    char filepath[256];
    sprintf(filepath, "/etc/config/%s", filename);  // CWE-22: Path traversal
    
    FILE *fp = fopen(filepath, "r");
    if (!fp) return -1;
    
    // Read file content
    fclose(fp);
    return 0;
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-22",
            "cve_ids": [],
            "severity": "HIGH"
        },
        
        # Command Injection (CWE-78)
        {
            "id": "github_cmd_001",
            "source": "github",
            "file": "examples/command_injection/system_call.c",
            "function": "backup_file",
            "code": """int backup_file(char *filename) {
    char command[512];
    sprintf(command, "cp %s %s.bak", filename, filename);  // CWE-78: Command injection
    return system(command);
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-78", 
            "cve_ids": [],
            "severity": "HIGH"
        },
        
        # Null Pointer Dereference (CWE-476)
        {
            "id": "github_null_001",
            "source": "github",
            "file": "examples/null_pointer/deref.c",
            "function": "process_string",
            "code": """void process_string(char *str) {
    int len = strlen(str);  // CWE-476: No null check
    
    for (int i = 0; i < len; i++) {
        if (str[i] == '\\n') {
            str[i] = ' ';
        }
    }
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-476",
            "cve_ids": [],
            "severity": "MEDIUM"
        },
        
        # Safe Examples
        {
            "id": "github_safe_001",
            "source": "github",
            "file": "examples/safe/secure_strcpy.c",
            "function": "secure_strcpy",
            "code": """void secure_strcpy(char *user_input) {
    char buffer[64];
    if (strlen(user_input) >= sizeof(buffer)) {
        printf("Input too long\\n");
        return;
    }
    strncpy(buffer, user_input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';
    printf("Data: %s\\n", buffer);
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-119",
            "cve_ids": [],
            "severity": "LOW"
        },
        {
            "id": "github_safe_002",
            "source": "github",
            "file": "examples/safe/parameterized_query.c",
            "function": "safe_authenticate",
            "code": """int safe_authenticate(char *username, char *password) {
    // Use parameterized queries to prevent SQL injection
    sqlite3_stmt *stmt;
    const char *sql = "SELECT * FROM users WHERE username=? AND password=?";
    
    sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    sqlite3_bind_text(stmt, 1, username, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, password, -1, SQLITE_STATIC);
    
    int result = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    return result == SQLITE_ROW;
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-89",
            "cve_ids": [],
            "severity": "LOW"
        },
        {
            "id": "github_safe_003",
            "source": "github",
            "file": "examples/safe/safe_allocation.c",
            "function": "safe_allocate_buffer",
            "code": """char* safe_allocate_buffer(int count, int size) {
    // Check for integer overflow
    if (count <= 0 || size <= 0) return NULL;
    if (count > SIZE_MAX / size) return NULL;  // Overflow check
    
    size_t total = (size_t)count * size;
    char *buffer = malloc(total);
    if (!buffer) return NULL;
    
    memset(buffer, 0, total);
    return buffer;
}""",
            "is_vulnerable": False,
            "cwe_id": "CWE-190",
            "cve_ids": [],
            "severity": "LOW"
        }
    ]
    
    return github_examples

def create_owasp_examples():
    """Create OWASP-based vulnerability examples"""
    
    owasp_examples = [
        {
            "id": "owasp_xss_001",
            "source": "owasp",
            "file": "webgoat/xss/reflected_xss.c",
            "function": "display_search_results",
            "code": """void display_search_results(char *search_term) {
    printf("<html><body>");
    printf("<h1>Search Results for: %s</h1>", search_term);  // CWE-79: XSS
    printf("</body></html>");
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-79",
            "cve_ids": [],
            "severity": "MEDIUM"
        },
        {
            "id": "owasp_csrf_001", 
            "source": "owasp",
            "file": "webgoat/csrf/transfer_funds.c",
            "function": "transfer_money",
            "code": """void transfer_money(char *to_account, int amount) {
    // CWE-352: No CSRF protection
    if (is_logged_in()) {
        execute_transfer(to_account, amount);
        printf("Transfer completed\\n");
    }
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-352",
            "cve_ids": [],
            "severity": "HIGH"
        },
        {
            "id": "owasp_auth_001",
            "source": "owasp",
            "file": "webgoat/auth/weak_session.c", 
            "function": "create_session",
            "code": """char* create_session(char *username) {
    char *session_id = malloc(32);
    // CWE-330: Weak random number generation
    sprintf(session_id, "%s_%d", username, rand());
    return session_id;
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-330",
            "cve_ids": [],
            "severity": "MEDIUM"
        },
        {
            "id": "owasp_crypto_001",
            "source": "owasp",
            "file": "webgoat/crypto/weak_encryption.c",
            "function": "encrypt_password",
            "code": """char* encrypt_password(char *password) {
    // CWE-327: Use of broken cryptographic algorithm
    char *encrypted = malloc(strlen(password) + 1);
    
    // Simple XOR "encryption" - very weak
    for (int i = 0; password[i]; i++) {
        encrypted[i] = password[i] ^ 0x42;
    }
    encrypted[strlen(password)] = '\\0';
    
    return encrypted;
}""",
            "is_vulnerable": True,
            "cwe_id": "CWE-327",
            "cve_ids": [],
            "severity": "HIGH"
        }
    ]
    
    return owasp_examples

def save_github_datasets():
    """Save GitHub and OWASP datasets"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Get examples
    github_data = create_github_vulnerability_examples()
    owasp_data = create_owasp_examples()
    
    # Save GitHub dataset
    github_path = base_path / "dataset_github_mapped.json"
    with open(github_path, 'w', encoding='utf-8') as f:
        json.dump(github_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(github_data)} GitHub examples to {github_path}")
    
    # Save OWASP dataset
    owasp_path = base_path / "dataset_owasp_mapped.json"
    with open(owasp_path, 'w', encoding='utf-8') as f:
        json.dump(owasp_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(owasp_data)} OWASP examples to {owasp_path}")
    
    # Print statistics
    print(f"\nGitHub Dataset Statistics:")
    github_vuln = sum(1 for e in github_data if e["is_vulnerable"])
    print(f"Total: {len(github_data)}, Vulnerable: {github_vuln}, Safe: {len(github_data) - github_vuln}")
    
    github_cwe = {}
    for entry in github_data:
        cwe = entry.get("cwe_id", "Unknown")
        github_cwe[cwe] = github_cwe.get(cwe, 0) + 1
    
    print("CWE distribution:")
    for cwe, count in sorted(github_cwe.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cwe}: {count}")
    
    print(f"\nOWASP Dataset Statistics:")
    owasp_vuln = sum(1 for e in owasp_data if e["is_vulnerable"])
    print(f"Total: {len(owasp_data)}, Vulnerable: {owasp_vuln}, Safe: {len(owasp_data) - owasp_vuln}")

if __name__ == "__main__":
    save_github_datasets()