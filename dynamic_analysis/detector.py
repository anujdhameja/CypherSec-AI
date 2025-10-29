import os
import subprocess
import sys
import json
import re

# --- CONFIGURATION ---
SRC_FILE = sys.argv[1]
FUZZ_FILE = sys.argv[2]
# --- NEW: Get the project's directory from the SRC_FILE path ---
PROJECT_DIR = os.path.dirname(SRC_FILE)

# Reports generated inside the container
STRACE_REPORT_FILE = os.path.join(PROJECT_DIR, "strace_report.txt")
TRIVY_REPORT_FILE = os.path.join(PROJECT_DIR, "trivy_report.json")

# Whitelist of "safe" commands for our DAST scan
SAFE_COMMANDS = {
    '/usr/bin/clear', '/usr/bin/tty', '/usr/bin/gcc',
    '/usr/bin/g++', '/usr/bin/python3', '/usr/bin/javac',
    '/usr/bin/java', '/usr/bin/php', '/usr/bin/dotnet',
    '/bin/sh'
}


def run_dependency_scan():
    """
    STAGE 1: Runs Trivy to scan for known CVEs in dependency files.
    """
    print(f"Starting dependency scan (Stage 1) on {PROJECT_DIR}...", file=sys.stderr)
    vulnerabilities = []

    # --- NEW: Scan the specific PROJECT_DIR, not the whole filesystem ---
    trivy_command = [
        'trivy', 'fs', PROJECT_DIR,
        '--format', 'json',
        '--output', TRIVY_REPORT_FILE,
        '--exit-code', '0',
        '--no-progress',
        '--skip-db-update'
    ]

    try:
        subprocess.run(trivy_command, check=True, capture_output=True)
    except FileNotFoundError:
        return [{"error": "Trivy not found in container."}]
    except subprocess.CalledProcessError as e:
        return [{"error": "Trivy scan failed", "details": e.stderr.decode()}]

    try:
        with open(TRIVY_REPORT_FILE, 'r') as f:
            trivy_data = json.load(f)
        if trivy_data.get('Results'):
            for result in trivy_data['Results']:
                if result.get('Vulnerabilities'):
                    for vuln in result['Vulnerabilities']:
                        vulnerabilities.append({
                            "vulnerability": "Known CVE Detected (SCA)",
                            "type": "Dependency",
                            "file": result.get('Target'),
                            "cve_id": vuln.get('VulnerabilityID'),
                            "package": vuln.get('PkgName'),
                            "installed_version": vuln.get('InstalledVersion'),
                            "fixed_version": vuln.get('FixedVersion', 'N/A'),
                            "severity": vuln.get('Severity'),
                            "title": vuln.get('Title', 'N/A')
                        })
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return vulnerabilities


def run_dynamic_scan():
    """
    STAGE 2: Runs our existing strace/fuzz dynamic analysis
    """
    print(f"Starting dynamic scan (Stage 2) on {SRC_FILE}...", file=sys.stderr)

    # --- NEW: Change directory into the project folder ---
    # This is critical for compilers (Java, C#) to find files
    os.chdir(PROJECT_DIR)

    # Now SRC_FILE path must be relative (just the basename)
    src_file_basename = os.path.basename(SRC_FILE)

    def compile_and_run_target():
        ext = os.path.splitext(src_file_basename)[1]
        run_command = []
        try:
            if ext == '.py':
                run_command = ['python3', src_file_basename]
            elif ext == '.c':
                subprocess.run(['gcc', src_file_basename, '-o', 'app'], check=True, capture_output=True)
                run_command = ['./app']
            elif ext == '.cpp':
                subprocess.run(['g++', src_file_basename, '-o', 'app'], check=True, capture_output=True)
                run_command = ['./app']
            elif ext == '.java':
                subprocess.run(['javac', src_file_basename], check=True, capture_output=True)
                run_command = ['java', 'Main']
            elif ext == '.php':
                run_command = ['php', src_file_basename]
            elif ext == '.cs':
                os.mkdir('cs_proj')
                os.rename(src_file_basename, 'cs_proj/Program.cs')
                subprocess.run(['dotnet', 'build', 'cs_proj'], check=True, capture_output=True)
                run_command = ['dotnet', 'run', '--project', 'cs_proj']
            else:
                return [{"error": f"Unsupported file type: {ext}", "type": "Dynamic"}]
        except subprocess.CalledProcessError as e:
            return [{"error": "Compilation/Setup Failed", "type": "Dynamic",
                     "details": e.stderr.decode() if e.stderr else str(e)}]

        try:
            # We use the relative path for fuzz.txt now
            with open("fuzz.txt", 'r') as f:
                fuzz_input = f.read()
            # strace report is also relative now
            strace_command = ['strace', '-f', '-e', 'trace=execve,socket', '-o', "strace_report.txt"] + run_command
            subprocess.run(strace_command, input=fuzz_input, text=True, timeout=5, capture_output=True)
        except subprocess.TimeoutExpired:
            return [{"vulnerability": "Potential Denial of Service (DAST)", "type": "Dynamic",
                     "details": "Program timed out (5s) under fuzz."}]
        except Exception as e:
            return [{"error": f"Runtime failed: {e}", "type": "Dynamic"}]
        return None

    def analyze_strace_report():
        vulnerabilities = []
        try:
            with open("strace_report.txt", 'r') as f:
                content = f.read()
            execve_calls = re.findall(r'execve\("([^"]+)"', content)
            for command in execve_calls:
                if command in SAFE_COMMANDS:
                    continue
                else:
                    vulnerabilities.append({
                        "vulnerability": "Command Execution Detected (DAST)",
                        "type": "Dynamic",
                        "details": f"The program tried to execute a new command: {command}"
                    })
            if "socket(" in content:
                vulnerabilities.append({
                    "vulnerability": "Network Activity Detected (DAST)",
                    "type": "Dynamic",
                    "details": "The program tried to create a network socket."
                })
        except FileNotFoundError:
            pass
        except Exception as e:
            vulnerabilities.append({"error": f"Error analyzing strace report: {e}", "type": "Dynamic"})
        return vulnerabilities

    compile_error = compile_and_run_target()
    if compile_error: return compile_error
    return analyze_strace_report()


if __name__ == "__main__":
    all_vulnerabilities = []
    all_vulnerabilities.extend(run_dependency_scan())
    all_vulnerabilities.extend(run_dynamic_scan())
    print(json.dumps(all_vulnerabilities, indent=4))