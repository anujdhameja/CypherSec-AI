import docker
import os
import json
import base64
import shutil
import random
import string

# --- 1. CONFIGURATION ---
IMAGE_NAME = "multi-lang-fuzzer"
WORK_DIR = "temp_work_dir"  # The temp folder on your computer
CONTAINER_WORK_DIR = "/app/scanned_project"  # The project folder *inside* the container

# --- 2. CONNECT TO DOCKER ---
try:
    client = docker.from_env()
    client.ping()
    print("[+] Connected to Docker successfully.")
except Exception as e:
    print(f"[!] Error: Could not connect to Docker. Is it running?")
    print(f"    Details: {e}")
    exit(1)


# --- (build_sandbox_image function is unchanged) ---
def build_sandbox_image():
    """Builds the main Docker sandbox image."""
    print(f"[+] Building sandbox image: '{IMAGE_NAME}' (this may take a few minutes)...")
    try:
        client.images.build(path=".", tag=IMAGE_NAME, rm=True)
        print("[+] Image built successfully.")
        return True
    except Exception as e:
        print(f"[!] Error: Could not build Docker image. {e}")
        return False


def main():
    # --- 1. Build the image first ---
    # We do this once at the start.
    if not client.images.list(name=IMAGE_NAME):
        if not build_sandbox_image():
            return
    else:
        print("[+] Sandbox image is already built. Skipping build.")

    # --- 2. Get User Input ---
    target_path = input("Enter the full path to the code file you want to analyze: ").strip()
    if not os.path.exists(target_path):
        print(f"[!] Error: File not found at '{target_path}'")
        return

    # --- 3. Setup Local Work Directory ---
    print(f"[+] Preparing local work directory...")
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)  # Clean up old runs
    os.mkdir(WORK_DIR)

    # Copy the user's file into our temp work dir
    src_file_name = os.path.basename(target_path)
    shutil.copy(target_path, os.path.join(WORK_DIR, src_file_name))

    # --- NEW: Find and copy requirements.txt ---
    target_dir = os.path.dirname(target_path)
    reqs_path = os.path.join(target_dir, "requirements.txt")
    if os.path.exists(reqs_path):
        print(f"[+] Found and copied 'requirements.txt'.")
        shutil.copy(reqs_path, os.path.join(WORK_DIR, "requirements.txt"))
    else:
        print("[+] No 'requirements.txt' found. Skipping.")

    # We will pass these paths *into* the container
    container_src_path = f"{CONTAINER_WORK_DIR}/{src_file_name}"
    container_fuzz_path = f"{CONTAINER_WORK_DIR}/fuzz.txt"

    print(f"\n[+] Starting Fuzzing... (10 runs)")
    all_reports = {}

    # --- 4. The Fuzzing Loop ---
    for i in range(10):
        # ... (fuzz generation is unchanged) ...
        fuzz_data = "test_input" + random.choice([';', '|', '&', '`', '(', ')', '$', '<', '>']) + " ls " + "".join(
            random.choices(string.ascii_letters + string.digits, k=5))
        fuzz_file_path = os.path.join(WORK_DIR, "fuzz.txt")
        with open(fuzz_file_path, 'w') as f:
            f.write(fuzz_data)

        try:
            # Mount our temp dir as a "volume" inside the container
            volumes = {
                os.path.abspath(WORK_DIR): {'bind': CONTAINER_WORK_DIR, 'mode': 'rw'}
            }

            # This is the command the container will run
            command = ['python3', 'detector.py', container_src_path, container_fuzz_path]

            container = client.containers.run(
                IMAGE_NAME,
                command=command,
                volumes=volumes,
                network_disabled=True,
                remove=True,
                cap_drop=['ALL']
            )

            output = container.decode('utf-8')
            report = json.loads(output)

            if report:
                print(f"  - Run {i + 1}: Vulnerability found!")
                all_reports[f"Run {i + 1} (Input: {fuzz_data!r})"] = report
            else:
                print(f"  - Run {i + 1}: Clean.")

        except docker.errors.ContainerError as e:
            all_reports[f"Run {i + 1} (Fuzz: {fuzz_data!r})"] = {"error": "Container Failed", "logs": e.stderr.decode()}
        except Exception as e:
            all_reports[f"Run {i + 1} (Fuzz: {fuzz_data!r})"] = {"error": f"Controller Failed: {e}"}

    # --- 5. Final Report ---
    print("\n--- FUZZING COMPLETE ---")
    if all_reports:
        print(json.dumps(all_reports, indent=4))
    else:
        print("No vulnerabilities found in 10 fuzz runs.")

    # --- 6. Cleanup ---
    shutil.rmtree(WORK_DIR)
    print("\n[+] Cleanup complete.")


if __name__ == "__main__":
    main()