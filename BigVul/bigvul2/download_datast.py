import os
import subprocess
import requests
import gdown
from tqdm import tqdm
from git import Repo

# ========= CONFIG ==========
BASE_DIR = "./vulnerability_datasets"
os.makedirs(BASE_DIR, exist_ok=True)

def download_file(url, dest):
    """Stream download with progress bar"""
    if os.path.exists(dest):
        print(f"[SKIP] {dest} already exists.")
        return
    print(f"[DOWNLOADING] {url}")
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        desc=os.path.basename(dest),
        total=total, unit='iB', unit_scale=True
    ) as bar:
        for data in r.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def git_clone(url, dest):
    if os.path.exists(dest):
        print(f"[SKIP] Repo already exists: {dest}")
        return
    print(f"[CLONING] {url}")
    Repo.clone_from(url, dest)

# ========= JULIET ==========
juliet_dir = os.path.join(BASE_DIR, "Juliet")
os.makedirs(juliet_dir, exist_ok=True)

juliet_links = {
    "C-C++": "https://samate.nist.gov/SARD/downloads/testsuite/Juliet_Test_Suite_v1.3_for_C_Cpp.zip",
    "Java": "https://samate.nist.gov/SARD/downloads/testsuite/Juliet_Test_Suite_v1.3_for_Java.zip"
}
for name, link in juliet_links.items():
    download_file(link, os.path.join(juliet_dir, f"Juliet_{name}.zip"))

# ========= SARD ==========
sard_dir = os.path.join(BASE_DIR, "SARD")
os.makedirs(sard_dir, exist_ok=True)
# SARD datasets are large and split by CWEs; here we get a representative subset.
sard_links = [
    "https://samate.nist.gov/SARD/downloads/testsuite/SARD_TestSuite.zip"
]
for link in sard_links:
    download_file(link, os.path.join(sard_dir, os.path.basename(link)))

# ========= BIG-VUL ==========
bigvul_dir = os.path.join(BASE_DIR, "BigVul")
os.makedirs(bigvul_dir, exist_ok=True)
# Kaggle mirror (requires Kaggle CLI if you have account)
print("[INFO] Please manually download Big-Vul from Kaggle:")
print("       https://www.kaggle.com/datasets/aziz0x00/big-vul-dataset")

# ========= VULDEEPECKER ==========
vdp_dir = os.path.join(BASE_DIR, "VulDeePecker")
git_clone("https://github.com/CGCL-codes/VulDeePecker.git", vdp_dir)

# ========= OWASP BENCHMARK ==========
owasp_dir = os.path.join(BASE_DIR, "OWASP_Benchmark")
git_clone("https://github.com/OWASP/Benchmark.git", owasp_dir)

# ========= DVWA ==========
dvwa_dir = os.path.join(BASE_DIR, "DVWA")
git_clone("https://github.com/digininja/DVWA.git", dvwa_dir)

# ========= WEBGOAT ==========
webgoat_dir = os.path.join(BASE_DIR, "WebGoat")
git_clone("https://github.com/WebGoat/WebGoat.git", webgoat_dir)

print("\n✅ All dataset downloads/clones completed (except Big-Vul which needs Kaggle manual download).")
