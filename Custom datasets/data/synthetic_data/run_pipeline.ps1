param()

# Keep everything strictly under this directory (script directory)
$ErrorActionPreference = "Stop"
$Base = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Base

$Work = Join-Path $Base "pipeline"
$KaggleDst = Join-Path $Work "kaggle_cisa_2022"
$OutRoot = $Base  # outputs live directly in the base directory per requirement

New-Item -ItemType Directory -Path $Work -Force | Out-Null
New-Item -ItemType Directory -Path $KaggleDst -Force | Out-Null
New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null

Write-Output "Working dir: $Work"
Write-Output "Output root: $OutRoot"
Write-Output "Step 0: ensuring Python deps..."

# Helper to run inline Python blocks on Windows (write temp file and execute)
function Invoke-PythonBlock {
    param(
        [Parameter(Mandatory=$true)][string]$Code,
        [Parameter(Mandatory=$true)][string]$Name
    )
    $pyPath = Join-Path $Work $Name
    $null = New-Item -ItemType File -Path $pyPath -Force
    Set-Content -Path $pyPath -Value $Code -Encoding UTF8
    & python $pyPath
    Remove-Item $pyPath -Force -ErrorAction SilentlyContinue
}

# 0) Ensure python packages (best-effort if user didn't run the BAT)
$codeDeps = @"
import subprocess, sys
def ensure(pkgs):
    subprocess.call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.call([sys.executable, '-m', 'pip', 'install'] + pkgs)
ensure(['pandas','tqdm','requests','kaggle'])
print('Deps ensured')
"@
Invoke-PythonBlock -Code $codeDeps -Name "_00_deps.py"
Write-Output "Deps ensured."

# 1) Download Kaggle dataset (attempt)
Write-Output "Downloading Kaggle dataset (requires %USERPROFILE%\\.kaggle\\kaggle.json)..."
Set-Location $KaggleDst
Write-Output "Step 1: Kaggle download to $KaggleDst ..."
try {
    kaggle datasets download thedevastator/exploring-cybersecurity-risk-via-2022-cisa-vulne -p . --unzip
} catch {
    Write-Output "Kaggle CLI failed. Ensure kaggle.json is placed correctly and you are logged in."
}

# Ensure at least one CSV exists
$csvList = Get-ChildItem -Path $KaggleDst -Filter *.csv -Recurse -ErrorAction SilentlyContinue
if (-not $csvList -or $csvList.Count -eq 0) {
    Write-Output "No CSV file found in the Kaggle download folder. Please download manually into: $KaggleDst"
    exit 1
}
Write-Output ("Found {0} CSV file(s) in Kaggle folder" -f $csvList.Count)
Write-Output "Step 2: cleaning/normalizing..."

# 2) Clean & normalize: produce cisa_clean.csv
$cleanPath = Join-Path $Work "cisa_clean.csv"
Write-Output "Cleaning CSV..."
$codeClean = @"
import pandas as pd, glob, os
src_dir = r"$KaggleDst"
out = r"$cleanPath"
files = sorted(glob.glob(os.path.join(src_dir, "*.csv")))
if not files:
    raise SystemExit("No CSVs to merge")
# read and align columns
frames = []
for fn in files:
    df = pd.read_csv(fn, dtype=str, encoding='utf-8', on_bad_lines='skip', engine='python').fillna("")
    frames.append(df)
df = pd.concat(frames, ignore_index=True, sort=False).fillna("")
possible_desc = ["description","summary","vulnerability_description","short_description","details","desc"]
desc_col = None
for c in possible_desc:
    if c in df.columns:
        desc_col = c
        break
if desc_col is None:
    desc_col = df.columns[0]
cve_col = "cve" if "cve" in df.columns else ( "CVE" if "CVE" in df.columns else ("cve_id" if "cve_id" in df.columns else "") )
cwe_col = "cwe" if "cwe" in df.columns else ( "CWE" if "CWE" in df.columns else "" )
outdf = pd.DataFrame({
    "cve": df[cve_col] if cve_col in df.columns else "",
    "cwe": df[cwe_col] if cwe_col in df.columns else "",
    "description": df[desc_col],
    "product": df["product"] if "product" in df.columns else (df["product_name"] if "product_name" in df.columns else ""),
    "vendor": df["vendor"] if "vendor" in df.columns else (df["vendor_project"] if "vendor_project" in df.columns else "")
})
outdf.to_csv(out, index=False)
print("Wrote cleaned CSV to", out)
"@
Invoke-PythonBlock -Code $codeClean -Name "_01_clean.py"
Write-Output "Cleaned CSV at $cleanPath"

# 3) Map language heuristically using product/vendor
$mappedPath = Join-Path $Work "cisa_lang.csv"
Write-Output "Inferring languages..."
$codeLang = @"
import pandas as pd
fn = r"$cleanPath"
out = r"$mappedPath"
df = pd.read_csv(fn, dtype=str).fillna("")
def map_lang(product, vendor):
    p = (str(product)+" "+str(vendor)).lower()
    if any(x in p for x in ["php","wordpress","drupal","joomla","phpmyadmin"]):
        return "PHP"
    if any(x in p for x in ["java","tomcat","jre","jboss","glassfish","websphere"]):
        return "JAVA"
    if any(x in p for x in ["python","django","flask","pypi"]):
        return "PYTHON"
    if any(x in p for x in ["c#","dotnet",".net","asp.net","aspnet"]):
        return "C#"
    if any(x in p for x in ["c++","cpp","cplusplus"]):
        return "C++"
    if any(x in p for x in ["glibc","libc","kernel","openssl","libssl","libxml","libxslt",".h","c library"]):
        return "C"
    return ""
df["language"] = df.apply(lambda r: map_lang(r.get("product",""), r.get("vendor","")), axis=1)
df.to_csv(out, index=False)
print("Wrote language-mapped CSV to", out)
"@
Invoke-PythonBlock -Code $codeLang -Name "_02_lang.py"
Write-Output "Language-mapped CSV at $mappedPath"

# 4) Infer missing CWE from description (simple keyword mapping)
$preppedPath = Join-Path $Work "cisa_prepped.csv"
Write-Output "Inferring missing CWEs..."
$codeCwe = @"
import pandas as pd
fn = r"$mappedPath"
out = r"$preppedPath"
df = pd.read_csv(fn, dtype=str).fillna("")
kw_to_cwe = {
    "sql injection": "CWE-89",
    "xss": "CWE-79",
    "cross-site scripting": "CWE-79",
    "buffer overflow": "CWE-119",
    "out of bounds": "CWE-119",
    "command injection": "CWE-78",
    "os command": "CWE-78",
    "eval(": "CWE-94",
    "null pointer": "CWE-476",
    "input validation": "CWE-20",
    "path traversal": "CWE-22"
}
def infer_cwe(row):
    if row.get("cwe"):
        return row["cwe"]
    desc = (row.get("description","") or "").lower()
    for k,v in kw_to_cwe.items():
        if k in desc:
            return v
    return ""
df["cwe"] = df.apply(infer_cwe, axis=1)
df.to_csv(out, index=False)
print("Wrote prepped CSV to", out)
"@
Invoke-PythonBlock -Code $codeCwe -Name "_03_cwe.py"
Write-Output "Prepped CSV at $preppedPath"

# 5) Filter to supported CWEs & languages
$forGenPath = Join-Path $Work "cisa_for_gen.csv"
Write-Output "Filtering supported CWEs and languages..."
$codeFilter = @"
import pandas as pd
fn = r"$preppedPath"
out = r"$forGenPath"
df = pd.read_csv(fn, dtype=str).fillna("")
supported = set(["CWE-79","CWE-89","CWE-78","CWE-94","CWE-119","CWE-120","CWE-476","CWE-20","CWE-22"])
langs = set(["C","C++","C#","PYTHON","JAVA","PHP"])
df2 = df[df['cwe'].isin(supported) & df['language'].isin(langs)]
df2.to_csv(out, index=False)
print("Wrote filtered CSV to", out, " (rows kept:", len(df2), ")")
"@
Invoke-PythonBlock -Code $codeFilter -Name "_04_filter.py"
Write-Output "Filter output at $forGenPath"

# 6) Call generator script to create synthetic code + patched versions
Write-Output "Generating synthetic code samples..."
Set-Location $Base
python .\generate_synthetic_vulns.py --input "$forGenPath" --outdir "$OutRoot" --variants 1 --maxrows 0

Write-Output "Pipeline finished. Check output at $OutRoot"

