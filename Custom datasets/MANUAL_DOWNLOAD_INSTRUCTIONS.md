
# MANUAL DATASET DOWNLOAD INSTRUCTIONS

## 1. NIST Juliet Test Suite (REQUIRED)

### Download Steps:
1. Open browser and go to: https://samate.nist.gov/SARD/
2. Click on "Test Suites" in the navigation
3. Find "Juliet Test Suite v1.3 for C/C++"
4. Click download link (juliet-test-suite-v1.3-for-c-cpp.zip)
5. Save to: C:/Devign/devign/Custom datasets/data/juliet_raw/

### Alternative URLs (if main site is down):
- Mirror 1: https://github.com/NIST-SARD/juliet-test-suite-cplusplus
- Mirror 2: Search for "NIST Juliet Test Suite" on academic repositories

### File Details:
- Size: ~500MB compressed, ~2GB extracted
- Contains: 64,000+ test cases for C/C++
- Format: Individual .c and .cpp files organized by CWE type

## 2. GitHub Vulnerability Database (AUTOMATED)

### Collection Method:
- GitHub API search for repositories with vulnerability keywords
- Security advisory database
- Manual curation of high-quality examples

### Search Queries Used:
- "cwe-119 vulnerable language:c"
- "buffer overflow vulnerable language:c"
- "sql injection vulnerable language:c"
- "use after free vulnerable language:c"

## 3. OWASP Examples (AUTOMATED)

### Sources:
- OWASP WebGoat project
- OWASP Top 10 documentation
- OWASP Code Review Guide examples

## 4. Additional Sources (OPTIONAL)

### CVE Database:
- Visit: https://cve.mitre.org/
- Search for CVEs with code examples
- Focus on C/C++ vulnerabilities

### Exploit Database:
- Visit: https://www.exploit-db.com/
- Filter by platform: C/C++
- Look for proof-of-concept code

## PROCESSING AFTER DOWNLOAD

1. Run: python download_juliet_suite.py
2. Run: python collect_github_vulnerabilities.py  
3. Run: python combine_all_datasets.py

## EXPECTED RESULTS

After successful collection:
- dataset_devign_mapped.json: ~2,000 entries
- dataset_juliet_mapped.json: ~10,000 entries (sample)
- dataset_github_mapped.json: ~100 entries
- dataset_owasp_mapped.json: ~50 entries
- dataset_combined.json: ~12,000 unique entries

Total estimated dataset size: 10,000-15,000 vulnerability examples
