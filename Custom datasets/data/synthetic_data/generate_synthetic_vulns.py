#!/usr/bin/env python3
# generate_synthetic_vulns.py
import os, csv, argparse, uuid
from datetime import datetime
import pandas as pd

def rnd(prefix="v"):
    return prefix + "_" + uuid.uuid4().hex[:8]

def template_cwe79(language, meta):
    if language == "PHP":
        vname = rnd("input")
        return "<?php\n// CWE-79 (XSS) - synthetic example\n// %s\nif ($_SERVER['REQUEST_METHOD'] === 'GET') {\n    $%s = $_GET['name'];\n    echo \"<h1>Hello \" . $%s . \"</h1>\";\n}\n?>" % (meta, vname, vname)
    if language == "JAVA":
        return "// CWE-79 (XSS) - synthetic Java snippet\n// %s\nout.println(\"<h1>Welcome \" + request.getParameter(\"user\") + \"</h1>\");" % meta
    if language == "PYTHON":
        return "# CWE-79 (XSS) - Flask snippet (synthetic)\n# %s\nfrom flask import Flask, request\napp = Flask(__name__)\n@app.route('/')\ndef index():\n    user = request.args.get('user')\n    return \"<h1>Hello %s</h1>\" % user\n" % meta
    return None

def template_cwe89(language, meta):
    if language == "PHP":
        return "<?php\n// CWE-89 (SQL Injection)\n// %s\n$conn = new PDO('sqlite:sample.db');\n$name = $_GET['name'];\n$sql = \"SELECT * FROM users WHERE name = '\" . $name . \"'\";\nforeach ($conn->query($sql) as $row) {\n    echo $row['name'];\n}\n?>" % meta
    if language == "PYTHON":
        return "# CWE-89 (SQL Injection)\n# %s\nimport sqlite3\nconn = sqlite3.connect('sample.db')\nc = conn.cursor()\nname = input('Enter name: ')\nquery = \"SELECT * FROM users WHERE name = '%s'\" % name\nfor row in c.execute(query):\n    print(row)\n" % meta
    if language == "JAVA":
        return "// CWE-89 (SQL Injection) - JDBC snippet\n// %s\nString name = request.getParameter(\"name\");\nString sql = \"SELECT * FROM users WHERE name = '\" + name + \"'\";\nStatement st = conn.createStatement();\nResultSet rs = st.executeQuery(sql);\n" % meta
    if language == "C#":
        return "// CWE-89 (SQL Injection) - ADO.NET snippet\n// %s\nstring name = Request.QueryString[\"name\"]; \nstring sql = \"SELECT * FROM users WHERE name = '\" + name + \"'\";\nusing(var cmd = new SqlCommand(sql, conn)) {\n    var rdr = cmd.ExecuteReader();\n}\n" % meta
    return None

def template_cwe78(language, meta):
    if language == "PYTHON":
        return "# CWE-78 (OS Command Injection)\n# %s\nimport os\narg = input('filename: ')\nos.system('dir ' + arg)\n" % meta
    if language == "JAVA":
        return "// CWE-78 (OS Command Injection)\n// %s\nString param = request.getParameter(\"cmd\");\nRuntime.getRuntime().exec(\"dir \" + param);\n" % meta
    if language == "PHP":
        return "<?php\n// CWE-78 (OS Command Injection)\n// %s\n$cmd = $_GET['cmd'];\nsystem('dir ' . $cmd);\n?>\n" % meta
    return None

def template_cwe94(language, meta):
    if language == "PYTHON":
        return "# CWE-94 (Code Injection)\n# %s\nuser = input('expr: ')\nresult = eval(user)\nprint(result)\n" % meta
    if language == "PHP":
        return "<?php\n// CWE-94 (Code Injection)\n// %s\n$code = $_GET['code'];\neval($code);\n?>\n" % meta
    return None

def template_cwe119(language, meta):
    if language == "C":
        return "/* CWE-119 (Buffer Overflow) - synthetic example\n   %s\n*/\n#include <stdio.h>\n#include <string.h>\nvoid vuln(char *input) {\n    char buf[16];\n    strcpy(buf, input);\n    printf(\"%s\\n\", buf);\n}\nint main(int argc, char **argv) {\n    if (argc > 1) vuln(argv[1]);\n    return 0;\n}\n" % meta
    if language == "C++":
        return "// CWE-119 (Buffer Overflow) - synthetic example\n// %s\n#include <iostream>\n#include <cstring>\nvoid vuln(const char *input) {\n    char buf[16];\n    strcpy(buf, input);\n    std::cout << buf << std::endl;\n}\nint main(int argc, char** argv) {\n    if (argc>1) vuln(argv[1]);\n    return 0;\n}\n" % meta
    return None

def template_cwe476(language, meta):
    if language == "C":
        return "/* CWE-476 (NULL Pointer Deref) */\n// %s\n#include <stdio.h>\nint main() {\n    char *p = NULL;\n    printf(\"%c\\n\", p[0]);\n    return 0;\n}\n" % meta
    if language == "C++":
        return "// CWE-476 (NULL Pointer Deref)\n// %s\n#include <iostream>\nint main() {\n    int *p = nullptr;\n    std::cout << p[0] << std::endl;\n    return 0;\n}\n" % meta
    return None

def template_cwe20(language, meta):
    if language == "PYTHON":
        return "# CWE-20 (Improper Input Validation)\n# %s\nval = int(input('Enter number: '))\nif val > 1000:\n    print('Large number:', val)\n" % meta
    if language == "JAVA":
        return "// CWE-20 (Improper Input Validation)\n// %s\nint v = Integer.parseInt(request.getParameter('n'));\n" % meta
    if language == "PHP":
        return "<?php\n// CWE-20 (Improper Input Validation)\n// %s\n$n = $_GET['n'];\necho intval($n);\n?>\n" % meta
    return None

CWE_TO_TEMPLATE = {
    "CWE-79": template_cwe79,
    "CWE-89": template_cwe89,
    "CWE-78": template_cwe78,
    "CWE-94": template_cwe94,
    "CWE-119": template_cwe119,
    "CWE-120": template_cwe119,
    "CWE-476": template_cwe476,
    "CWE-20": template_cwe20
}

EXT = {"C":".c","C++":".cpp","C#":".cs","PYTHON":".py","JAVA":".java","PHP":".php"}

def make_patch(code, cwe, language):
    if cwe == "CWE-89" and language == "PYTHON":
        return code.replace("query = \"SELECT * FROM users WHERE name = '%s'\" % name",
                            "query = \"SELECT * FROM users WHERE name = ?\"\nparams = (name,)\nc.execute(query, params)")
    if cwe in ("CWE-119","CWE-120") and language in ("C","C++"):
        return code.replace("strcpy(buf, input);", "strncpy(buf, input, sizeof(buf)-1); buf[sizeof(buf)-1] = '\\0';")
    if cwe == "CWE-79" and language == "PHP":
        return code.replace("echo \"<h1>Hello \" . $", "echo \"<h1>Hello \" . htmlspecialchars($")
    if cwe == "CWE-78" and language == "PYTHON":
        return code.replace("os.system('dir ' + arg)", "import shlex\narg_safe = shlex.quote(arg)\nos.system('dir ' + arg_safe)")
    if cwe == "CWE-94" and language == "PYTHON":
        return code.replace("result = eval(user)", "import ast\n# safer: parse then handle allowed nodes (illustrative)\nprint('eval removed for safety')")
    return None

def generate_for_row(row, outdir, variants=1):
    cve = str(row.get('cve','')).strip() or "NO_CVE"
    cwe = str(row.get('cwe','')).strip()
    desc = str(row.get('description','')).strip().replace('\n',' ')
    langRaw = str(row.get('language','')).strip().lower()
    langMap = {"c":"C","cpp":"C++","c++":"C++","c#":"C#","csharp":"C#","python":"PYTHON","py":"PYTHON","java":"JAVA","php":"PHP"}
    lang = langMap.get(langRaw, langRaw.upper())
    meta = "CVE=%s; CWE=%s; desc=%s" % (cve, cwe, (desc[:180] if desc else ""))
    results = []
    if not cwe:
        return results
    tpl = CWE_TO_TEMPLATE.get(cwe)
    if tpl is None:
        return results
    for _ in range(variants):
        code = tpl(lang, meta)
        if not code:
            continue
        rand = uuid.uuid4().hex[:8]
        fname = "%s_%s_%s_%s" % (lang, cwe.replace('-',''), cve.replace('/','_'), rand)
        ext = EXT.get(lang, ".txt")
        dirpath = os.path.join(outdir, lang, cwe)
        os.makedirs(dirpath, exist_ok=True)
        fpath = os.path.join(dirpath, fname + ext)
        header = "/* Synthetic vulnerable sample - generated %s UTC */\n" % datetime.utcnow().isoformat()
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(code)
            results.append(fpath)
            patched = make_patch(code, cwe, lang)
            if patched:
                fixed_path = os.path.join(dirpath, fname + "_fixed" + ext)
                with open(fixed_path, "w", encoding="utf-8") as f:
                    f.write(header)
                    f.write(patched)
                results.append(fixed_path)
        except Exception as e:
            print("Error writing", fpath, e)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--outdir", "-o", required=True)
    parser.add_argument("--variants", "-v", type=int, default=1)
    parser.add_argument("--maxrows", type=int, default=0)
    args = parser.parse_args()
    df = pd.read_csv(args.input, dtype=str).fillna("")
    processed = []
    row_count = 0
    for idx, row in df.iterrows():
        if args.maxrows and row_count >= args.maxrows:
            break
        row_count += 1
        gen = generate_for_row(row, args.outdir, variants=args.variants)
        for p in gen:
            processed.append({
                "input_index": idx,
                "cve": row.get("cve",""),
                "cwe": row.get("cwe",""),
                "language": row.get("language",""),
                "generated_file": p
            })
        if row_count % 50 == 0:
            print("Processed", row_count, "rows...")
    idxfile = os.path.join(args.outdir, "generated_index.csv")
    keys = ["input_index","cve","cwe","language","generated_file"]
    with open(idxfile, "w", newline='', encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=keys)
        writer.writeheader()
        for r in processed:
            writer.writerow(r)
    print("Done. Generated files:", len(processed), "Index at", idxfile)

if __name__ == "__main__":
    main()


