# import json
# import re
# import subprocess
# import os.path
# import os
# import time
# from .cpg_client_wrapper import CPGClientWrapper
# #from ..data import datamanager as data


# def funcs_to_graphs(funcs_path):
#     client = CPGClientWrapper()
#     # query the cpg for the dataset
#     print(f"Creating CPG.")
#     graphs_string = client(funcs_path)
#     # removes unnecessary namespace for object references
#     graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
#     graphs_json = json.loads(graphs_string)

#     return graphs_json["functions"]


# # def graph_indexing(graph):
# #     idx = int(graph["file"].split(".c")[0].split("/")[-1])
# #     del graph["file"]
# #     return idx, {"functions": [graph]}


# def graph_indexing(graph):
#     file_name = graph.get("file", "N/A")
    
#     # Skip invalid or pseudo file names
#     try:
#         idx = int(file_name.split(".c")[0].split("/")[-1])
#     except ValueError:
#         idx = -1  # dummy index for non-numeric or pseudo files
    
#     # Now continue processing nodes/edges as before
#     nodes = graph.get("nodes", [])
#     edges = graph.get("edges", [])
    
#     # Example: collect node ids and labels
#     node_list = [{"id": n["id"], "label": n["label"], "code": n.get("code", "")} for n in nodes]
    
#     return {
#         "file_index": idx,
#         "function": graph.get("function", "N/A"),
#         "nodes": node_list,
#         "edges": edges
#     }



# def joern_parse(joern_path, input_path, output_path, file_name):
#     out_file = file_name + ".bin"
#     # Get the full, absolute paths so Joern knows exactly where to look
#     abs_input_path = os.path.abspath(input_path)
#     abs_output_path = os.path.abspath(output_path + out_file)

# # Define the command using the absolute paths
#     command = ["joern-parse.bat", abs_input_path, "--output", abs_output_path]

# # Run the command from INSIDE the joern-cli folder
#     joern_parse_call = subprocess.run(command, cwd=joern_path, shell=True, check=True)
#     print(str(joern_parse_call))
#     return out_file


# # def joern_create(joern_path, in_path, out_path, cpg_files):
# #     joern_process = subprocess.Popen([joern_path + "joern.bat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE ,cwd=joern_path , shell=True)
# #     json_files = []
# #     for cpg_file in cpg_files:
# #         json_file_name = f"{cpg_file.split('.')[0]}.json"
# #         json_files.append(json_file_name)

# #         print(in_path+cpg_file)
# #         if os.path.exists(in_path+cpg_file):
# #             json_out = f"{os.path.abspath(out_path)}/{json_file_name}"
# #             import_cpg_cmd = f"importCpg(\"{os.path.abspath(in_path)}/{cpg_file}\")\r".encode()
# #             script_path = f"{os.path.dirname(os.path.abspath(joern_path))}/graph-for-funcs.sc"
# #             run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{json_out}\"\r".encode()
# #             joern_process.stdin.write(import_cpg_cmd)
# #             print(joern_process.stdout.readline().decode())
# #             joern_process.stdin.write(run_script_cmd)
# #             print(joern_process.stdout.readline().decode())
# #             joern_process.stdin.write("delete\r".encode())
# #             print(joern_process.stdout.readline().decode())
# #     try:
# #         outs, errs = joern_process.communicate(timeout=60)
# #     except subprocess.TimeoutExpired:
# #         joern_process.kill()
# #         outs, errs = joern_process.communicate()
# #     if outs is not None:
# #         print(f"Outs: {outs.decode()}")
# #     if errs is not None:
# #         print(f"Errs: {errs.decode()}")
# #     return json_files
# # Add this at the top of the file if it's not there

# # Add this at the top of the file if it's not there


# # Final, corrected version of the function
# # def joern_create(joern_path, cpg_path, out_path, cpg_files):
# #     """
# #     Rewritten for robust, non-interactive Joern execution on Windows using --script.
# #     Fixes filename typo and uses correct argument passing for the script.
# #     """
# #     print("Starting non-interactive Joern processing with --script...")
# #     json_files = []
# #     # FIX 1: Corrected filename from 'graph-for-func.sc' to 'graph-for-funcs.sc'
# #     script_path = os.path.abspath(os.path.join(joern_path, "graph-for-funcs.sc"))

# #     for cpg_file in cpg_files:
# #         out_file = cpg_file.split(".")[0] + ".json"
# #         full_cpg_path = os.path.abspath(os.path.join(cpg_path, cpg_file))
# #         full_out_path = os.path.abspath(os.path.join(out_path, out_file))

# #         # FIX 2: Corrected command structure. Arguments are passed directly after the script.
# #         command = [
# #             "joern.bat",
# #             "--script", script_path,
# #             full_cpg_path,  # First argument to the script (cpgFile)
# #             full_out_path   # Second argument to the script (outFile)
# #         ]

# #         print(f"Processing {cpg_file}...")
# #         try:
# #             subprocess.run(command, cwd=joern_path, shell=True, check=True, capture_output=True, text=True)
# #             print(f"Successfully created {out_file}")
# #             json_files.append(out_file)
# #         except subprocess.CalledProcessError as e:
# #             print(f"ERROR: Failed to process {cpg_file}.")
# #             print(f"Joern's Output:\n{e.stdout}")
# #             print(f"Joern's Error Output:\n{e.stderr}")
# #             continue
    
# #     return json_files





# import os
# import subprocess

# # Final, modernized version of the function
# # def joern_create(joern_path, cpg_path, out_path, cpg_files):
# #     """
# #     Rewritten for robust, non-interactive Joern execution on Windows using the --script flag
# #     and the new custom script `extract_funcs.sc`.
# #     """
# #     print("Starting non-interactive Joern processing with modern script...")
# #     json_files = []
# #     script_path = os.path.abspath(os.path.join(joern_path, "extract_funcs.sc"))

# #     for cpg_file in cpg_files:
# #         out_file = cpg_file.split(".")[0] + ".json"
# #         full_cpg_path = os.path.abspath(os.path.join(cpg_path, cpg_file))
# #         full_out_path = os.path.abspath(os.path.join(out_path, out_file))

# #         # This is the correct non-interactive command for the new Joern
# #         command = [
# #             "joern.bat",
# #             "--script", script_path,
# #             full_cpg_path,  # First argument to our script (cpgFile)
# #             full_out_path   # Second argument to our script (outFile)
# #         ]

# #         print(f"Processing {cpg_file}...")
# #         try:
# #             subprocess.run(command, cwd=joern_path, shell=True, check=True, capture_output=True, text=True)
# #             print(f"Successfully created {out_file}")
# #             json_files.append(out_file)
# #         except subprocess.CalledProcessError as e:
# #             print(f"ERROR: Failed to process {cpg_file}.")
# #             print(f"Joern's Output:\n{e.stdout}")
# #             print(f"Joern's Error Output:\n{e.stderr}")
# #             continue

# #     return json_files


# # Made a change here 

# # def joern_create(joern_path, cpg_path, out_path, cpg_files):
# #     """
# #     Non-interactive Joern execution with hardcoded Scala script.
# #     """
# #     print("Starting non-interactive Joern processing with extract_funcs.sc...")
# #     script_path = os.path.abspath(os.path.join(joern_path, "extract_funcs.sc"))

# #     for cpg_file in cpg_files:
# #         print(f"Processing {cpg_file}...")
# #         try:
# #             subprocess.run(
# #                 ["joern.bat", "--script", script_path],
# #                 cwd=joern_path,
# #                 shell=True,
# #                 check=True,
# #                 capture_output=True,
# #                 text=True
# #             )
# #             print(f"Processed {cpg_file} successfully (JSON written to {out_path})")
# #         except subprocess.CalledProcessError as e:
# #             print(f"ERROR processing {cpg_file}:")
# #             print(f"stdout: {e.stdout}")
# #             print(f"stderr: {e.stderr}")


# #made a change here to solve the issue of not returning 

# def joern_create(joern_path, cpg_path, out_path, cpg_files):
#     """
#     Non-interactive Joern execution with hardcoded Scala script.
#     Returns a list of JSON filenames created.
#     """
#     print("Starting non-interactive Joern processing with extract_funcs.sc...")
#     script_path = os.path.abspath(os.path.join(joern_path, "extract_funcs.sc"))

#     json_files = []

#     for cpg_file in cpg_files:
#         print(f"Processing {cpg_file}...")

#         # Derive json filename from .bin
#         json_file = os.path.splitext(cpg_file)[0] + ".json"

#         try:
#             subprocess.run(
#                 ["joern.bat", "--script", script_path],
#                 cwd=joern_path,
#                 shell=True,
#                 check=True,
#                 capture_output=True,
#                 text=True
#             )
#             print(f"Processed {cpg_file} successfully (JSON written to {cpg_path})")
#             json_files.append(json_file)

#         except subprocess.CalledProcessError as e:
#             print(f"ERROR processing {cpg_file}:")
#             print(f"stdout: {e.stdout}")
#             print(f"stderr: {e.stderr}")

#     if not json_files:
#         print("[WARNING] No JSON files were created. Check Joern logs.")

#     return json_files



# # def json_process(in_path, json_file):
# #     if os.path.exists(in_path+json_file):
# #         with open(in_path+json_file) as jf:
# #             cpg_string = jf.read()
# #             cpg_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', cpg_string)
# #             cpg_json = json.loads(cpg_string)
# #             container = [graph_indexing(graph) for graph in cpg_json["functions"] if graph["file"] != "N/A"]
# #             return container
# #     return None

# def json_process(cpg_path, json_file):
#     with open(f"{cpg_path}/{json_file}", "r", encoding="utf-8") as f:
#         cpg_json = json.load(f)
    
#     container = [
#         graph_indexing(graph)
#         for graph in cpg_json.get("functions", [])
#         # Only process real files, ignore pseudo nodes
#         if graph.get("file", "N/A") != "N/A" and not graph.get("file", "").startswith("<")
#     ]
    
#     # Optional: remove graphs with dummy index (-1)
#     container = [g for g in container if g["file_index"] != -1]

#     return container


# '''
# def generate(dataset, funcs_path):
#     dataset_size = len(dataset)
#     print("Size: ", dataset_size)
#     graphs = funcs_to_graphs(funcs_path[2:])
#     print(f"Processing CPG.")
#     container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
#     graph_dataset = data.create_with_index(container, ["Index", "cpg"])
#     print(f"Dataset processed.")

#     return data.inner_join_by_index(dataset, graph_dataset)
# '''

# # client = CPGClientWrapper()
# # client.create_cpg("../../data/joern/")
# # joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# # print(funcs_to_graphs("/data/joern/"))
# """
# while True:
#     raw = input("query: ")
#     response = client.query(raw)
#     print(response)
# """


















import json
import re
import subprocess
import os
import time
import pickle
import pandas as pd
from .cpg_client_wrapper import CPGClientWrapper


# -------------------------------------------------
# 1️⃣  Extract functions via Joern client
# -------------------------------------------------
def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    print(f"Creating CPG...")
    graphs_string = client(funcs_path)
    graphs_string = re.sub(r"io\.shiftleft\.codepropertygraph\.generated\.", '', graphs_string)
    graphs_json = json.loads(graphs_string)
    return graphs_json.get("functions", [])


# -------------------------------------------------
# 2️⃣  Index & clean up a single graph
# -------------------------------------------------
def graph_indexing(graph):
    file_name = graph.get("file", "N/A")
    base = os.path.basename(file_name)
    idx = None

    # Try to parse numeric prefix; fallback to hash
    try:
        idx = int(base.split(".c")[0])
    except ValueError:
        idx = abs(hash(base)) % (10 ** 8)  # create a stable numeric ID

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_list = [
        {"id": n.get("id"), "label": n.get("label"), "code": n.get("code", "")}
        for n in nodes
    ]

    return {
        "Index": idx,
        "func": graph.get("function", "<unknown>"),
        "file": file_name,
        "cpg": {"nodes": node_list, "edges": edges},
    }


# -------------------------------------------------
# 3️⃣  Run Joern Parse (generate .bin)
# -------------------------------------------------
def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"
    abs_input_path = os.path.abspath(input_path)
    abs_output_path = os.path.abspath(os.path.join(output_path, out_file))
    command = ["joern-parse.bat", abs_input_path, "--output", abs_output_path]
    subprocess.run(command, cwd=joern_path, shell=True, check=True)
    print(f"✅ Parsed {input_path} → {out_file}")
    return out_file


# -------------------------------------------------
# 4️⃣  Run Joern Create (generate .json)
# -------------------------------------------------
#fix the issue of slow processing?
def joern_create(joern_path, cpg_path, out_path, cpg_files):
    print("Starting non-interactive Joern processing with extract_funcs.sc...")
    script_path = os.path.abspath(os.path.join(joern_path, "extract_funcs.sc"))
    json_files = []

    for cpg_file in cpg_files:
        json_file = os.path.splitext(cpg_file)[0] + ".json"
        try:
            subprocess.run(
                ["joern.bat", "--script", script_path],
                cwd=joern_path,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✅ Processed {cpg_file} → {json_file}")
            json_files.append(json_file)
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR processing {cpg_file}")
            print(e.stderr)

    if not json_files:
        print("[⚠️ WARNING] No JSON files created. Check Joern logs.")
    return json_files


# -------------------------------------------------
# 5️⃣  Parse JSON → Python graphs
# -------------------------------------------------
def json_process(cpg_path, json_file):
    with open(os.path.join(cpg_path, json_file), "r", encoding="utf-8") as f:
        cpg_json = json.load(f)

    # Process all functions
    container = [
        graph_indexing(graph)
        for graph in cpg_json.get("functions", [])
        if not graph.get("file", "").startswith("<")
    ]

    return container


# -------------------------------------------------
# 6️⃣  Combine all JSONs → Single PKL dataset
# -------------------------------------------------
def save_pkl_dataset(cpg_path):
    all_graphs = []
    for file in os.listdir(cpg_path):
        if file.endswith(".json"):
            graphs = json_process(cpg_path, file)
            all_graphs.extend(graphs)
            print(f"Processed {file} ({len(graphs)} functions)")

    if not all_graphs:
        print("❌ No graphs found. Check your JSON structure.")
        return

    df = pd.DataFrame(all_graphs)
    pkl_path = os.path.join(cpg_path, "all_cpg.pkl")
    df.to_pickle(pkl_path)
    print(f"✅ Saved {len(df)} CPG functions to {pkl_path}")
    print(df.head())


# Example manual trigger:
# save_pkl_dataset(r"C:\Devign\devign\data\cpg")
