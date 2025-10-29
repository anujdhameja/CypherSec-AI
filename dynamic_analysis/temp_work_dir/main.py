import os
from tree_sitter import Language, Parser
import json
import sys

# --- 1. SETUP: Adaptively load or build grammars ---
VENDOR_PATH = "grammars"


def load_languages_adaptively():
    """
    Tries to load languages using the modern 'load' method,
    and falls back to the older 'build_library' method if needed.
    """
    print("Attempting to load language grammars...")

    # Check if we are using a modern version of the library
    if hasattr(Language, 'load'):
        print("Using modern 'Language.load()' method.")
        return {
            'python': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-python")),
            'java': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-java")),
            'javascript': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-javascript")),
            'php': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-php")),
            'c': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-c")),
            'c_sharp': Language.load(os.path.join(VENDOR_PATH, "tree-sitter-c-sharp"))
        }

    # Check if we are using an older version with build_library
    elif hasattr(Language, 'build_library'):
        print("Using fallback 'Language.build_library()' method.")
        library_path = 'build/languages.so'
        Language.build_library(
            library_path,
            [os.path.join(VENDOR_PATH, d) for d in os.listdir(VENDOR_PATH) if
             os.path.isdir(os.path.join(VENDOR_PATH, d))]
        )
        return {
            'python': Language(library_path, 'python'),
            'java': Language(library_path, 'java'),
            'javascript': Language(library_path, 'javascript'),
            'php': Language(library_path, 'php'),
            'c': Language(library_path, 'c'),
            'c_sharp': Language(library_path, 'c_sharp')
        }
    else:
        # If neither method exists, the installation is fundamentally broken
        print("FATAL ERROR: Your 'tree-sitter' library installation is corrupted or an unsupported version.",
              file=sys.stderr)
        print("It has neither a 'load' nor a 'build_library' method.", file=sys.stderr)
        sys.exit(1)


LANGUAGES = load_languages_adaptively()
print("Grammars loaded successfully.")

# --- 2. DEFINE QUERY AND LANGUAGE MAP ---
FUNCTION_QUERY = """
[
  (function_definition name: (identifier) @function.name) @function.definition
  (method_declaration name: (identifier) @function.name) @function.definition
  (function_declaration name: (identifier) @function.name) @function.definition
]
"""

LANGUAGE_MAP = {
    '.py': {'lang': LANGUAGES['python'], 'query': FUNCTION_QUERY, 'name': 'Python'},
    '.java': {'lang': LANGUAGES['java'], 'query': FUNCTION_QUERY, 'name': 'Java'},
    '.js': {'lang': LANGUAGES['javascript'], 'query': FUNCTION_QUERY, 'name': 'JavaScript'},
    '.php': {'lang': LANGUAGES['php'], 'query': FUNCTION_QUERY, 'name': 'PHP'},
    '.c': {'lang': LANGUAGES['c'], 'query': FUNCTION_QUERY, 'name': 'C/C++'},
    '.h': {'lang': LANGUAGES['c'], 'query': FUNCTION_QUERY, 'name': 'C/C++'},
    '.cpp': {'lang': LANGUAGES['c'], 'query': FUNCTION_QUERY, 'name': 'C/C++'},
    '.cs': {'lang': LANGUAGES['c_sharp'], 'query': FUNCTION_QUERY, 'name': 'C#'}
}


# --- 3. The rest of the script (analyzer and main function) remains the same ---
def analyze_code_with_treesitter(file_path):
    # ... (This function is unchanged from the previous version)
    _, extension = os.path.splitext(file_path.lower())
    if extension not in LANGUAGE_MAP: return None
    config = LANGUAGE_MAP[extension]
    language = config['lang']
    print(f"-> Analyzing {os.path.basename(file_path)}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
    except Exception as e:
        print(f"   Error reading file {os.path.basename(file_path)}: {e}");
        return None
    parser = Parser();
    parser.set_language(language)
    tree = parser.parse(bytes(source_code, "utf8"));
    query = language.query(config['query'])
    captures = query.captures(tree.root_node);
    functions = [];
    processed_nodes = set()
    for node, capture_name in captures:
        if capture_name == 'function.definition' and node.id not in processed_nodes:
            func_name_node = next((n for n, c in captures if c == 'function.name' and n.parent.id == node.id), None)
            func_name = func_name_node.text.decode('utf8') if func_name_node else 'anonymous'
            functions.append(
                {'file': os.path.basename(file_path), 'name': func_name, 'start_line': node.start_point[0] + 1,
                 'language': config['name']})
            processed_nodes.add(node.id)
    return functions


def main():
    # ... (This function is unchanged from the previous version)
    path_input = input("Enter the full path to the file or directory to analyze: ").strip()
    if not path_input: print("No path entered. Exiting."); return
    all_results = []
    if os.path.isfile(path_input):
        results = analyze_code_with_treesitter(path_input)
        if results: all_results.extend(results)
    elif os.path.isdir(path_input):
        print(f"\nScanning directory: {path_input}")
        for root, _, files in os.walk(path_input):
            for file in files:
                file_path = os.path.join(root, file)
                results = analyze_code_with_treesitter(file_path)
                if results: all_results.extend(results)
    else:
        print(f"Error: The path '{path_input}' is not a valid file or directory.");
        return
    if all_results:
        json_output = json.dumps(all_results, indent=4)
        print("\n--- Analysis Complete ---");
        print(json_output)
    else:
        print("\nNo functions were found or no supported files could be analyzed.")


if __name__ == "__main__":
    main()