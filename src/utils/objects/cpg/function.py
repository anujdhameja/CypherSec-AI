# from .ast import AST


# class Function:
#     def __init__(self, function):
#         self.name = function["function"]
#         self.id = function["id"].split(".")[-1]
#         self.indentation = 1
#         self.ast = AST(function["AST"], self.indentation)

#     def __str__(self):
#         indentation = self.indentation * "\t"
#         return f"{indentation}Function Name: {self.name}\n{indentation}Id: {self.id}\n{indentation}AST:{self.ast}"

#     def get_nodes(self):
#         return self.ast.nodes

#     def get_nodes_types(self):
#         return self.ast.get_nodes_type()




# class Function:
#     def __init__(self, function_data):
#         # Handle both wrapped and flat structures
#         if "function" in function_data:
#             function_data = function_data["function"]

#         self.nodes = function_data.get("nodes", [])
#         self.edges = function_data.get("edges", [])
#         self.name = function_data.get("name", "<unknown>")

#     def get_nodes(self):
#         # Convert node list into {id: Node()} dict
#         from .node import Node
#         nodes_dict = {}
#         for node_data in self.nodes:
#             try:
#                 node = Node(node_data)
#                 nodes_dict[node.id] = node
#             except Exception as e:
#                 print(f"⚠️ Skipping invalid node: {e}")
#         return nodes_dict


from .node import Node

class Function:
    def __init__(self, function_data):
        # Support wrapped {"function": {...}} and flat
        if "function" in function_data:
            function_data = function_data["function"]

        # Keep only dict nodes
        self.nodes = [n for n in function_data.get("nodes", []) if isinstance(n, dict)]
        self.edges = function_data.get("edges", [])
        self.name = function_data.get("name", "<unknown>")

    def get_nodes(self):
        nodes_dict = {}
        for node_data in self.nodes:
            try:
                node = Node(node_data)
                nodes_dict[node.id] = node
            except Exception as e:
                print(f"⚠️ Skipping invalid node: {e}")
        return nodes_dict
