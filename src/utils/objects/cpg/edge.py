# class Edge:
# 	def __init__(self, edge, indentation):
# 		self.id = edge["id"].split(".")[-1]
# 		self.type = self.id.split("@")[0]
# 		self.node_in = edge["in"].split(".")[-1]
# 		self.node_out = edge["out"].split(".")[-1]
# 		self.indentation = indentation + 1

# 	def __str__(self):
# 		indentation = self.indentation * "\t"
# 		return f"\n{indentation}Edge id: {self.id}\n{indentation}Node in: {self.node_in}\n{indentation}Node out: {self.node_out}\n"




class Edge:
    def __init__(self, edge, indentation=0):
        try:
            # Handle different possible field names
            self.id = str(edge.get("id", "")).split(".")[-1]
            self.type = edge.get("type", edge.get("label", self.id.split("@")[0]))
            self.node_in = str(edge.get("in", edge.get("node_in", ""))).split(".")[-1]
            self.node_out = str(edge.get("out", edge.get("node_out", ""))).split(".")[-1]
            self.indentation = indentation + 1
        except Exception as e:
            print(f"Error creating edge from data: {edge}")
            raise

    def __str__(self):
        indentation = self.indentation * "\t"
        return (f"\n{indentation}Edge id: {self.id}"
                f"\n{indentation}Type: {self.type}"
                f"\n{indentation}Node in: {self.node_in}"
                f"\n{indentation}Node out: {self.node_out}\n")