
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {
  val cpgFile = new File("C:\\Devign\\devign\\data\\cpg_tmp\\cpp_cpg.bin")
  val jsonFile = new File("C:\\Devign\\devign\\data\\cpg_tmp\\cpp_cpg.json")
  
  println(s"Loading CPG from: ${cpgFile.getName}")
  val cpg: Cpg = CpgLoader.load(cpgFile.getAbsolutePath)
  
  val functions = cpg.method.internal.l
  println(s"Found ${functions.size} methods")
  
  val functionsJson = functions.map { method =>
    val methodName = if (method.name != null) method.name else ""
    val fileName = if (method.location != null && method.location.filename != null) method.location.filename else "N/A"
    
    val nodes = method.ast.l.map { node =>
      val nodeId = node.id
      val label = if (node.label != null) node.label else ""
      val code = if (node.code != null) node.code else ""
      
      // Simple JSON without complex escaping
      s"""    {"id": $nodeId, "label": "$label", "code": "$code"}"""
    }.mkString(",\n")
    
    val edges = method.ast.l.flatMap { src =>
      src._astOut.l.map { dst =>
        s"""    {"source": ${src.id}, "target": ${dst.id}, "label": "AST"}"""
      }
    }.mkString(",\n")
    
    s"""  {
  "function": "$methodName",
  "file": "$fileName", 
  "nodes": [
$nodes
  ],
  "edges": [
$edges
  ]
}"""
  }.mkString(",\n")
  
  val finalJson = s"""{
"functions": [
$functionsJson
]
}"""
  
  new java.io.PrintWriter(jsonFile) { write(finalJson); close() }
  println(s"Successfully wrote JSON to: ${jsonFile.getName}")
}
