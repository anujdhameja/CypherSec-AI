
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {
  println("Processing ONLY multi-language CPG files")

  val cpgDir = new File("C:\\Devign\\devign\\data\\cpg_tmp")
  
  // Only process our specific files
  val targetFiles = List("cpp_cpg.bin", "csharp_cpg.bin", "java_cpg.bin", "python_cpg.bin")
  
  targetFiles.foreach { fileName =>
    val binFile = new File(cpgDir, fileName)
    if (binFile.exists()) {
      println(s"[*] Processing: $fileName")
      
      val cpg: Cpg = CpgLoader.load(binFile.getAbsolutePath)
      println(s"[*] Successfully loaded CPG from: $fileName")

      val functionsJson = cpg.method.internal.map { method =>
        val methodName = if (method.name != null) method.name.replace("\\", "\\\\").replace("\"", "\\\"") else ""
        val fileName = if (method.location != null && method.location.filename != null) method.location.filename.replace("\\", "\\\\").replace("\"", "\\\"") else "N/A"

        val nodes = method.ast.map { node =>
          val codeStr = if (node.code != null) node.code.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") else ""
          val label = if (node.label != null) node.label.replace("\\", "\\\\").replace("\"", "\\\"") else ""
          s\"\"\"    {\"id\": ${node.id}, \"label\": \"$label\", \"code\": \"$codeStr\"}\"\"\"
        }.l.mkString(",\n")

        val edges = method.ast.flatMap { src =>
          src._astOut.map { dst =>
            s\"\"\"    {\"source\": ${src.id}, \"target\": ${dst.id}, \"label\": \"AST\"}\"\"\"
          }
        }.l.mkString(",\n")

        s\"\"\"  {
  \"function\": \"$methodName\",
  \"file\": \"$fileName\",
  \"nodes\": [
$nodes
  ],
  \"edges\": [
$edges
  ]
}\"\"\"
      }.l.mkString(",\n")

      val finalJson = s\"\"\"{
\"functions\": [
$functionsJson
]
}\"\"\"
      
      val jsonFile = new File(cpgDir, fileName.replace(".bin", ".json"))
      new java.io.PrintWriter(jsonFile) { write(finalJson); close() }
      println(s"[*] Successfully wrote JSON to: ${jsonFile.getName}")
    } else {
      println(s"[!] File not found: $fileName")
    }
  }
}
