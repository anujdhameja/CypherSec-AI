import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {
  println("Executing extract_funcs.sc which was customly created")
  val inDir = new File("C:\\Devign\\devign\\data\\cpg")
  val outD  = new File("C:\\Devign\\devign\\output")
  outD.mkdirs()

  val binFiles = inDir.listFiles().filter(_.getName.endsWith(".bin"))
  println(s"[*] Found ${binFiles.size} CPG bin files.")
  
  binFiles.foreach { file =>
    println("${file.getAbsolutePath}")
    println(s"[*] Loading CPG from: ${file.getName}")
    val cpg: Cpg = CpgLoader.load(file.getAbsolutePath)
    
    println(s"[*] Successfully loaded CPG from: ${file.getName}")
    val functionsJson = cpg.method.internal.map { method =>
      def escape(s: String): String = s.replace("\\", "\\\\").replace("\"", "\\\"")
      val methodName = method.name
      val nodes = method.ast.map { node =>
        s"""    {"id": ${node.id}, "label": "${node.label}", "code": "${escape(node.code)}"}"""
      }.l.mkString(",\n")
      val edges = method.ast.flatMap { src =>
        src._astOut.map { dst =>
          s"""    {"source": ${src.id}, "target": ${dst.id}, "label": "AST"}"""
        }
      }.l.mkString(",\n")

      s"""  {
      "function": "$methodName",
      "nodes": [\n$nodes\n  ],
      "edges": [\n$edges\n  ]
    }"""
    }.l.mkString(",\n")

    val finalJson = s"""{\n"functions": [\n$functionsJson\n]\n}"""
    val outFile = new File(outD, file.getName.replace(".bin", ".json"))
    new java.io.PrintWriter(outFile) { write(finalJson); close() }
    println(s"[*] Successfully wrote graph data to ${outFile.getName}")
  }
}
