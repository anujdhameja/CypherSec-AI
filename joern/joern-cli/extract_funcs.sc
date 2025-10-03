import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {
  println("Executing extract_funcs.sc (robust version)")

  val inDir = new File("C:\\Devign\\devign\\data\\cpg")
  val outD  = new File("C:\\Devign\\devign\\data\\cpg")
  outD.mkdirs()

  val binFiles = inDir.listFiles().filter(_.getName.endsWith(".bin"))
  println(s"[*] Found ${binFiles.size} CPG bin files.")

  // Helper: robust JSON string escaping
  def escape(s: String): String = {
    if (s == null) ""
    else s.flatMap {
      case '\\' => "\\\\"
      case '\"' => "\\\""
      case '\n' => "\\n"
      case '\r' => "\\r"
      case '\t' => "\\t"
      case c if c.isControl => ""  // remove other control chars
      case c => c.toString
    }
  }

  binFiles.foreach { file =>
    println(s"[*] Loading CPG from: ${file.getName}")
    val cpg: Cpg = CpgLoader.load(file.getAbsolutePath)
    println(s"[*] Successfully loaded CPG from: ${file.getName}")

    val functionsJson = cpg.method.internal.map { method =>
  val methodName = escape(method.name)
  val fileName = if (method.location != null && method.location.filename != null) escape(method.location.filename) else "N/A"

  val nodes = method.ast.map { node =>
    val codeStr = escape(node.code)
    s"""    {"id": ${node.id}, "label": "${escape(node.label)}", "code": "$codeStr"}"""
  }.l.mkString(",\n")

  val edges = method.ast.flatMap { src =>
    src._astOut.map { dst =>
      s"""    {"source": ${src.id}, "target": ${dst.id}, "label": "AST"}"""
    }
  }.l.mkString(",\n")

  s"""  {
  "function": "$methodName",
  "file": "$fileName",
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

      