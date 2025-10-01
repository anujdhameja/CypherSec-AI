// This is the content for process_all.sc

import java.io.File

val cpgDir = new File("C:/Devign/devign/data/cpg/")
val scriptPath = "C:/Devign/devign/joern/joern-cli/graph-for-funcs.sc"

println(s"[*] Searching for .bin files in ${cpgDir.getAbsolutePath}")
val binFiles = cpgDir.listFiles.filter(_.getName.endsWith(".bin")).sortBy(_.getName)
println(s"[*] Found ${binFiles.length} files to process.")

binFiles.foreach { binFile =>
  val binPath = binFile.getAbsolutePath.replace('\\', '/')
  val jsonPath = binPath.replace(".bin", ".json")
  println(s"[+] Processing ${binFile.getName}...")

  // Load the CPG
  importCpg(binPath)

  // Run the script to create the JSON (using the compatible `runScript` command)
  runScript(scriptPath, Map("outFile" -> jsonPath))

  // Close the project to free up memory before the next one
  close
}

println("[!] All processing complete.")