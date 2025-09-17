@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set "JOERN_CLI_PATH=C:\Devign\devign\joern\joern-cli"
set "CPG_PATH=C:\Devign\devign\data\cpg"
set "SCRIPT_PATH=%JOERN_CLI_PATH%\graph-for-funcs.sc"
set "TEMP_SCRIPT=%TEMP%\joern_temp.sc"
REM ---------------------

echo [+] Starting simple graph processing...
echo.

REM --- PROCESS FILE 0 ---
echo [|] Processing 0_cpg.bin...
set "cpgFile=%CPG_PATH%\0_cpg.bin"
set "outFile=%CPG_PATH%\0_cpg.json"
(
    echo importCpg("!cpgFile:\=/!")
    echo runScript("!SCRIPT_PATH:\=/!", Map("outFile" -> "!outFile:\=/!"^)^)
) > "!TEMP_SCRIPT!"
"%JOERN_CLI_PATH%\joern.bat" --script "!TEMP_SCRIPT!"
echo [+] Finished 0_cpg.bin.

echo.

REM --- PROCESS FILE 1 ---
echo [|] Processing 1_cpg.bin...
set "cpgFile=%CPG_PATH%\1_cpg.bin"
set "outFile=%CPG_PATH%\1_cpg.json"
(
    echo importCpg("!cpgFile:\=/!")
    echo runScript("!SCRIPT_PATH:\=/!", Map("outFile" -> "!outFile:\=/!"^)^)
) > "!TEMP_SCRIPT!"
"%JOERN_CLI_PATH%\joern.bat" --script "!TEMP_SCRIPT!"
echo [+] Finished 1_cpg.bin.

REM --- You can copy the block above to process more files (2_cpg.bin, etc.) ---

del "!TEMP_SCRIPT!" > nul 2>&1
echo.
echo [!] All processing is complete.
endlocal