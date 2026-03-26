@echo off
REM =============================================================================
REM MOE Trace Analysis Pipeline (Windows)
REM =============================================================================
REM Complete analysis pipeline for DeepSeek MOE model traces:
REM 1. Analyze individual trace files (step/layer segmentation)
REM 2. Aggregate results across all ranks
REM
REM Usage:
REM   run_full_analysis.bat <trace_directory> [options]
REM
REM Arguments:
REM   trace_directory    Directory containing *.trace.json or *.trace.json.gz files
REM
REM Options:
REM   --gap-threshold-us  Step detection threshold in microseconds (default: 10000)
REM   --last-n-steps      Number of last steps to use for aggregation (default: 20)
REM   --max-traces        Max trace files to process (0=all, default: 0)
REM
REM Example:
REM   run_full_analysis.bat D:\traces --gap-threshold-us 10000 --last-n-steps 20
REM =============================================================================

setlocal EnableDelayedExpansion

REM Default values
set GAP_THRESHOLD_US=10000
set LAST_N_STEPS=20
set MAX_TRACES=0
set TRACE_DIR=

REM Parse arguments
:parse_args
if "%~1"=="" goto :check_args
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--gap-threshold-us" (
    set GAP_THRESHOLD_US=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--last-n-steps" (
    set LAST_N_STEPS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-traces" (
    set MAX_TRACES=%~2
    shift
    shift
    goto :parse_args
)
if not defined TRACE_DIR (
    set TRACE_DIR=%~1
    shift
    goto :parse_args
)
shift
goto :parse_args

:show_help
echo Usage: %~nx0 ^<trace_directory^> [options]
echo.
echo Arguments:
echo   trace_directory    Directory containing trace files
echo.
echo Options:
echo   --gap-threshold-us  Step detection threshold (default: 10000)
echo   --last-n-steps      Steps to use for aggregation (default: 20)
echo   --max-traces        Max files to process, 0=all (default: 0)
echo   --help, -h          Show this help message
exit /b 0

:check_args
if not defined TRACE_DIR (
    echo Error: Please specify a trace directory
    echo Usage: %~nx0 ^<trace_directory^> [options]
    exit /b 1
)

if not exist "%TRACE_DIR%" (
    echo Error: Directory not found: %TRACE_DIR%
    exit /b 1
)

REM Get absolute path
for %%F in ("%TRACE_DIR%") do set TRACE_DIR=%%~dpnxF

REM Set output directories (under input directory)
set ANALYSIS_DIR=%TRACE_DIR%\analysis_results
set AGGREGATED_DIR=%TRACE_DIR%\aggregated_results

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Check for Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    where python3 >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: Python not found
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

echo ================================================================================
echo MOE Trace Analysis Pipeline
echo ================================================================================
echo.
echo Configuration:
echo   Input directory:    %TRACE_DIR%
echo   Analysis output:    %ANALYSIS_DIR%
echo   Aggregated output:  %AGGREGATED_DIR%
echo   Gap threshold:      %GAP_THRESHOLD_US%us
echo   Last N steps:       %LAST_N_STEPS%
echo   Max traces:         %MAX_TRACES%
echo.

REM Check for required Python packages
echo Checking dependencies...
%PYTHON_CMD% -c "import ijson" 2>nul
if %errorlevel% neq 0 (
    echo Warning: ijson not found. Installing...
    %PYTHON_CMD% -m pip install ijson
)
echo Dependencies OK
echo.

REM Count trace files
echo Scanning for trace files...
set TRACE_COUNT=0
for /r "%TRACE_DIR%" %%F in (*.trace.json *.trace.json.gz) do (
    set /a TRACE_COUNT+=1
)
echo   Found %TRACE_COUNT% trace files
echo.

if %TRACE_COUNT% equ 0 (
    echo Error: No trace files found in %TRACE_DIR%
    exit /b 1
)

REM =============================================================================
REM Step 1: Run individual trace analysis
REM =============================================================================
echo ================================================================================
echo Step 1: Analyzing Individual Trace Files
echo ================================================================================
echo.

REM Clean up previous analysis results
if exist "%ANALYSIS_DIR%" (
    echo Removing previous analysis results...
    rmdir /s /q "%ANALYSIS_DIR%"
)
mkdir "%ANALYSIS_DIR%"

REM Run analysis
%PYTHON_CMD% "%SCRIPT_DIR%run_analysis.py" ^
    --trace-dir "%TRACE_DIR%" ^
    --output-dir "%ANALYSIS_DIR%" ^
    --gap-threshold-us %GAP_THRESHOLD_US% ^
    --max-traces %MAX_TRACES%

if %errorlevel% neq 0 (
    echo Error: Analysis failed
    exit /b 1
)

echo.

REM Check if analysis produced results
set ANALYSIS_COUNT=0
for /r "%ANALYSIS_DIR%" %%F in (*.analysis.csv) do (
    set /a ANALYSIS_COUNT+=1
)

if %ANALYSIS_COUNT% equ 0 (
    echo Error: No analysis results generated
    exit /b 1
)

echo Analysis complete: %ANALYSIS_COUNT% files generated
echo.

REM =============================================================================
REM Step 2: Aggregate results across ranks
REM =============================================================================
echo ================================================================================
echo Step 2: Aggregating Results Across Ranks
echo ================================================================================
echo.

REM Clean up previous aggregated results
if exist "%AGGREGATED_DIR%" (
    echo Removing previous aggregated results...
    rmdir /s /q "%AGGREGATED_DIR%"
)
mkdir "%AGGREGATED_DIR%"

REM Run aggregation
%PYTHON_CMD% "%SCRIPT_DIR%aggregate_analysis.py" ^
    --input-dir "%ANALYSIS_DIR%" ^
    --output-dir "%AGGREGATED_DIR%" ^
    --last-n-steps %LAST_N_STEPS%

if %errorlevel% neq 0 (
    echo Error: Aggregation failed
    exit /b 1
)

echo.

REM =============================================================================
REM Step 3: Generate component time visualization
REM =============================================================================
echo ================================================================================
echo Step 3: Generating Component Time Visualization
echo ================================================================================
echo.

set COMPONENT_PLOTS_DIR=%TRACE_DIR%\component_plots

REM Clean up previous plots
if exist "%COMPONENT_PLOTS_DIR%" (
    echo Removing previous plots...
    rmdir /s /q "%COMPONENT_PLOTS_DIR%"
)
mkdir "%COMPONENT_PLOTS_DIR%"

REM Run component time analysis
%PYTHON_CMD% "%SCRIPT_DIR%analyze_rank_component_time.py" ^
    "%ANALYSIS_DIR%" ^
    "%COMPONENT_PLOTS_DIR%"

if %errorlevel% neq 0 (
    echo Error: Component visualization failed
    exit /b 1
)

echo.

REM =============================================================================
REM Summary
REM =============================================================================
echo ================================================================================
echo Analysis Pipeline Complete!
echo ================================================================================
echo.
echo Output locations:
echo   1. Individual analysis:  %ANALYSIS_DIR%
echo      - %ANALYSIS_COUNT% CSV files (*.analysis.csv)
echo.
echo   2. Aggregated results:   %AGGREGATED_DIR%
echo      - aggregated_stats_per_layer.csv  (per-layer statistics)
echo      - aggregated_stats_averaged.csv   (averaged across layers)
echo      - aggregation_summary.txt         (human-readable summary)
echo.
echo   3. Component plots:      %COMPONENT_PLOTS_DIR%
echo      - rank_X_component_time.png (one plot per rank)
echo.

REM Show quick summary if available
if exist "%AGGREGATED_DIR%\aggregated_stats_averaged.csv" (
    echo Quick Summary (Averaged across layers):
    echo.
    for /f "skip=1 tokens=3,6,7 delims=," %%a in (%AGGREGATED_DIR%\aggregated_stats_averaged.csv) do (
        set stage=%%a
        set mean=%%b
        set std=%%c
        echo   !stage!: !mean! ms ^(±!std! ms^)
    )
    echo.
)

echo Done!
endlocal
