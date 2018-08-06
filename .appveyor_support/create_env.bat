:: Show each command and output
@echo on

:: Check CONDA_ENV_TYPE was set.
if "$%CONDA_ENV_TYPE%" == "" (
    echo "Set $CONDA_ENV_TYPE externally."
    exit 1
)

:: Activate conda.
call "%MINICONDA_DIR%\Scripts\activate.bat"
if errorlevel 1 exit 1

:: Create a temporary directory for the environment.
python -c "import tempfile; print(tempfile.mkdtemp())" > tmp_dir.txt
if errorlevel 1 exit 1
set /p CONDA_ENV_PREFIX=<tmp_dir.txt
if errorlevel 1 exit 1
del tmp_dir.txt
if errorlevel 1 exit 1
set "CONDA_ENV_PATH=%CONDA_ENV_PREFIX%\%CONDA_ENV_TYPE%"
set "SCRIPT_DIR=%~dp0"
set "CONDA_ENV_SPEC=%SCRIPT_DIR%\environments\%CONDA_ENV_TYPE%.yml"

:: Fill the temporary directory.
call "%MINICONDA_DIR%\Scripts\activate.bat"
if errorlevel 1 exit 1
conda.exe env create -p "%CONDA_ENV_PATH%" -f "%CONDA_ENV_SPEC%"
if errorlevel 1 exit 1
call "%MINICONDA_DIR%\Scripts\activate.bat" "%CONDA_ENV_PATH%"
if errorlevel 1 exit 1
