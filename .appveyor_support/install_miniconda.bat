:: Configure batch
@echo on

:: Miniconda Config:
set "MINICONDA_VERSION=4.5.4"
set "MINICONDA_MD5=1c73051ccd997770288275ee6474b423"
set "MINICONDA_INSTALLER=%USERPROFILE%\miniconda.exe"
set "MINICONDA_DIR=%USERPROFILE%\miniconda"
set "MINICONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-%MINICONDA_VERSION%-Windows-x86_64.exe"

:: Install Miniconda.
powershell -Command "(New-Object Net.WebClient).DownloadFile('%MINICONDA_URL%', '%MINICONDA_INSTALLER%')"
if errorlevel 1 exit 1
( certutil -hashfile "%MINICONDA_INSTALLER%" md5 | findstr /v ":" ) > "%MINICONDA_INSTALLER%.md5"
if errorlevel 1 exit 1
type "%MINICONDA_INSTALLER%.md5"
if errorlevel 1 exit 1
set /p MINICONDA_MD5_FOUND=<%MINICONDA_INSTALLER%.md5
if errorlevel 1 exit 1
set "MINICONDA_MD5_FOUND=%MINICONDA_MD5_FOUND: =%"
if errorlevel 1 exit 1
echo "%MINICONDA_MD5_FOUND%" | findstr /c:"%MINICONDA_MD5%"
if errorlevel 1 exit 1
start /wait "" %MINICONDA_INSTALLER% /InstallationType=JustMe    ^
                                     /AddToPath=0                ^
                                     /RegisterPython=0           ^
                                     /S                          ^
                                     /D=%MINICONDA_DIR%
if errorlevel 1 exit 1
del "%MINICONDA_INSTALLER%"
if errorlevel 1 exit 1
del "%MINICONDA_INSTALLER%.md5"
if errorlevel 1 exit 1

:: Activate conda.
call "%MINICONDA_DIR%\Scripts\activate.bat"
if errorlevel 1 exit 1

:: Configure conda.
conda.exe config --set show_channel_urls true
if errorlevel 1 exit 1
conda.exe config --set auto_update_conda false
if errorlevel 1 exit 1
conda.exe config --set add_pip_as_python_dependency true
if errorlevel 1 exit 1

:: Patch VS 2008 for 64-bit support.
conda.exe install --quiet --yes "conda-forge::vs2008_express_vc_python_patch"
call setup_x64
