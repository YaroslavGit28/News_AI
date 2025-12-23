@echo off
setlocal enabledelayedexpansion

REM Resolve important paths
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"

echo ================================================
echo   Persona News - Dev Environment Bootstrap
echo ================================================
echo.

REM Ensure backend virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo [backend] Creating virtual environment...
  pushd "%BACKEND_DIR%"
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Ensure Python is installed.
    pause
    exit /b 1
  )
  call ".venv\Scripts\activate.bat"
  echo [backend] Installing dependencies...
  pip install -r requirements.txt
  call ".venv\Scripts\deactivate.bat"
  popd
) else (
  echo [backend] Using existing virtual environment.
)

REM Ensure frontend dependencies exist
if not exist "%FRONTEND_DIR%\node_modules" (
  echo [frontend] Installing npm dependencies...
  pushd "%FRONTEND_DIR%"
  npm install
  popd
) else (
  echo [frontend] node_modules already present.
)

echo.
echo Launching services...

REM Start backend server
start "Persona Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && call ""%VENV_DIR%\Scripts\activate.bat"" && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Small delay before starting frontend
timeout /t 2 /nobreak >nul

REM Start frontend
start "Persona Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && set NEXT_PUBLIC_API_BASE=http://localhost:8000 && npm run dev"

REM Verify backend health endpoint
echo.
echo Checking backend availability...
for /l %%I in (1,1,15) do (
  powershell -Command "try {Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/health' -TimeoutSec 2 ^| Out-Null; exit 0} catch {exit 1}"
  if not errorlevel 1 (
    echo Backend is responding on http://localhost:8000
    goto done_check
  )
  timeout /t 1 /nobreak >nul >nul
)
echo WARNING: Backend didn't respond yet. Check the ""Persona Backend"" window for errors.

:done_check
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Both servers are now starting in separate windows.
echo This window can be closed; services keep running.
pause

