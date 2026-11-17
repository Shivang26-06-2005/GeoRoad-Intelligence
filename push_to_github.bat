@echo off
title Push GeoRoad Intelligence to GitHub
echo ========================================================
echo   Push GeoRoad Intelligence to GitHub
echo ========================================================
echo.
echo Make sure you have created an EMPTY repository on GitHub:
echo   https://github.com/new
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/Shivang26-06-2005/GeoRoad-Intelligence.git): "

if "%REPO_URL%"=="" (
    echo Error: No repository URL provided!
    pause
    exit /b
)

echo.
echo Setting remote origin to %REPO_URL% ...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo Pushing to main branch...
git push -u origin main

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo   SUCCESS: Project successfully uploaded to GitHub!
    echo ========================================================
) else (
    echo ========================================================
    echo   Push failed. If prompted, please authenticate via your browser.
    echo ========================================================
)
pause
