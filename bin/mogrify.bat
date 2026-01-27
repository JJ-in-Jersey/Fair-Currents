@echo off

if "%~1"=="" (
    echo Usage: %~nx0 ^<directory^>
    exit /b
)

if not exist "%~1" (
    echo Directory not found: %~1
    exit /b
)

echo Processing PNG files in %~1...
magick mogrify -fuzz 15%% -transparent "rgb(0,255,0)" -channel A -morphology erode diamond:2 -filter Lanczos -resize 900x "%~1\*.png"
echo Done!