@echo off

if "%~1"=="" (
    echo Usage: %~nx0 ^<directory^>
    exit /b
)

if not exist "%~1" (
    echo Directory not found: %~1
    exit /b
)

echo Processing PNG files in %~1
magick mogrify -fuzz 15%% -transparent "rgb(0,0,255)" -channel A -morphology Erode Disk:1 +channel -filter Lanczos -resize 900x -background none -alpha background -strip -trim +repage "%~1/*.png"
echo Done!

rem magick mogrify -fuzz 15% -transparent "rgb(0,0,255)" -channel A -morphology Erode Disk:1 +channel -filter Lanczos -resize 900x -background none -alpha background *.png