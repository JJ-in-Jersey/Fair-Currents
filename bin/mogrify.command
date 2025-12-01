#!/bin/zsh

TARGET_DIR="$1"

# Change into the target directory provided by the first argument ($1)
cd "$TARGET_DIR"

magick mogrify +repage -strip -trim -monitor *.png(N)

# Return to the previous working directory (where the script was run from)
cd - > /dev/null
