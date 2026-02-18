#!/bin/bash

DOI=$1
FILES=$2

if [ -z "$DOI" ] || [ -z "$FILES" ]; then
    echo "Usage: bash sub-dv.sh <doi> <file_path>"
    echo "Example: bash sub-dv.sh doi:10.7910/DVN/NRZZRC 128/tension/3c"
    exit 1
fi

for i in $(seq 1 5); do
    echo "=== Run $i of 5 ==="
    java -jar DVUploader-v1.3.0-beta.jar \
        -key=f61c98d4-6808-476c-96d6-2db94be2e499 \
        -did=$DOI \
        -server=https://dataverse.harvard.edu \
        -limit=100 \
        $FILES
done
