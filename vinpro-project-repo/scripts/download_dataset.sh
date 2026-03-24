#!/usr/bin/env bash
# Download the 3D2cut Single Guyot dataset.
#
# The dataset is hosted at: https://doi.org/10.34777/azf6-tm83
# License: CC BY-NC-SA
#
# Since the dataset is hosted on a DOI-based repository, automatic
# download may not be straightforward. This script provides instructions.

set -e

TARGET_DIR="${1:-./data/3D2cut_Single_Guyot}"

echo "============================================="
echo "  3D2cut Single Guyot Dataset Download"
echo "============================================="
echo ""
echo "The dataset is publicly available for academic research at:"
echo "  https://doi.org/10.34777/azf6-tm83"
echo ""
echo "License: CC BY-NC-SA"
echo ""
echo "Please download the dataset manually from the above link"
echo "and extract it to: ${TARGET_DIR}"
echo ""
echo "Expected structure after extraction:"
echo "  ${TARGET_DIR}/"
echo "  ├── 01-TrainAndValidationSet/"
echo "  │   ├── *.jpg"
echo "  │   └── *_annotation.json"
echo "  └── 02-IndependentTestSet/"
echo "      ├── *.jpg"
echo "      └── *_annotation.json"
echo ""

# Create target directory
mkdir -p "${TARGET_DIR}"
echo "Created directory: ${TARGET_DIR}"
echo ""
echo "After downloading, you can train with:"
echo "  python train.py --data_path ${TARGET_DIR}"
