#!/usr/bin/env bash
# Download the 3D2cut Single Guyot dataset from Zenodo.
#
# Source:  https://doi.org/10.34777/azf6-tm83
# Host:    https://zenodo.org/records/7679898
# License: CC BY-NC-SA 4.0
# Size:    ~4.4 GB compressed
#
# Usage:
#   ./scripts/download_dataset.sh              # default: ../dataset/3D2cut_Single_Guyot
#   ./scripts/download_dataset.sh /my/path     # custom target directory

set -e

TARGET_DIR="${1:-../dataset/3D2cut_Single_Guyot}"
ZENODO_URL="https://zenodo.org/records/7679898/files/3D2cut_Single_Guyot.tar.gz?download=1"
ARCHIVE="3D2cut_Single_Guyot.tar.gz"
EXPECTED_MD5="cef72f9661b82a05c385dcc69a05bc13"

echo "============================================="
echo "  3D2cut Single Guyot Dataset Download"
echo "============================================="
echo ""
echo "  Source:  https://doi.org/10.34777/azf6-tm83"
echo "  License: CC BY-NC-SA 4.0"
echo "  Size:    ~4.4 GB compressed"
echo "  Target:  ${TARGET_DIR}"
echo ""

# Skip if dataset already exists
if [ -d "${TARGET_DIR}/01-TrainAndValidationSet" ] && [ -d "${TARGET_DIR}/02-IndependentTestSet" ]; then
    TRAIN_COUNT=$(find "${TARGET_DIR}/01-TrainAndValidationSet" -name "*.jpg" -o -name "*.jpeg" | wc -l)
    TEST_COUNT=$(find "${TARGET_DIR}/02-IndependentTestSet" -name "*.jpg" -o -name "*.jpeg" | wc -l)
    echo "Dataset already exists at ${TARGET_DIR}"
    echo "  Train images: ${TRAIN_COUNT}"
    echo "  Test images:  ${TEST_COUNT}"
    echo ""
    echo "To re-download, remove the directory first:"
    echo "  rm -rf ${TARGET_DIR}"
    exit 0
fi

# Create target directory
mkdir -p "${TARGET_DIR}"

# Download
echo "Downloading from Zenodo (~4.4 GB)..."
wget --progress=bar:force -O "${TARGET_DIR}/${ARCHIVE}" "${ZENODO_URL}"

# Verify checksum
echo ""
echo "Verifying MD5 checksum..."
ACTUAL_MD5=$(md5sum "${TARGET_DIR}/${ARCHIVE}" | awk '{print $1}')
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
    echo "WARNING: MD5 mismatch!"
    echo "  Expected: ${EXPECTED_MD5}"
    echo "  Got:      ${ACTUAL_MD5}"
    echo "  The file may be corrupted. Continuing anyway..."
else
    echo "  Checksum OK: ${EXPECTED_MD5}"
fi

# Extract
echo ""
echo "Extracting..."
tar -xzf "${TARGET_DIR}/${ARCHIVE}" -C "${TARGET_DIR}" --strip-components=1

# Clean up archive
rm -f "${TARGET_DIR}/${ARCHIVE}"

# Verify
TRAIN_COUNT=$(find "${TARGET_DIR}/01-TrainAndValidationSet" -name "*.jpg" -o -name "*.jpeg" | wc -l)
TEST_COUNT=$(find "${TARGET_DIR}/02-IndependentTestSet" -name "*.jpg" -o -name "*.jpeg" | wc -l)

echo ""
echo "============================================="
echo "  Done!"
echo "============================================="
echo "  Location:     ${TARGET_DIR}"
echo "  Train images: ${TRAIN_COUNT}"
echo "  Test images:  ${TEST_COUNT}"
echo ""
echo "Usage:"
echo "  python train.py --data_path ${TARGET_DIR}"
echo "  python evaluate.py --data_path ${TARGET_DIR} --checkpoint path/to/model.pt"