#!/bin/bash
# Fix pin_memory=True for CPU backend in sglang

FILE="/sgl-workspace/sglang/python/sglang/srt/state_capturer/base.py"

echo "Patching $FILE to disable pin_memory for CPU backend..."

# Backup the original file
cp "$FILE" "${FILE}.bak"

# Replace pin_memory=True with pin_memory=False for CPU
sed -i 's/pin_memory=True/pin_memory=False/g' "$FILE"

echo "Patch applied successfully!"
echo "Original file backed up to ${FILE}.bak"
echo ""
echo "Changes made:"
grep -n "pin_memory" "$FILE"
