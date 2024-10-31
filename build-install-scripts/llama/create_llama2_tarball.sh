#!/bin/bash

set -x

# Function to display usage
usage() {
  echo "Usage: $0 --tarball-name <tarball-name> [--commit-ver <commit-version>] --files <file1> <file2> ..."
  exit 1
}

# Parse arguments
TARBALL_NAME=""
COMMIT_VER=""
FILES=()

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --tarball-name) TARBALL_NAME="$2"; shift ;;
    --commit-ver) COMMIT_VER="$2"; shift ;;
    --files) shift; while [[ "$1" && "$1" != --* ]]; do FILES+=("$1"); shift; done ;;
    *) echo "Unknown parameter: $1"; usage ;;
  esac
  shift
done

# Check if tarball name and files are provided
if [ -z "$TARBALL_NAME" ] || [ ${#FILES[@]} -eq 0 ]; then
  usage
fi

# Check if we are inside a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "This script must be run inside a git repository."
  exit 1
fi

# Get the top-level directory of the repository
TOP_LEVEL_DIR=$(git rev-parse --show-toplevel)

# Record current commit hash
CURRENT_COMMIT=$(git rev-parse HEAD)

# If a commit version is specified, checkout that version
if [ -n "$COMMIT_VER" ]; then
  git -C "$TOP_LEVEL_DIR" checkout "$COMMIT_VER" || { echo "Failed to checkout commit $COMMIT_VER"; exit 1; }
fi

# Create a NOTE file with the commit version
NOTE_CONTENT="Commit version: ${COMMIT_VER:-$CURRENT_COMMIT}"
echo "$NOTE_CONTENT" > NOTE

# Create the tarball including specified files and the NOTE file
tar -czf "$TARBALL_NAME" NOTE "${FILES[@]}"


rm NOTE

# go back to the original commit
if [ -n "$COMMIT_VER" ]; then
  git -C "$TOP_LEVEL_DIR" checkout "$CURRENT_COMMIT"
fi

echo "Tarball $TARBALL_NAME created successfully."

