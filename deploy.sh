#!/bin/bash

# Terminate the script if any command fails
set -e

echo "========== 1. Cleaning and Generating Static Files =========="
hexo clean
hexo g

echo "========== 2. Deploying to GitHub Pages Branch =========="
# This pushes the 'public' folder content to your production branch (e.g., main/master)
hexo d

echo "========== 3. Backing up Source Code to 'source' Branch =========="
# Adding all changes, including new posts and configuration tweaks
git add .

# Committing with a dynamic timestamp
current_time=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Site updated: $current_time"

# Pushing the raw project files to your backup branch
git push origin source

echo "========== SUCCESS: Blog deployed and Source backed up! =========="
# Pause for user to see the result
read -p "Press enter to close..."