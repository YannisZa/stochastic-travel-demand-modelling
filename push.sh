#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mPushing updates to GitHub...\033[0m\n"

# Add changes to git.
git add .

# Commit changes.
msg="rebuilding offline site $(date)"
if [ -n "$*" ]; then
	msg="$1"
fi
git commit -m "$msg"

# Define branch
$branch="$2"

# Push source and build repos.
git push origin $branch
