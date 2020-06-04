#!/bin/sh

# If a command fails then the deploy stops
set -e

printf "\033[0;32mPushing updates to GitHub...\033[0m\n"

# Add changes to git.
git add .

# Commit changes.
msg="commiting to project $(date)"
if [ -n "$*" ]; then
	msg="$*"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin master
