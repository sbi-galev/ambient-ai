#!/usr/bin/env bash
# Build the static archive and publish it to the `gh-pages` branch, which
# GitHub Pages serves directly (Settings -> Pages -> Deploy from a branch ->
# gh-pages / root). The built site/ is git-ignored on the main branch; this
# script force-pushes a single fresh commit to gh-pages, so that branch only
# ever holds the latest generated site (no history to scrub, no Actions
# workflow involved). GitHub rebuilds Pages automatically on each push.
#
# Usage:  ./publish_site.sh [extra export_static.py flags]
# e.g.    ./publish_site.sh --no-llm
set -euo pipefail
cd "$(dirname "$0")"

BRANCH=gh-pages
REMOTE=$(git remote get-url origin)
NAME=$(git config user.name || echo "site publisher")
EMAIL=$(git config user.email || echo "site@localhost")

# 1. Build the archive into ./site (git-ignored). --clean wipes it first.
python3 export_static.py --clean "$@"

# 2. Publish site/ as the new gh-pages tip. We use a throwaway repo inside
#    site/ (which the parent .gitignore excludes) and force-push it.
cd site
rm -rf .git
git init -q
git add -A
git -c user.name="$NAME" -c user.email="$EMAIL" commit -qm "Update static archive"
git push -f "$REMOTE" "HEAD:$BRANCH"
rm -rf .git

echo "Published $(ls -1 *.html 2>/dev/null | wc -l) page(s) to '$BRANCH'."
