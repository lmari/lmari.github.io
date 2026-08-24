#!/usr/bin/env bash
set -euo pipefail

branch="$(git branch --show-current)"
if [[ "$branch" != "redesign-2026" ]]; then
  echo "Refusing to run: current branch is '$branch', expected 'redesign-2026'." >&2
  exit 1
fi

python tools/preprod_cleanup.py

# chatting/index.html is a Jekyll wrapper: its sidebar lives in the included
# source file, so ensure the wrapper itself carries the navigation metadata.
python - <<'PY'
from pathlib import Path
import re
p = Path('chatting/index.html')
s = p.read_text(encoding='utf-8')
m = re.match(r'\A---\s*\n(.*?)\n---\s*\n', s, re.S)
body = s[m.end():] if m else s
old = m.group(1).splitlines() if m else []
old = [line for line in old if not re.match(r'^\s*(section|root)\s*:', line)]
fm = ['section: chatting', 'root: "../"'] + old
p.write_text('---\n' + '\n'.join(fm).strip() + '\n---\n' + body, encoding='utf-8')
PY

# The build is the final structural gate before committing anything.
bundle exec jekyll build

git diff --check

# Keep build products and this one-shot runner out of the commit.
rm -rf _site
rm -- "$0"

git add -A
if git diff --cached --quiet; then
  echo "No pre-production changes to commit."
  exit 0
fi

git config user.name "Luca Mari"
git config user.email "lmari@liuc.it"
git commit -m "Complete pre-production site cleanup"
git push origin redesign-2026

echo "Pre-production cleanup complete and pushed to redesign-2026."
