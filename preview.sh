#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-4000}"
BUNDLE_PATH="${BUNDLE_PATH:-vendor/bundle}"
export BUNDLE_PATH

if ! bundle check >/dev/null 2>&1; then
  printf 'Jekyll dependencies are not available in %s.\n' "$BUNDLE_PATH" >&2
  printf 'Install them once with: BUNDLE_PATH=%q bundle install\n' "$BUNDLE_PATH" >&2
  exit 1
fi

printf 'Jekyll preview: http://127.0.0.1:%s\n' "$PORT"
if command -v hostname >/dev/null 2>&1; then
  for ip in $(hostname -I 2>/dev/null || true); do
    case "$ip" in
      127.*|::1) ;;
      *) printf 'LAN preview:    http://%s:%s\n' "$ip" "$PORT" ;;
    esac
  done
fi

bundle exec jekyll serve --livereload --host "$HOST" --port "$PORT"
