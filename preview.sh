#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-4000}"

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
