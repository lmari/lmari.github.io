#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4000}"

bundle exec jekyll serve --livereload --host "$HOST" --port "$PORT"
