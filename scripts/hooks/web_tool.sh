#!/usr/bin/env bash
# Run a web/ dev tool (eslint or prettier) on files passed by pre-commit with repo-root paths.
# Usage: scripts/hooks/web_tool.sh <tool> [tool args...] -- <files...>
set -euo pipefail

tool="$1"
shift
tool_args=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  tool_args+=("$1")
  shift
done
[[ $# -gt 0 ]] && shift

files=()
for f in "$@"; do
  files+=("${f#web/}")
done
[[ ${#files[@]} -eq 0 ]] && exit 0

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/web"
if [[ ! -x "node_modules/.bin/$tool" ]]; then
  echo "web/node_modules is missing; run 'npm ci' in web/ before committing." >&2
  exit 1
fi
exec "node_modules/.bin/$tool" "${tool_args[@]}" "${files[@]}"
