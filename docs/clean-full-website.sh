#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
docs_src="$script_dir/src"

rm -rf "$docs_src/format/catalog"
rm -rf "$docs_src/format/namespace"
rm -f "$docs_src/format/layout.png"
rm -rf "$docs_src/integrations/huggingface"
rm -rf "$docs_src/integrations/duckdb"
rm -rf "$docs_src/integrations/spark"
rm -rf "$docs_src/integrations/ray"
rm -rf "$docs_src/integrations/trino"
rm -f "$docs_src/community/project-specific/.pages"
rm -rf "$docs_src/community/project-specific/lance"
rm -f "$docs_src/community/project-specific/namespace.md"
rm -f "$docs_src/community/project-specific/namespace-impls.md"
rm -f "$docs_src/community/project-specific/ray.md"
rm -f "$docs_src/community/project-specific/spark.md"
rm -f "$docs_src/community/project-specific/trino.md"

git -C "$repo_root" restore --source=HEAD --worktree docs/src/format/.pages docs/src/integrations/.pages 2>/dev/null || true
