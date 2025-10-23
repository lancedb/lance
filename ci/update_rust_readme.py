#!/usr/bin/env python3
"""
Script to update rust/README.md with cargo metadata information.

This script:
1. Extracts workspace crate information using `cargo metadata`
2. Generates a mermaid diagram showing dependencies between crates
3. Generates a list of crates with descriptions
4. Updates rust/README.md between comment markers
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


def get_cargo_metadata() -> dict:
    """Run cargo metadata and return parsed JSON."""
    result = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def get_workspace_crates(metadata: dict) -> Dict[str, dict]:
    """Extract workspace crates and their information."""
    workspace_members = set(metadata["workspace_members"])
    crates = {}

    for package in metadata["packages"]:
        pkg_id = package["id"]
        if pkg_id in workspace_members:
            # Extract crate name from path
            manifest_path = Path(package["manifest_path"])
            # Get relative path from workspace root
            rel_path = manifest_path.parent.relative_to(Path(metadata["workspace_root"]))

            crates[package["name"]] = {
                "description": package.get("description", ""),
                "path": str(rel_path),
                "dependencies": [],
            }

    # Now extract dependencies between workspace crates
    for package in metadata["packages"]:
        if package["id"] not in workspace_members:
            continue

        pkg_name = package["name"]
        for dep in package["dependencies"]:
            dep_name = dep["name"]
            # Only include dependencies that are workspace members
            if dep_name in crates:
                crates[pkg_name]["dependencies"].append(dep_name)

    return crates


def generate_mermaid_diagram(crates: Dict[str, dict]) -> str:
    """Generate a mermaid diagram showing crate dependencies."""
    lines = [
        "<!-- This section is auto-generated. Run `python ci/update_rust_readme.py` to update. -->",
        "```mermaid",
        "graph TD",
    ]

    # Sort crates by name for consistent output
    sorted_crates = sorted(crates.keys())

    # Add nodes with descriptions as tooltips
    for crate_name in sorted_crates:
        crate_info = crates[crate_name]
        # Use short name for node label
        node_label = crate_name.replace("lance-", "")
        lines.append(f'    {crate_name}["{node_label}"]')

    lines.append("")

    # Add edges
    edges: Set[tuple] = set()
    for crate_name in sorted_crates:
        for dep in crates[crate_name]["dependencies"]:
            if dep in crates:  # Only workspace dependencies
                edges.add((crate_name, dep))

    for src, dest in sorted(edges):
        lines.append(f"    {src} --> {dest}")

    lines.append("```")
    return "\n".join(lines)


def generate_crate_list(crates: Dict[str, dict]) -> str:
    """Generate a markdown list of crates with descriptions."""
    lines = [
        "<!-- This section is auto-generated. Run `python ci/update_rust_readme.py` to update. -->",
        "## Workspace Crates",
        "",
    ]

    # Sort crates by name
    sorted_crates = sorted(crates.items())

    for crate_name, info in sorted_crates:
        description = info["description"] or "No description available"
        path = info["path"]
        lines.append(f"- **{crate_name}** (`{path}/`) - {description}")

    return "\n".join(lines)


def update_readme(diagram: str, crate_list: str, readme_path: Path):
    """Update README.md with generated content between markers."""
    if not readme_path.exists():
        print(f"Error: {readme_path} does not exist", file=sys.stderr)
        sys.exit(1)

    content = readme_path.read_text()

    # Define markers
    DIAGRAM_START = "<!-- BEGIN_CARGO_DIAGRAM -->"
    DIAGRAM_END = "<!-- END_CARGO_DIAGRAM -->"
    LIST_START = "<!-- BEGIN_CARGO_CRATE_LIST -->"
    LIST_END = "<!-- END_CARGO_CRATE_LIST -->"

    # Update diagram section
    if DIAGRAM_START in content and DIAGRAM_END in content:
        start_idx = content.index(DIAGRAM_START) + len(DIAGRAM_START)
        end_idx = content.index(DIAGRAM_END)
        content = (
            content[:start_idx]
            + "\n" + diagram + "\n"
            + content[end_idx:]
        )
    else:
        print(f"Warning: Could not find diagram markers in {readme_path}", file=sys.stderr)
        print(f"Add these markers to your README:\n{DIAGRAM_START}\n{DIAGRAM_END}", file=sys.stderr)

    # Update crate list section
    if LIST_START in content and LIST_END in content:
        start_idx = content.index(LIST_START) + len(LIST_START)
        end_idx = content.index(LIST_END)
        content = (
            content[:start_idx]
            + "\n" + crate_list + "\n"
            + content[end_idx:]
        )
    else:
        print(f"Warning: Could not find crate list markers in {readme_path}", file=sys.stderr)
        print(f"Add these markers to your README:\n{LIST_START}\n{LIST_END}", file=sys.stderr)

    readme_path.write_text(content)
    print(f"Updated {readme_path}")


def main():
    # Get workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    readme_path = workspace_root / "rust" / "README.md"

    # Get cargo metadata
    print("Fetching cargo metadata...")
    metadata = get_cargo_metadata()

    # Extract workspace crates
    print("Extracting workspace crates...")
    crates = get_workspace_crates(metadata)
    print(f"Found {len(crates)} workspace crates")

    # Generate content
    print("Generating mermaid diagram...")
    diagram = generate_mermaid_diagram(crates)

    print("Generating crate list...")
    crate_list = generate_crate_list(crates)

    # Update README
    print(f"Updating {readme_path}...")
    update_readme(diagram, crate_list, readme_path)

    print("Done!")


if __name__ == "__main__":
    main()
