# Release process

We create a full release of Lance up to every 2 weeks. In between full releases,
we make preview releases of the latest features and bug fixes, which are hosted
on fury.io. This allows us to release frequently and get feedback on new features
while keeping under the PyPI project size limits.

## Overview of Automated Release Process

The Lance release process is now automated using `bump-my-version` to eliminate 
manual version updates. The workflow handles version bumping, breaking change 
validation, and release creation automatically.

## Choosing a full versus preview release

There are three conditions that can trigger a full release:

1. There's a bugfix we urgently want to get out to a broad audience
2. We want to make a release of LanceDB that requires new features from Lance
   (LanceDB can't depend on a preview release of Lance)
3. It's been two weeks since we last released a full release.

Otherwise, we should make a preview release.

## Make a preview release

First, make sure the CI on main is green.

Trigger the `Create release` action with the following parameters:
- **release_type**: Choose based on changes (patch/minor/major)
- **release_channel**: `preview`
- **dry_run**: `false` (use `true` to test first)
- **draft_release**: `true` (to review release notes before publishing)

This will create a tag on the current main with format `vX.Y.Z-beta.N`. After
creating the tag, the action will create a GitHub release for the new tag.
Once that release is published, it will trigger publish jobs for Python.

The action will automatically generate release notes. **Please review these
and make any necessary edits.**

> [!NOTE]
> Preview releases are not published to crates.io, since Rust is a source
> distribution. Users can simply point to the tag on GitHub in their `Cargo.toml`.

## Make a full release

First, make sure the CI on main is green.

Trigger the `Create release` action with the following parameters:
- **release_type**: Choose based on changes (patch/minor/major)
- **release_channel**: `stable`
- **dry_run**: `false` (use `true` to test first)
- **draft_release**: `true` (to review release notes before publishing)

The workflow will:
1. Check for breaking changes automatically
2. Update all version numbers using bump-my-version
3. Create a commit with the version update
4. Create a tag with format `vX.Y.Z`
5. Push both the commit and tag
6. Create a GitHub release

The action will automatically generate release notes. **Please review these
and make any necessary edits.**

Once that release is published, it will trigger publish jobs for Rust, Python, and Java.

## Version Management

### Automated Version Bumping

The release process now uses `bump-my-version` configured in `.bumpversion.toml` to:
- Synchronize versions across all Rust crates
- Update Python and Java package versions
- Update all Cargo.lock files automatically

### Release Types

- **patch**: Bug fixes and minor improvements (0.32.1 → 0.32.2)
- **minor**: New features or breaking changes (0.32.1 → 0.33.0)
- **major**: Major breaking changes (0.32.1 → 1.0.0)

The breaking change detection script (`scripts/check_breaking_changes.py`) will
prevent patch releases when breaking changes are detected.

## Breaking Change policy

We try to avoid breaking changes, but sometimes they are necessary. When there
are breaking changes, we will increment the minor version. (This is valid
semantic versioning because we are still in `0.x` versions.)

### Automatic Breaking Change Detection

The release workflow automatically detects breaking changes by:
- Analyzing commit messages for breaking change indicators
- Checking for changes in public Rust APIs
- Detecting migration files

When a PR makes a breaking change, the PR author should mark the PR using the
conventional commit markers: either exclamation mark after the type
(such as `feat!: change signature of func`) or have `BREAKING CHANGE` in the
body of the PR.

### What Constitutes a Breaking Change

Some things that are considered breaking changes:

* Upgrading a dependency pin that is in the Rust API. In particular, upgrading
  `DataFusion` and `Arrow` are breaking changes. Changing dependencies that are
  not exposed in our public API are not considered breaking changes.
* Changing the signature of a public function or method.
* Removing a public function or method.

We do make exceptions for APIs that are marked as experimental. These are APIs
that are under active development and not in major use. These changes should not
receive the `breaking-change` label.

## Local Testing

To test the release process locally:

```bash
# Install bump-my-version
pip install bump-my-version

# Test version bumping (dry run)
python ci/bump_version.py patch --dry-run

# Check for breaking changes
python ci/check_breaking_changes.py
```

## Troubleshooting

### Version Mismatch
If versions become out of sync:
```bash
python ci/bump_version.py patch --no-validate
```

### Failed Release
If a release fails:
1. Check the GitHub Actions logs
2. Fix any issues
3. Re-run with `dry_run: true` first
4. Once successful, run with `dry_run: false`

### Manual Version Update
If you need to update versions manually:
```bash
bump-my-version bump --new-version 0.33.0
cargo update -p lance  # Update lock files
```

## Key Files

- `.bumpversion.toml` - Configuration for version management
- `ci/bump_version.py` - Version update orchestration
- `ci/check_breaking_changes.py` - Breaking change detection
- `.github/workflows/make-release-commit.yml` - Main release workflow
- `.github/workflows/bump-version/action.yml` - Version bump action