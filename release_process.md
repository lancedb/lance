# Release process

We create a full release of Lance up to every 2 weeks. In between full releases,
we make preview releases of the latest features and bug fixes, which are hosted
on fury.io. This allows us to release frequently and get feedback on new features
while keeping under the PyPI project size limits.

## Choosing a full versus preview release

There are three conditions that can trigger a full release:

1. There's a bugfix we urgently want to get out to a broad audience
2. We want to make a release of LanceDB that requires new features from Lance
   (LanceDB can't depend on a preview release of Lance)
3. It's been two weeks since we last released a full release.

Otherwise, we should make a preview release.

## Make a preview release

First, make sure the CI on main is green.

Trigger the `Create release commit` action, with default parameters. You can
use the default parameters.

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

Trigger the `Create release commit` action, with `type` set to `stable`.

This will create a tag on the current main with format `vX.Y.Z`. After creating
the tag, the action will create a GitHub release for the new tag. In addition,
it will increment the patch version in the `Cargo.toml` and `pyproject.toml` files.

The action will automatically generate release notes. **Please review these
and make any necessary edits.**

Once that release is published, it will trigger publish jobs for Rust and Python.

## Incrementing the minor version.

We increment the patch version whenever we make a release. However, sometimes we
want to increment the minor version, such as when there is a breaking change
or a major new feature. This should be done manually in the PR that introduces
the breaking change or new feature. Be sure to check that we haven't already
incremented the minor version in the current release cycle.


## Breaking Change policy

We try to avoid breaking changes, but sometimes they are necessary. When there
are breaking changes, we will increment the minor version. (This is valid 
semantic versioning because we are still in `0.x` versions.)

When a PR makes a breaking change, the PR author should mark the PR using the 
conventional commit markers: either exclamation mark after the type
(such as `feat!: change signature of func`) or have `BREAKING CHANGE` in the
body of the PR. A CI job will add a `breaking-change` label to the PR, which is
what will ultimately be used to CI to determine if the minor version should be
incremented.

A CI job will validate that if a `breaking-change` label is added, the minor
version is incremented in the `Cargo.toml` and `pyproject.toml` files. The only
exception is if it has already been incremented since the last stable release.

**It is the responsibility of the PR author to increment the minor version when
appropriate.**

Some things that are considered breaking changes:

* Upgrading a dependency pin that is in the Rust API. In particular, upgrading
  `DataFusion` and `Arrow` are breaking changes. Changing dependencies that are
  not exposed in our public API are not considered breaking changes.
* Changing the signature of a public function or method.
* Removing a public function or method.

We do make exceptions for APIs that are marked as experimental. These are APIs
that are under active development and not in major use. These changes should not
receive the `breaking-change` label.
