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
