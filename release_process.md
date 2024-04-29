# Release process

We create a full release of Lance up to every 4 weeks. In between full releases,
we make preview releases of the latest features and bug fixes, which are hosted
on fury.io. This allows us to release frequency and get feedback on new features
while keeping under the PyPI project size limits.

## Make a preview release

First, make sure the CI on main is green.

Trigger the `Create release commit` action, with default parameters. You can
use the default parameters.

This will create a tag on the current main with format `vX.Y.Z-beta.N`. After
creating the tag, the action will create a GitHub release for the new tag.
Once that release is published, it will trigger publish jobs for Rust and Python.

## Make a full release

First, make sure the CI on main is green.

Trigger the `Create release commit` action, with `type` set to `stable`.

This will create a tag on the current main with format `vX.Y.Z`. After creating
the tag, the action will create a GitHub release for the new tag. In addition,
it will increment the patch version in the `Cargo.toml` and `pyproject.toml` files.

Once that release is published, it will trigger publish jobs for Rust and Python.

## Incrementing the minor version.

We increment the patch version whenever we make a release. However, sometimes we
want to increment the minor version, such as when there is a breaking change
or a major new feature. This should be done manually in the PR that introduces
the breaking change or new feature. Be sure to check that we haven't already
incremented the minor version in the current release cycle.
