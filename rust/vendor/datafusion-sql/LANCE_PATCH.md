# Lance DataFusion SQL patch

This directory contains the `datafusion-sql` 54.1.0 crate source with the fix
from [apache/datafusion#23058](https://github.com/apache/datafusion/pull/23058)
backported. Remove this vendored crate and the three Cargo patch entries after a
DataFusion release includes that fix.
