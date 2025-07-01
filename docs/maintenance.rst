Maintenance
===========

To maintain good performance and manageability of your Lance datasets, you will
want to periodically run the following maintenance tasks:

1. **Compact files** - Reduces file count and improves scan performance
2. **Optimize indices** - Updates indices with new data for faster queries
3. **Cleanup old versions** - Reduces disk usage by removing old data

Compact Files
-------------

**Purpose:** Compacts small files in the dataset, reducing the total number of files and improving scan performance.

**API:** :meth:`lance.dataset.DatasetOptimizer.compact_files`.

**When to run:**

* After many small writes or updates
* When you notice degraded scan performance
* When fragments have a high percentage of deleted rows

**Effects:**

* **Reduces file count** - Merges small fragments into larger ones
* **Removes deleted rows** - Permanently removes soft-deleted data from files
* **Improves scan performance** - Fewer files to read means faster scans

**Usage:**

.. code-block:: python

    import lance

    dataset = lance.dataset("/path/to/dataset")
    
    # Basic compaction
    metrics = dataset.optimize.compact_files()
    
    # Advanced compaction with custom settings
    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=1024*1024,  # Target 1M rows per fragment
        materialize_deletions=True,          # Remove soft-deleted rows
        materialize_deletions_threshold=0.1, # Compact if >10% rows deleted
        num_threads=4                        # Use 4 threads for compaction
    )
    
    print(f"Compacted {metrics.fragments_removed} fragments")

Optimize Indices
----------------

**Purpose:** Optimizes index performance by adding new data to existing indexes.

**API:** :meth:`lance.dataset.DatasetOptimizer.optimize_indices`.

**When to run:**

* After adding significant amounts of new data or updating existing data

**Effects:**

* **Reduced search latency** - Eliminates expensive unindexed searches on new data

**Usage:**

.. code-block:: python

    import lance

    dataset = lance.dataset("/path/to/dataset")
    
    # Optimize all indices with default settings
    dataset.optimize.optimize_indices()
    
    # Optimize specific indices
    dataset.optimize.optimize_indices(
        index_names=["vector_index", "my_scalar_index"]
    )
    
    # Create new delta index for new data
    dataset.optimize.optimize_indices(num_indices_to_merge=0)
    
    # Retrain entire index (slower but more accurate)
    dataset.optimize.optimize_indices(retrain=True)


Cleanup Old Versions
--------------------

**Purpose:** Removes old dataset versions and their associated data files to free up disk space.

**API:** :meth:`lance.dataset.LanceDataset.cleanup_old_versions`.

**When to run:**

* When disk usage becomes a concern
* As part of regular maintenance to control storage costs
* When you're confident old versions are no longer needed

**Effects:**

* **Reduces disk usage** - Permanently removes old data files
* **Frees up storage space** - Reclaims space from previous dataset versions
* **Irreversible data loss** - Old versions cannot be restored after cleanup

.. warning::
   **Concurrent Operations Warning:** Setting ``older_than`` to zero (or very small values) 
   is dangerous when there are concurrent write operations. This can lead to write failures,
   usually from "file not found" errors. Always use a safe time buffer (at least several
   minutes, preferably hours) when other processes might be writing to the dataset.

.. note::
    **Tagged Versions Protection:** Versions with tags will never be deleted, regardless of age.
    To remove a version that has been tagged, you must first delete the associated tag.

**Usage:**

.. code-block:: python

    import lance
    from datetime import timedelta

    dataset = lance.dataset("/path/to/dataset")
    
    # Cleanup versions older than 2 weeks (default)
    stats = dataset.cleanup_old_versions()
    
    # Cleanup versions older than 7 days
    stats = dataset.cleanup_old_versions(older_than=timedelta(days=7))
    
    # Advanced cleanup with custom settings
    stats = dataset.cleanup_old_versions(
        older_than=timedelta(days=7),
        delete_unverified=False,           # Don't delete unverified files
        error_if_tagged_old_versions=True  # Error if tagged versions would be deleted
    )
    
    print(f"Removed {stats.old_versions} old versions")
    print(f"Freed {stats.bytes_removed} bytes")


Auto-Cleanup
~~~~~~~~~~~~

**Purpose:** Automatically runs cleanup operations at specified intervals during write operations.

**API:**

 * :meth:`lance.dataset.DatasetOptimizer.enable_auto_cleanup`.
 * :meth:`lance.dataset.DatasetOptimizer.disable_auto_cleanup`.

**When to enable:**

* For datasets with frequent writes that you want to maintain automatically
* When you want to prevent unlimited disk growth without manual intervention
* For production systems where regular maintenance might be forgotten

**Effects:**

* **Automatic disk management** - Prevents unlimited storage growth
* **Possible write delays** - Cleanup operations may slow down write operations
* **Reduced manual maintenance** - Eliminates need for manual cleanup scheduling

**Usage:**

.. code-block:: python

    import lance
    from lance.dataset import AutoCleanupConfig

    # Enable auto-cleanup when creating a new dataset
    auto_cleanup_config = AutoCleanupConfig(
        interval=20,              # Run cleanup every 20 commits
        older_than_seconds=14*24*60*60  # Keep versions for 14 days
    )
    
    lance.write_dataset(
        data, 
        "/path/to/dataset",
        auto_cleanup_options=auto_cleanup_config
    )
    
    # Enable auto-cleanup on existing dataset
    dataset = lance.dataset("/path/to/dataset")
    dataset.optimize.enable_auto_cleanup(auto_cleanup_config)
    
    # Disable auto-cleanup
    dataset.optimize.disable_auto_cleanup()

**Configuration:**

- ``interval`` - Number of commits between cleanup operations
- ``older_than_seconds`` - Age threshold in seconds for version cleanup

.. note::
   **Auto-Cleanup Behavior:** Auto-cleanup runs in the background during write operations, so it may not immediately free up space. However, it will prevent unbounded growth of old versions over time.

