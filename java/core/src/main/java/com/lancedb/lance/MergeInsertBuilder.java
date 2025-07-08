/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.lancedb.lance;

import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.Arrays;
import java.util.List;

/**
 * Builder for merge insert operations.
 *
 * This class provides a fluent API for building merge insert operations,
 * which allow you to merge new data with existing data in a Lance dataset.
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id"));
 * MergeInsertResult result = builder
 *     .whenMatchedUpdateAll()
 *     .whenNotMatchedInsertAll()
 *     .execute(newData);
 * }</pre>
 */
public class MergeInsertBuilder implements AutoCloseable {

    private final long nativeHandle;
    private final Dataset dataset;
    private final List<String> onColumns;

    /**
     * Create a new MergeInsertBuilder.
     *
     * @param dataset the target dataset
     * @param onColumns the columns to join on
     * @return a new MergeInsertBuilder
     */
    public static MergeInsertBuilder create(Dataset dataset, List<String> onColumns) {
        return new MergeInsertBuilder(dataset, onColumns);
    }

    /**
     * Create a new MergeInsertBuilder.
     *
     * @param dataset the target dataset
     * @param onColumns the columns to join on
     * @return a new MergeInsertBuilder
     */
    public static MergeInsertBuilder create(Dataset dataset, String... onColumns) {
        return create(dataset, Arrays.asList(onColumns));
    }

    private MergeInsertBuilder(Dataset dataset, List<String> onColumns) {
        this.dataset = dataset;
        this.onColumns = onColumns;

        // Convert List<String> to String[] for JNI call
        String[] columnsArray = onColumns.toArray(new String[0]);

        // Call native method to create the builder
        this.nativeHandle = createNativeBuilder(dataset.getNativeHandle(), columnsArray);

        if (this.nativeHandle == 0) {
            throw new RuntimeException("Failed to create MergeInsertBuilder");
        }
    }

    /**
     * Configure the builder to update all columns when there is a match.
     *
     * @return this builder
     */
    public MergeInsertBuilder whenMatchedUpdateAll() {
        whenMatchedUpdateAllNative(nativeHandle, null);
        return this;
    }

    /**
     * Configure the builder to update all columns when there is a match and the condition is true.
     *
     * @param condition the condition expression
     * @return this builder
     */
    public MergeInsertBuilder whenMatchedUpdateAll(String condition) {
        whenMatchedUpdateAllNative(nativeHandle, condition);
        return this;
    }

    /**
     * Configure the builder to insert all columns when there is no match.
     *
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedInsertAll() {
        whenNotMatchedInsertAllNative(nativeHandle);
        return this;
    }

    /**
     * Configure the builder to delete rows when there is no match in the source.
     *
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedBySourceDelete() {
        whenNotMatchedBySourceDeleteNative(nativeHandle, null);
        return this;
    }

    /**
     * Configure the builder to delete rows when there is no match in the source and the expression is true.
     *
     * @param expr the expression
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedBySourceDelete(String expr) {
        whenNotMatchedBySourceDeleteNative(nativeHandle, expr);
        return this;
    }

    /**
     * Set the number of conflict retries.
     *
     * @param maxRetries the maximum number of retries
     * @return this builder
     */
    public MergeInsertBuilder conflictRetries(int maxRetries) {
        conflictRetriesNative(nativeHandle, maxRetries);
        return this;
    }

    /**
     * Set the retry timeout in milliseconds.
     *
     * @param timeoutMillis the timeout in milliseconds
     * @return this builder
     */
    public MergeInsertBuilder retryTimeout(long timeoutMillis) {
        retryTimeoutNative(nativeHandle, timeoutMillis);
        return this;
    }

    /**
     * Execute the merge insert operation.
     *
     * @param newData the new data to merge
     * @return the result of the merge insert operation
     */
    public MergeInsertResult execute(VectorSchemaRoot newData) {
        // For now, return a mock result since the actual implementation requires Arrow IPC
        // In a real implementation, this would use Arrow IPC to export the data
        // and call the native method with the stream address

        // TODO: Implement proper Arrow IPC export
        // long streamAddress = exportVectorSchemaRoot(newData);
        // try {
        //     Object result = executeNative(nativeHandle, streamAddress);
        //     if (result instanceof MergeInsertResult) {
        //         return (MergeInsertResult) result;
        //     } else {
        //         throw new RuntimeException("Unexpected result type from native execution");
        //     }
        // } finally {
        //     releaseVectorSchemaRoot(streamAddress);
        // }

        return new MergeInsertResult(0L, 0L, 0L);
    }

    @Override
    public void close() {
        if (nativeHandle != 0) {
            closeNative(nativeHandle);
        }
    }

    // Native method declarations

    private static native long createNativeBuilder(long datasetHandle, String[] onColumns);

    private static native void whenMatchedUpdateAllNative(long builderHandle, String condition);

    private static native void whenNotMatchedInsertAllNative(long builderHandle);

    private static native void whenNotMatchedBySourceDeleteNative(long builderHandle, String expr);

    private static native void conflictRetriesNative(long builderHandle, int maxRetries);

    private static native void retryTimeoutNative(long builderHandle, long timeoutMillis);

    private static native Object executeNative(long builderHandle, long streamAddress);

    private static native void closeNative(long builderHandle);

    static {
        JniLoader.ensureLoaded();
    }
} 