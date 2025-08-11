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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.c.Data;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Builder for merge insert operations.
 * This class collects all configuration options and executes the operation in a single call to Rust.
 */
public class MergeInsertBuilder implements AutoCloseable {
    private final long datasetHandle;
    private final List<String> onColumns;
    private final BufferAllocator allocator;
    
    // Configuration options
    private String whenMatchedConfig = "do_nothing";
    private String whenNotMatchedConfig = "do_nothing";
    private String whenNotMatchedBySourceConfig = "do_nothing";
    private int maxRetries = 0;
    private long timeoutMillis = 0;
    
    // Native method declarations
    private static native Object executeWithConfigNative(
        long datasetHandle,
        String[] onColumns,
        String whenMatchedConfig,
        String whenNotMatchedConfig,
        String whenNotMatchedBySourceConfig,
        int maxRetries,
        long timeoutMillis,
        long streamAddress
    );
    
    static {
        System.loadLibrary("lance_jni");
    }
    
    /**
     * Creates a new MergeInsertBuilder.
     * 
     * @param datasetHandle Native handle to the dataset
     * @param onColumns Columns to match on
     * @param allocator Buffer allocator for Arrow operations
     */
    MergeInsertBuilder(long datasetHandle, List<String> onColumns, BufferAllocator allocator) {
        this.datasetHandle = datasetHandle;
        this.onColumns = onColumns;
        this.allocator = allocator;
    }
    
    /**
     * Configure what to do when records match.
     * 
     * @param config Configuration string: "update_all", "do_nothing", or a condition expression
     * @return this builder
     */
    public MergeInsertBuilder whenMatched(String config) {
        this.whenMatchedConfig = config;
        return this;
    }
    
    /**
     * Configure what to do when records don't match.
     * 
     * @param config Configuration string: "insert_all", "do_nothing"
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatched(String config) {
        this.whenNotMatchedConfig = config;
        return this;
    }
    
    /**
     * Configure what to do when source records don't match.
     * 
     * @param config Configuration string: "delete", "do_nothing", or a condition expression
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedBySource(String config) {
        this.whenNotMatchedBySourceConfig = config;
        return this;
    }
    
    /**
     * Set the maximum number of retries for conflicts.
     * 
     * @param maxRetries Maximum number of retries
     * @return this builder
     */
    public MergeInsertBuilder conflictRetries(int maxRetries) {
        this.maxRetries = maxRetries;
        return this;
    }
    
    /**
     * Set the retry timeout in milliseconds.
     * 
     * @param timeoutMillis Timeout in milliseconds
     * @return this builder
     */
    public MergeInsertBuilder retryTimeout(long timeoutMillis) {
        this.timeoutMillis = timeoutMillis;
        return this;
    }
    
    /**
     * Execute the merge insert operation with the current configuration.
     * 
     * @param root VectorSchemaRoot containing the data to merge
     * @return MergeInsertResult with operation statistics
     * @throws RuntimeException if the operation fails
     */
    public MergeInsertResult execute(VectorSchemaRoot root) {
        try {
            // Convert VectorSchemaRoot to ArrowArrayStream
            long streamAddress = convertToStream(root);
            
            // Execute with all configuration
            Object result = executeWithConfigNative(
                datasetHandle,
                onColumns.toArray(new String[0]),
                whenMatchedConfig,
                whenNotMatchedConfig,
                whenNotMatchedBySourceConfig,
                maxRetries,
                timeoutMillis,
                streamAddress
            );
            
            return (MergeInsertResult) result;
        } catch (Exception e) {
            throw new RuntimeException("Failed to execute merge insert", e);
        }
    }
    
    /**
     * Convert VectorSchemaRoot to ArrowArrayStream for native execution.
     */
    private long convertToStream(VectorSchemaRoot root) throws IOException {
        // Write VectorSchemaRoot to ByteArrayOutputStream using ArrowStreamWriter
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, baos)) {
            writer.start();
            writer.writeBatch();
            writer.end();
        }
        
        // Read from ByteArrayInputStream using ArrowStreamReader
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ArrowStreamReader reader = new ArrowStreamReader(bais, allocator);
        
        // Export ArrowReader to ArrowArrayStream and get the memory address
        try (org.apache.arrow.c.ArrowArrayStream stream = org.apache.arrow.c.ArrowArrayStream.allocateNew(allocator)) {
            Data.exportArrayStream(allocator, reader, stream);
            return stream.memoryAddress();
        }
    }
    
    @Override
    public void close() {
        // No cleanup needed for the new design
    }
}
