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
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;  
import org.apache.arrow.vector.ipc.ArrowStreamWriter;  
import java.io.ByteArrayOutputStream;  
import java.nio.channels.Channels;  
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.time.Duration;

public class MergeInsertBuilder implements AutoCloseable {
    private long nativeBuilderHandle;
    private Dataset dataset;
    private List<String> onColumns;
    private BufferAllocator allocator;
    
    // 私有构造函数，通过 Dataset.mergeInsert() 创建
    MergeInsertBuilder(Dataset dataset, List<String> onColumns) {
        this.dataset = dataset;
        this.onColumns = onColumns;
        this.allocator = dataset.getAllocator();
        this.nativeBuilderHandle = createNativeBuilder(
            dataset.getNativeHandle(), 
            onColumns.toArray(new String[0])
        );
    }
    
    /**
     * 配置匹配行的更新行为
     * @param condition 可选的 SQL 条件表达式
     * @return this builder
     */
    public MergeInsertBuilder whenMatchedUpdateAll(String condition) {
        whenMatchedUpdateAllNative(nativeBuilderHandle, condition);
        return this;
    }
    
    /**
     * 配置未匹配行的插入行为
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedInsertAll() {
        whenNotMatchedInsertAllNative(nativeBuilderHandle);
        return this;
    }
    
    /**
     * 配置源表中未匹配行的删除行为
     * @param expr 可选的删除条件表达式
     * @return this builder
     */
    public MergeInsertBuilder whenNotMatchedBySourceDelete(String expr) {
        whenNotMatchedBySourceDeleteNative(nativeBuilderHandle, expr);
        return this;
    }
    
    /**
     * 设置冲突重试次数
     * @param maxRetries 最大重试次数，默认 10
     * @return this builder
     */
    public MergeInsertBuilder conflictRetries(int maxRetries) {
        conflictRetriesNative(nativeBuilderHandle, maxRetries);
        return this;
    }
    
    /**
     * 设置重试超时时间
     * @param timeout 超时时间，默认 30 秒
     * @return this builder
     */
    public MergeInsertBuilder retryTimeout(Duration timeout) {
        retryTimeoutNative(nativeBuilderHandle, timeout.toMillis());
        return this;
    }
    
    /**
     * 执行 merge insert 操作
     * @param stream Arrow 数据流
     * @return 合并统计信息
     */
    public MergeInsertResult execute(ArrowArrayStream stream) {
        return executeNative(nativeBuilderHandle, stream.memoryAddress());
    }
    
    /**
     * 执行 merge insert 操作（VectorSchemaRoot 版本）
     * @param root Arrow 数据
     * @return 合并统计信息
     */
    public MergeInsertResult execute(VectorSchemaRoot root) {
        try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            // 将 VectorSchemaRoot 转换为 ArrowArrayStream
            convertToStream(root, stream);
            return execute(stream);
        }
    }
    
    //VectorSchemaRoot 到 ArrowArrayStream 转换
    private static void convertToStream(VectorSchemaRoot root, ArrowArrayStream stream) {  
        try {  
            // 创建内存输出流  
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();  
          
            // 使用 ArrowStreamWriter 写入数据  
            try (ArrowStreamWriter writer = new ArrowStreamWriter(  
                    root,   
                    null,   
                    Channels.newChannel(outputStream))) {  
              
                writer.start();  
                writer.writeBatch();  
                writer.end();  
            }   
          
            // 将字节数组转换为 ArrowArrayStream  
            byte[] data = outputStream.toByteArray();  
            Data.exportArrayStream(root.getAllocator(), root, stream);  
          
        } catch (Exception e) {  
            throw new RuntimeException("Failed to convert VectorSchemaRoot to ArrowArrayStream", e);  
        }  
    }
    
    @Override
    public void close() {
        if (nativeBuilderHandle != 0) {
            closeNative(nativeBuilderHandle);
            nativeBuilderHandle = 0;
        }
    }
    
    // Native 方法声明
    private static native long createNativeBuilder(long datasetHandle, String[] onColumns);
    private static native void whenMatchedUpdateAllNative(long builderHandle, String condition);
    private static native void whenNotMatchedInsertAllNative(long builderHandle);
    private static native void whenNotMatchedBySourceDeleteNative(long builderHandle, String expr);
    private static native void conflictRetriesNative(long builderHandle, int maxRetries);
    private static native void retryTimeoutNative(long builderHandle, long timeoutMillis);
    private static native MergeInsertResult executeNative(long builderHandle, long streamAddress);
    private static native void closeNative(long builderHandle);
    private static native void convertToStream(VectorSchemaRoot root, ArrowArrayStream stream);
}