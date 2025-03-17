package com.lancedb.lance.file;

import java.io.IOException;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.DictionaryProvider;

import com.lancedb.lance.JniLoader;

public class LanceFileWriter implements AutoCloseable {

    static {
        JniLoader.ensureLoaded();
    }

    private long nativeFileWriterHandle;
    private BufferAllocator allocator;
    private DictionaryProvider dictionaryProvider;

    private static native LanceFileWriter openNative(String fileUri) throws IOException;

    private native void closeNative(long nativeLanceFileReaderHandle) throws IOException;

    private native void writeNative(
            long batchMemoryAddress,
            long schemaMemoryAddress) throws IOException;

    private LanceFileWriter() {
    }

    /**
     * Open a LanceFileWriter to write to a given file URI
     * 
     * @param path               the URI of the file to write to
     * @param allocator          the BufferAllocator to use for the writer
     * @param dictionaryProvider the DictionaryProvider to use for the writer
     * @return a new LanceFileWriter
     */
    public static LanceFileWriter open(
            String path,
            BufferAllocator allocator,
            DictionaryProvider dictionaryProvider)
            throws IOException {
        LanceFileWriter writer = openNative(path);
        writer.allocator = allocator;
        writer.dictionaryProvider = dictionaryProvider;
        return writer;
    }

    /**
     * Write a batch of data
     * 
     * @param batch the batch of data to write
     * @throws IOException if the batch cannot be written
     */
    public void write(VectorSchemaRoot batch) throws IOException {
        try (ArrowArray ffiArrowArray = ArrowArray.allocateNew(allocator);
                ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
            Data.exportVectorSchemaRoot(
                    allocator,
                    batch,
                    dictionaryProvider,
                    ffiArrowArray,
                    ffiArrowSchema);
            writeNative(
                    ffiArrowArray.memoryAddress(),
                    ffiArrowSchema.memoryAddress());
        }
    }

    /**
     * Close the LanceFileWriter
     * 
     * This method must be called to release resources when the writer is no longer
     * needed.
     * 
     * This method will also flush all remaining data and write the footer to the
     * file.
     * 
     * @throws Exception if the writer cannot be closed
     */
    @Override
    public void close() throws Exception {
        closeNative(nativeFileWriterHandle);
    }

}
