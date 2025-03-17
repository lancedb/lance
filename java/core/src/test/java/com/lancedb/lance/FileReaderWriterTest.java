package com.lancedb.lance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.lancedb.lance.file.LanceFileReader;
import com.lancedb.lance.file.LanceFileWriter;

public class FileReaderWriterTest {

    @TempDir
    private static Path tempDir;

    private VectorSchemaRoot createBatch(BufferAllocator allocator) throws IOException {
        Schema schema = new Schema(
                Arrays.asList(Field.nullable("x", new ArrowType.Int(64, true)),
                        Field.nullable("y", new ArrowType.Utf8())),
                null);
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        root.allocateNew();
        BigIntVector iVector = (BigIntVector) root.getVector("x");
        VarCharVector sVector = (VarCharVector) root.getVector("y");

        for (int i = 0; i < 100; i++) {
            iVector.setSafe(i, i);
            sVector.setSafe(i, new Text("s-" + i));
        }
        root.setRowCount(100);

        return root;
    }

    void createSimpleFile(String filePath) throws Exception {
        BufferAllocator allocator = new RootAllocator();
        try (LanceFileWriter writer = LanceFileWriter.open(filePath, allocator, null)) {
            try (VectorSchemaRoot batch = createBatch(allocator)) {
                writer.write(batch);
            }
        }
    }

    @Test
    void testBasicRead() throws Exception {
        BufferAllocator allocator = new RootAllocator();
        String filePath = tempDir.resolve("basic_read.lance").toString();
        createSimpleFile(filePath);
        LanceFileReader reader = LanceFileReader.open(filePath, allocator);

        Schema expectedSchema = new Schema(
                Arrays.asList(Field.nullable("x", new ArrowType.Int(64, true)),
                        Field.nullable("y", new ArrowType.Utf8())),
                null);

        assertEquals(100, reader.numRows());
        assertEquals(expectedSchema, reader.schema());

        try (ArrowReader batches = reader.readAll(100)) {
            assertTrue(batches.loadNextBatch());
            VectorSchemaRoot batch = batches.getVectorSchemaRoot();
            assertEquals(100, batch.getRowCount());
            assertEquals(2, batch.getSchema().getFields().size());
            assertFalse(batches.loadNextBatch());
        }

        try (ArrowReader batches = reader.readAll(15)) {
            for (int i = 0; i < 100; i += 15) {
                int expected = Math.min(15, 100 - i);
                assertTrue(batches.loadNextBatch());
                VectorSchemaRoot batch = batches.getVectorSchemaRoot();
                assertEquals(expected, batch.getRowCount());
                assertEquals(2, batch.getSchema().getFields().size());
            }
            assertFalse(batches.loadNextBatch());
        }

        reader.close();
        try {
            reader.numRows();
            fail("Expected LanceException to be thrown");
        } catch (IOException e) {
            assertEquals("FileReader has already been closed", e.getMessage());
        }

        // Ok to call schema after close
        assertEquals(expectedSchema, reader.schema());

        // close should be idempotent
        reader.close();
    }

    @Test
    void testBasicWrite() throws Exception {
        String filePath = tempDir.resolve("basic_write.lance").toString();
        createSimpleFile(filePath);
    }

    @Test
    void testWriteNoData() throws Exception {
        String filePath = tempDir.resolve("no_data.lance").toString();
        BufferAllocator allocator = new RootAllocator();

        LanceFileWriter writer = LanceFileWriter.open(filePath, allocator, null);

        try {
            writer.close();
            fail("Expected IllegalArgumentException to be thrown");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("no data provided"));
        }
    }

    @Test
    void testInvalidPath() {
        BufferAllocator allocator = new RootAllocator();
        try {
            LanceFileReader.open("/tmp/does_not_exist.lance", allocator);
            fail("Expected LanceException to be thrown");
        } catch (IOException e) {
            assertTrue(e.getMessage().contains("Not found: tmp/does_not_exist.lance"));
        }
        try {
            LanceFileReader.open("", allocator);
            fail("Expected LanceException to be thrown");
        } catch (RuntimeException e) {
            // expected, would be nice if it was an IOException, but it's not because
            // lance throws a wrapped error :(
        } catch (IOException e) {
            fail("Expected RuntimeException to be thrown");
        }
    }

}
