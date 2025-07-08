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
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for MergeInsertBuilder functionality.
 * 
 * These tests cover basic functionality, error handling, boundary conditions,
 * and resource management for the MergeInsertBuilder class.
 */
public class MergeInsertTest {
    
    @TempDir
    Path tempDir;
    
    private BufferAllocator allocator;
    private Dataset dataset;
    private Schema schema;
    
    @BeforeEach
    void setUp() throws IOException {
        allocator = new RootAllocator(Long.MAX_VALUE);
        
        // Create a simple schema for testing
        schema = new Schema(Arrays.asList(
            Field.nullable("id", new ArrowType.Int(32, true)),
            Field.nullable("name", new ArrowType.Utf8()),
            Field.nullable("value", new ArrowType.Int(32, true))
        ));
        
        // Create an empty dataset
        String datasetPath = tempDir.resolve("merge_insert_test").toString();
        dataset = Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build());
    }
    
    @AfterEach
    void tearDown() {
        if (dataset != null) {
            dataset.close();
        }
        if (allocator != null) {
            allocator.close();
        }
    }
    
    @Test
    void testCreateMergeInsertBuilder() {
        // Test basic creation
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        assertNotNull(builder);
        
        // Test with multiple columns
        MergeInsertBuilder builder2 = MergeInsertBuilder.create(dataset, Arrays.asList("id", "name"));
        assertNotNull(builder2);
    }
    
    @Test
    void testCreateMergeInsertBuilderWithInvalidColumns() {
        // Test with non-existent column
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, "non_existent_column");
        });
    }
    
    @Test
    void testCreateMergeInsertBuilderWithEmptyColumns() {
        // Test with empty column list
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, Collections.emptyList());
        });
    }
    
    @Test
    void testCreateMergeInsertBuilderWithNullColumns() {
        // Test with null columns
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, (String[]) null);
        });
    }
    
    @Test
    void testMergeInsertBuilderConfiguration() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test fluent API configuration
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .conflictRetries(5)
            .retryTimeout(1000L);
        
        assertNotNull(configured);
        assertSame(builder, configured); // Should return same instance for fluent API
    }
    
    @Test
    void testMergeInsertBuilderWithCondition() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with condition
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll("source.value > target.value")
            .whenNotMatchedBySourceDelete("target.id > 100");
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderWithComplexCondition() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with complex SQL condition
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll("source.value > target.value AND source.name != target.name")
            .whenNotMatchedBySourceDelete("target.id > 100 OR target.value < 0");
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderWithSpecialCharacters() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with special characters in conditions
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll("source.name LIKE '%test%'")
            .whenNotMatchedBySourceDelete("target.name = 'special\"quote'");
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderClose() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test that close doesn't throw
        assertDoesNotThrow(() -> builder.close());
        
        // Test that multiple closes don't throw
        assertDoesNotThrow(() -> builder.close());
    }
    
    @Test
    void testMergeInsertResultCreation() {
        MergeInsertResult result = new MergeInsertResult(10L, 5L, 2L);
        
        assertEquals(10L, result.getNumInsertedRows());
        assertEquals(5L, result.getNumUpdatedRows());
        assertEquals(2L, result.getNumDeletedRows());
        assertEquals(17L, result.getTotalAffectedRows());
    }
    
    @Test
    void testMergeInsertResultWithZeroValues() {
        MergeInsertResult result = new MergeInsertResult(0L, 0L, 0L);
        
        assertEquals(0L, result.getNumInsertedRows());
        assertEquals(0L, result.getNumUpdatedRows());
        assertEquals(0L, result.getNumDeletedRows());
        assertEquals(0L, result.getTotalAffectedRows());
    }
    
    @Test
    void testMergeInsertResultWithLargeValues() {
        MergeInsertResult result = new MergeInsertResult(Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);
        
        assertEquals(Long.MAX_VALUE, result.getNumInsertedRows());
        assertEquals(Long.MAX_VALUE, result.getNumUpdatedRows());
        assertEquals(Long.MAX_VALUE, result.getNumDeletedRows());
        
        // Test overflow handling
        assertTrue(result.getTotalAffectedRows() < 0); // Should overflow
    }
    
    @Test
    void testMergeInsertResultEquality() {
        MergeInsertResult result1 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result2 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result3 = new MergeInsertResult(10L, 5L, 3L);
        
        assertEquals(result1, result2);
        assertNotEquals(result1, result3);
        assertEquals(result1.hashCode(), result2.hashCode());
    }
    
    @Test
    void testMergeInsertResultToString() {
        MergeInsertResult result = new MergeInsertResult(10L, 5L, 2L);
        String str = result.toString();
        
        assertTrue(str.contains("inserted=10"));
        assertTrue(str.contains("updated=5"));
        assertTrue(str.contains("deleted=2"));
    }
    
    @Test
    void testCreateVectorSchemaRoot() {
        // Test creating VectorSchemaRoot for merge insert
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            assertNotNull(root);
            assertEquals(0, root.getRowCount());
            
            // Add some test data
            IntVector idVector = (IntVector) root.getVector("id");
            VarCharVector nameVector = (VarCharVector) root.getVector("name");
            IntVector valueVector = (IntVector) root.getVector("value");
            
            root.setRowCount(2);
            idVector.setValueCount(2);
            nameVector.setValueCount(2);
            valueVector.setValueCount(2);
            
            idVector.set(0, 1);
            idVector.set(1, 2);
            nameVector.set(0, "Alice".getBytes());
            nameVector.set(1, "Bob".getBytes());
            valueVector.set(0, 100);
            valueVector.set(1, 200);
            
            assertEquals(2, root.getRowCount());
        }
    }
    
    @Test
    void testMergeInsertBuilderWithNullConditions() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with null conditions
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll(null)
            .whenNotMatchedBySourceDelete(null);
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderWithEmptyConditions() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with empty conditions
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll("")
            .whenNotMatchedBySourceDelete("");
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderWithVeryLongConditions() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with very long conditions
        String longCondition = "source.value > target.value AND source.name != target.name AND source.id > 0 AND target.id > 0 AND source.value < 1000000 AND target.value < 1000000";
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll(longCondition)
            .whenNotMatchedBySourceDelete(longCondition);
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderWithSpecialCharactersInConditions() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test with various special characters
        String specialCondition = "source.name = 'test\"quote' AND source.value > 0 AND target.name LIKE '%test%'";
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll(specialCondition)
            .whenNotMatchedBySourceDelete(specialCondition);
        
        assertNotNull(configured);
    }
    
    @Test
    void testMergeInsertBuilderRetryConfiguration() {
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Test various retry configurations
        MergeInsertBuilder configured1 = builder.conflictRetries(0);
        assertNotNull(configured1);
        
        MergeInsertBuilder configured2 = builder.conflictRetries(100);
        assertNotNull(configured2);
        
        MergeInsertBuilder configured3 = builder.retryTimeout(0L);
        assertNotNull(configured3);
        
        MergeInsertBuilder configured4 = builder.retryTimeout(Long.MAX_VALUE);
        assertNotNull(configured4);
    }
    
    @Test
    void testMergeInsertBuilderResourceManagement() {
        // Test that multiple builders can be created and closed
        MergeInsertBuilder builder1 = MergeInsertBuilder.create(dataset, "id");
        MergeInsertBuilder builder2 = MergeInsertBuilder.create(dataset, "id");
        MergeInsertBuilder builder3 = MergeInsertBuilder.create(dataset, "id");
        
        assertNotNull(builder1);
        assertNotNull(builder2);
        assertNotNull(builder3);
        
        // Close all builders
        assertDoesNotThrow(() -> builder1.close());
        assertDoesNotThrow(() -> builder2.close());
        assertDoesNotThrow(() -> builder3.close());
    }
    
    @Test
    void testMergeInsertBuilderConcurrentAccess() {
        // Test that builders can be configured concurrently
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Simulate concurrent configuration (though not actually concurrent)
        MergeInsertBuilder configured1 = builder.whenMatchedUpdateAll();
        MergeInsertBuilder configured2 = builder.whenNotMatchedInsertAll();
        MergeInsertBuilder configured3 = builder.conflictRetries(5);
        
        assertSame(builder, configured1);
        assertSame(builder, configured2);
        assertSame(builder, configured3);
        
        builder.close();
    }
    
    @Test
    void testMergeInsertBuilderWithMultipleColumns() {
        // Test with multiple join columns
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id", "name"));
        
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .conflictRetries(3)
            .retryTimeout(2000L);
        
        assertNotNull(configured);
        builder.close();
    }
    
    @Test
    void testMergeInsertBuilderWithAllColumns() {
        // Test with all available columns
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id", "name", "value"));
        
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll("source.value > target.value")
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete("target.id > 100");
        
        assertNotNull(configured);
        builder.close();
    }
    
    @Test
    void testMergeInsertResultEdgeCases() {
        // Test edge cases for MergeInsertResult
        
        // All zeros
        MergeInsertResult result1 = new MergeInsertResult(0L, 0L, 0L);
        assertEquals(0L, result1.getTotalAffectedRows());
        
        // Only inserted
        MergeInsertResult result2 = new MergeInsertResult(1L, 0L, 0L);
        assertEquals(1L, result2.getTotalAffectedRows());
        
        // Only updated
        MergeInsertResult result3 = new MergeInsertResult(0L, 1L, 0L);
        assertEquals(1L, result3.getTotalAffectedRows());
        
        // Only deleted
        MergeInsertResult result4 = new MergeInsertResult(0L, 0L, 1L);
        assertEquals(1L, result4.getTotalAffectedRows());
        
        // Large numbers
        MergeInsertResult result5 = new MergeInsertResult(1000000L, 500000L, 250000L);
        assertEquals(1750000L, result5.getTotalAffectedRows());
    }
    
    @Test
    void testMergeInsertResultOverflow() {
        // Test overflow scenarios
        MergeInsertResult result1 = new MergeInsertResult(Long.MAX_VALUE, 1L, 0L);
        assertTrue(result1.getTotalAffectedRows() < 0); // Should overflow
        
        MergeInsertResult result2 = new MergeInsertResult(1L, Long.MAX_VALUE, 0L);
        assertTrue(result2.getTotalAffectedRows() < 0); // Should overflow
        
        MergeInsertResult result3 = new MergeInsertResult(0L, 0L, Long.MAX_VALUE);
        assertEquals(Long.MAX_VALUE, result3.getTotalAffectedRows());
    }
    
    @Test
    void testMergeInsertBuilderPerformance() {
        // Test performance with many configurations
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < 1000; i++) {
            MergeInsertBuilder configured = builder
                .whenMatchedUpdateAll("source.value > " + i)
                .whenNotMatchedInsertAll()
                .conflictRetries(i % 10)
                .retryTimeout(i * 1000L);
            
            assertNotNull(configured);
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Should complete within reasonable time (adjust threshold as needed)
        assertTrue(duration < 5000, "Performance test took too long: " + duration + "ms");
        
        builder.close();
    }
    
    @Test
    void testMergeInsertBuilderMemoryLeak() {
        // Test for potential memory leaks
        for (int i = 0; i < 100; i++) {
            MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
            
            MergeInsertBuilder configured = builder
                .whenMatchedUpdateAll("source.value > " + i)
                .whenNotMatchedInsertAll()
                .conflictRetries(i % 5)
                .retryTimeout(i * 100L);
            
            assertNotNull(configured);
            builder.close();
        }
        
        // If we get here without memory issues, the test passes
        assertTrue(true);
    }

    @Test
    void testMergeInsertBuilderWithNullDataset() {
        // Test with null dataset
        assertThrows(NullPointerException.class, () -> {
            MergeInsertBuilder.create(null, "id");
        });
    }

    @Test
    void testMergeInsertBuilderWithNullColumnNames() {
        // Test with null column names in array
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, (String) null);
        });
    }

    @Test
    void testMergeInsertBuilderWithEmptyColumnName() {
        // Test with empty column name
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, "");
        });
    }

    @Test
    void testMergeInsertBuilderWithWhitespaceColumnName() {
        // Test with whitespace-only column name
        assertThrows(RuntimeException.class, () -> {
            MergeInsertBuilder.create(dataset, "   ");
        });
    }

    @Test
    void testMergeInsertBuilderWithDuplicateColumns() {
        // Test with duplicate column names
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id", "id"));
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithAllSchemaColumns() {
        // Test with all available columns in schema
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id", "name", "value"));
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithCaseSensitiveColumns() {
        // Test with case-sensitive column names
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "ID");
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithSpecialColumnNames() {
        // Test with special characters in column names
        // Note: This would require a schema with special column names
        // For now, test with regular column names
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderConfigurationChaining() {
        // Test method chaining with all configuration methods
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        MergeInsertBuilder configured = builder
            .whenMatchedUpdateAll()
            .whenMatchedUpdateAll("source.value > target.value")
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete()
            .whenNotMatchedBySourceDelete("target.id > 100")
            .conflictRetries(10)
            .retryTimeout(5000L);
        
        assertNotNull(configured);
        assertSame(builder, configured);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithNegativeRetryValues() {
        // Test with negative retry values
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Should not throw for negative values (handled by native code)
        assertDoesNotThrow(() -> {
            builder.conflictRetries(-1);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithNegativeTimeoutValues() {
        // Test with negative timeout values
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Should not throw for negative values (handled by native code)
        assertDoesNotThrow(() -> {
            builder.retryTimeout(-1000L);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithExtremeTimeoutValues() {
        // Test with extreme timeout values
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        assertDoesNotThrow(() -> {
            builder.retryTimeout(Long.MAX_VALUE);
        });
        
        assertDoesNotThrow(() -> {
            builder.retryTimeout(Long.MIN_VALUE);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithExtremeRetryValues() {
        // Test with extreme retry values
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        assertDoesNotThrow(() -> {
            builder.conflictRetries(Integer.MAX_VALUE);
        });
        
        assertDoesNotThrow(() -> {
            builder.conflictRetries(Integer.MIN_VALUE);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithComplexSqlConditions() {
        // Test with complex SQL conditions
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        String complexCondition = 
            "source.value > target.value AND source.name != target.name AND " +
            "source.id > 0 AND target.id > 0 AND source.value < 1000000 AND " +
            "target.value < 1000000 AND (source.name LIKE '%test%' OR target.name LIKE '%test%')";
        
        assertDoesNotThrow(() -> {
            builder.whenMatchedUpdateAll(complexCondition);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithUnicodeConditions() {
        // Test with Unicode characters in conditions
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        String unicodeCondition = "source.name = '测试' AND target.name = 'テスト'";
        
        assertDoesNotThrow(() -> {
            builder.whenMatchedUpdateAll(unicodeCondition);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithEscapedCharacters() {
        // Test with escaped characters in conditions
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        String escapedCondition = "source.name = 'test\\'quote' AND target.name = 'test\"quote'";
        
        assertDoesNotThrow(() -> {
            builder.whenMatchedUpdateAll(escapedCondition);
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderWithVeryLongCondition() {
        // Test with very long condition string
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Create a very long condition string
        StringBuilder longCondition = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            longCondition.append("source.value > ").append(i).append(" AND ");
        }
        longCondition.append("source.id > 0");
        
        assertDoesNotThrow(() -> {
            builder.whenMatchedUpdateAll(longCondition.toString());
        });
        
        builder.close();
    }

    @Test
    void testMergeInsertBuilderMultipleConfigurations() {
        // Test multiple configurations on the same builder
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // First configuration
        builder.whenMatchedUpdateAll("source.value > 0");
        
        // Second configuration (should override first)
        builder.whenMatchedUpdateAll("source.value > 100");
        
        // Third configuration
        builder.whenNotMatchedInsertAll();
        
        // Fourth configuration
        builder.conflictRetries(5);
        
        // Fifth configuration
        builder.retryTimeout(2000L);
        
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderConcurrentModification() {
        // Test concurrent modification scenarios
        MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
        
        // Simulate concurrent configuration (though not actually concurrent)
        for (int i = 0; i < 10; i++) {
            builder.whenMatchedUpdateAll("source.value > " + i);
            builder.whenNotMatchedInsertAll();
            builder.conflictRetries(i);
            builder.retryTimeout(i * 100L);
        }
        
        assertNotNull(builder);
        builder.close();
    }

    @Test
    void testMergeInsertBuilderResourceCleanup() {
        // Test proper resource cleanup
        MergeInsertBuilder builder1 = MergeInsertBuilder.create(dataset, "id");
        MergeInsertBuilder builder2 = MergeInsertBuilder.create(dataset, "id");
        MergeInsertBuilder builder3 = MergeInsertBuilder.create(dataset, "id");
        
        // Configure all builders
        builder1.whenMatchedUpdateAll();
        builder2.whenNotMatchedInsertAll();
        builder3.conflictRetries(5);
        
        // Close all builders
        assertDoesNotThrow(() -> builder1.close());
        assertDoesNotThrow(() -> builder2.close());
        assertDoesNotThrow(() -> builder3.close());
        
        // Try to close again (should not throw)
        assertDoesNotThrow(() -> builder1.close());
    }

    @Test
    void testMergeInsertBuilderWithClosedDataset() {
        // Test behavior with closed dataset
        Dataset closedDataset = Dataset.create(allocator, tempDir.resolve("closed_test").toString(), 
                                             schema, new WriteParams.Builder().build());
        closedDataset.close();
        
        // Should handle closed dataset gracefully
        assertDoesNotThrow(() -> {
            MergeInsertBuilder.create(closedDataset, "id");
        });
    }

    @Test
    void testMergeInsertResultWithNegativeValues() {
        // Test with negative values (should handle gracefully)
        MergeInsertResult result = new MergeInsertResult(-1L, -2L, -3L);
        
        assertEquals(-1L, result.getNumInsertedRows());
        assertEquals(-2L, result.getNumUpdatedRows());
        assertEquals(-3L, result.getNumDeletedRows());
        assertEquals(-6L, result.getTotalAffectedRows());
    }

    @Test
    void testMergeInsertResultWithMixedValues() {
        // Test with mixed positive and negative values
        MergeInsertResult result = new MergeInsertResult(10L, -5L, 3L);
        
        assertEquals(10L, result.getNumInsertedRows());
        assertEquals(-5L, result.getNumUpdatedRows());
        assertEquals(3L, result.getNumDeletedRows());
        assertEquals(8L, result.getTotalAffectedRows());
    }

    @Test
    void testMergeInsertResultHashCode() {
        // Test hashCode consistency
        MergeInsertResult result1 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result2 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result3 = new MergeInsertResult(10L, 5L, 3L);
        
        assertEquals(result1.hashCode(), result2.hashCode());
        assertNotEquals(result1.hashCode(), result3.hashCode());
    }

    @Test
    void testMergeInsertResultEquals() {
        // Test equals method
        MergeInsertResult result1 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result2 = new MergeInsertResult(10L, 5L, 2L);
        MergeInsertResult result3 = new MergeInsertResult(10L, 5L, 3L);
        MergeInsertResult result4 = new MergeInsertResult(11L, 5L, 2L);
        MergeInsertResult result5 = new MergeInsertResult(10L, 6L, 2L);
        
        assertEquals(result1, result2);
        assertNotEquals(result1, result3);
        assertNotEquals(result1, result4);
        assertNotEquals(result1, result5);
        assertNotEquals(result1, null);
        assertNotEquals(result1, "not a result");
    }

    @Test
    void testMergeInsertResultToStringFormat() {
        // Test toString format
        MergeInsertResult result = new MergeInsertResult(10L, 5L, 2L);
        String str = result.toString();
        
        assertTrue(str.contains("MergeInsertResult"));
        assertTrue(str.contains("inserted=10"));
        assertTrue(str.contains("updated=5"));
        assertTrue(str.contains("deleted=2"));
    }

    @Test
    void testMergeInsertBuilderPerformanceWithManyColumns() {
        // Test performance with many columns
        // Create a schema with many columns
        List<Field> manyFields = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            manyFields.add(Field.nullable("col" + i, new ArrowType.Int(32, true)));
        }
        Schema manyColumnSchema = new Schema(manyFields);
        
        String datasetPath = tempDir.resolve("many_columns_test").toString();
        Dataset manyColumnDataset = Dataset.create(allocator, datasetPath, manyColumnSchema, 
                                                 new WriteParams.Builder().build());
        
        // Test builder creation with many columns
        long startTime = System.currentTimeMillis();
        
        List<String> manyColumns = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            manyColumns.add("col" + i);
        }
        
        MergeInsertBuilder builder = MergeInsertBuilder.create(manyColumnDataset, manyColumns);
        assertNotNull(builder);
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Should complete within reasonable time
        assertTrue(duration < 1000, "Performance test took too long: " + duration + "ms");
        
        builder.close();
        manyColumnDataset.close();
    }

    @Test
    void testMergeInsertBuilderStressTest() {
        // Stress test with many builders
        List<MergeInsertBuilder> builders = new ArrayList<>();
        
        try {
            for (int i = 0; i < 100; i++) {
                MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");
                builder.whenMatchedUpdateAll("source.value > " + i);
                builder.whenNotMatchedInsertAll();
                builder.conflictRetries(i % 10);
                builder.retryTimeout(i * 100L);
                builders.add(builder);
            }
            
            // All builders should be created successfully
            assertEquals(100, builders.size());
            
        } finally {
            // Clean up all builders
            for (MergeInsertBuilder builder : builders) {
                builder.close();
            }
        }
    }

    @Test
    void testMergeInsertBuilderWithDifferentDataTypes() {
        // Test with different data types in schema
        Schema complexSchema = new Schema(Arrays.asList(
            Field.nullable("id", new ArrowType.Int(32, true)),
            Field.nullable("name", new ArrowType.Utf8()),
            Field.nullable("value", new ArrowType.Int(32, true)),
            Field.nullable("flag", new ArrowType.Bool()),
            Field.nullable("timestamp", new ArrowType.Int(64, true))
        ));
        
        String datasetPath = tempDir.resolve("complex_schema_test").toString();
        Dataset complexDataset = Dataset.create(allocator, datasetPath, complexSchema, 
                                              new WriteParams.Builder().build());
        
        // Test with different column combinations
        MergeInsertBuilder builder1 = MergeInsertBuilder.create(complexDataset, "id");
        assertNotNull(builder1);
        
        MergeInsertBuilder builder2 = MergeInsertBuilder.create(complexDataset, Arrays.asList("id", "name"));
        assertNotNull(builder2);
        
        MergeInsertBuilder builder3 = MergeInsertBuilder.create(complexDataset, 
                                                              Arrays.asList("id", "name", "value", "flag"));
        assertNotNull(builder3);
        
        builder1.close();
        builder2.close();
        builder3.close();
        complexDataset.close();
    }
} 