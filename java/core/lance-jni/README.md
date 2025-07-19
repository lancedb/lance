## Java JNI Bindings

Lance provides comprehensive Java bindings through JNI (Java Native Interface), enabling high-performance data operations directly from Java applications.

### MergeInsertBuilder

The `MergeInsertBuilder` class provides a fluent API for building merge insert operations, which allow you to merge new data with existing data in a Lance dataset. This is similar to SQL's MERGE statement.

#### Basic Usage

```java
import com.lancedb.lance.*;

// Create a merge insert builder
MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id");

// Configure the operation
MergeInsertBuilder configured = builder
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .conflictRetries(5)
    .retryTimeout(1000L);

// Execute the merge insert
try (VectorSchemaRoot newData = createNewData()) {
    MergeInsertResult result = configured.execute(newData);
    
    System.out.println("Inserted: " + result.getNumInsertedRows());
    System.out.println("Updated: " + result.getNumUpdatedRows());
    System.out.println("Deleted: " + result.getNumDeletedRows());
}
```

#### API Reference

##### Constructor Methods

- `MergeInsertBuilder.create(Dataset dataset, String... onColumns)`
- `MergeInsertBuilder.create(Dataset dataset, List<String> onColumns)`

##### Configuration Methods

- `whenMatchedUpdateAll()` - Update all columns when there is a match
- `whenMatchedUpdateAll(String condition)` - Update all columns when there is a match and the condition is true
- `whenNotMatchedInsertAll()` - Insert all columns when there is no match
- `whenNotMatchedBySourceDelete()` - Delete rows when there is no match in the source
- `whenNotMatchedBySourceDelete(String expr)` - Delete rows when there is no match in the source and the expression is true
- `conflictRetries(int maxRetries)` - Set the number of conflict retries
- `retryTimeout(long timeoutMillis)` - Set the retry timeout in milliseconds

##### Execution Methods

- `execute(VectorSchemaRoot newData)` - Execute the merge insert operation
- `close()` - Clean up native resources

#### MergeInsertResult

The `MergeInsertResult` class contains statistics about the merge insert operation:

- `getNumInsertedRows()` - Number of rows inserted
- `getNumUpdatedRows()` - Number of rows updated  
- `getNumDeletedRows()` - Number of rows deleted
- `getTotalAffectedRows()` - Total number of affected rows

#### Error Handling

The JNI layer provides comprehensive error handling:

- Invalid column names throw `RuntimeException`
- Invalid SQL conditions are handled gracefully
- Memory management is automatic with proper cleanup
- Null conditions use default behavior

#### Testing

Comprehensive unit tests are provided in:

- `java/core/src/test/java/com/lancedb/lance/MergeInsertTest.java` - Unit tests for basic functionality, error handling, boundary conditions, and resource management
- `java/core/src/test/java/com/lancedb/lance/MergeInsertIntegrationTest.java` - Integration tests for complete workflows
- `java/core/lance-jni/src/tests.rs` - Rust-side JNI function tests

The tests cover:

**Basic Functionality:**
- Builder creation with various column configurations
- Fluent API configuration chaining
- Resource cleanup and memory management

**Error Handling:**
- Invalid column names and datasets
- Null and empty parameters
- Extreme values and edge cases
- SQL condition parsing errors

**Boundary Conditions:**
- Empty column lists
- Very long condition strings
- Unicode and special characters
- Negative and extreme numeric values

**Performance Testing:**
- Many columns and complex schemas
- Stress testing with multiple builders
- Memory leak detection
- Concurrent access patterns

**Resource Management:**
- Proper cleanup of native resources
- Multiple close operations
- Closed dataset handling
- Memory allocation patterns

#### Performance Considerations

- **Memory Management**: The JNI layer automatically manages memory for native objects
- **Resource Cleanup**: Always call `close()` on builders to prevent memory leaks
- **Concurrent Access**: Builders are not thread-safe; use separate instances per thread
- **Large Datasets**: For very large datasets, consider batching operations
- **Condition Complexity**: Complex SQL conditions may impact performance

#### Best Practices

1. **Always use try-with-resources**:
   ```java
   try (MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id")) {
       // Use builder
   }
   ```

2. **Validate inputs early**:
   ```java
   if (dataset == null || columns == null || columns.isEmpty()) {
       throw new IllegalArgumentException("Invalid parameters");
   }
   ```

3. **Handle errors gracefully**:
   ```java
   try {
       MergeInsertResult result = builder.execute(newData);
       // Process result
   } catch (RuntimeException e) {
       // Handle JNI errors
   }
   ```

4. **Use appropriate retry settings**:
   ```java
   builder.conflictRetries(10)
          .retryTimeout(5000L);
   ```

5. **Monitor performance**:
   ```java
   long startTime = System.currentTimeMillis();
   MergeInsertResult result = builder.execute(newData);
   long duration = System.currentTimeMillis() - startTime;
   ```

#### Advanced Usage Examples

**Conditional Updates:**
```java
MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id")
    .whenMatchedUpdateAll("source.value > target.value")
    .whenNotMatchedInsertAll();
```

**Complex Conditions:**
```java
String condition = "source.value > target.value AND source.name != target.name";
MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id")
    .whenMatchedUpdateAll(condition)
    .whenNotMatchedBySourceDelete("target.id > 100");
```

**Multiple Join Columns:**
```java
MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, Arrays.asList("id", "name"))
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll();
```

**High-Performance Configuration:**
```java
MergeInsertBuilder builder = MergeInsertBuilder.create(dataset, "id")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .conflictRetries(100)
    .retryTimeout(30000L);
```

#### Troubleshooting

**Common Issues:**

1. **Memory Leaks**: Ensure all builders are properly closed
2. **Performance Issues**: Check retry settings and condition complexity
3. **JNI Errors**: Verify column names and dataset validity
4. **Concurrent Access**: Use separate builders per thread

**Debug Tips:**

- Enable JNI logging: `-Djava.library.path=/path/to/lance-jni`
- Monitor memory usage during large operations
- Use smaller batch sizes for very large datasets
- Profile SQL conditions for performance bottlenecks
