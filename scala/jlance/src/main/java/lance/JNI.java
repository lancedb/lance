package lance;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;

public class JNI {
    public static void fillVector(long schemaAddress, long arrayAddress, FieldVector vec, BufferAllocator allocator) {
        try (ArrowArray arrow_array = ArrowArray.wrap(arrayAddress);
             ArrowSchema arrow_schema = ArrowSchema.wrap(schemaAddress)) {
            Data.exportVector(allocator, vec, null, arrow_array, arrow_schema);
        }
    }

    native public static void saveToLance(String path, FieldVector vec, BufferAllocator allocator);
}
