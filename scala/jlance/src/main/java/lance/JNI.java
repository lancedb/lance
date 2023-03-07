package lance;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.vector.ipc.ArrowReader;

public class JNI {
    static {
        System.loadLibrary("jlance");
    }

    public static void factCallback(int res) {
        System.out.println("factCallback: res = " + res);
    }

    public static void fillVector(long schemaAddress, long arrayAddress, FieldVector vec, BufferAllocator allocator) {
        try (ArrowArray arrow_array = ArrowArray.wrap(arrayAddress);
             ArrowSchema arrow_schema = ArrowSchema.wrap(schemaAddress)) {
            Data.exportVector(allocator, vec, null, arrow_array, arrow_schema);
        }
    }

    public static void fillStream(long streamAddress, ArrowReader reader,BufferAllocator allocator ) {
        Data.exportArrayStream(allocator, reader, ArrowArrayStream.wrap(streamAddress));
    }

    native public static void saveToLance(String path, FieldVector vec, BufferAllocator allocator);

    native public static void saveStreamToLance(String path, ArrowReader reader, BufferAllocator allocator);
}
