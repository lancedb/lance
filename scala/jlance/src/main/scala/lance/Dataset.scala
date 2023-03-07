package lance
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.ipc.ArrowReader


object Dataset {
  def writeDataset(path: String, allocator: BufferAllocator, reader: ArrowReader): Unit = {
    JNI.saveStreamToLance(path, reader, allocator)
  }
}
