package lance
import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.vector.FieldVector
import org.apache.arrow.vector.ipc.ArrowReader

case class Dataset(vec: FieldVector) {
  def saveToPath(path: String) = {
    JNI.saveToLance(path, vec, vec.getAllocator)
  }
}

object Dataset {
  def writeDataset(path: String, allocator: BufferAllocator, reader: ArrowReader): Unit = {
    JNI.saveStreamToLance(path, reader, allocator)
  }
  def fromArrow(vec: FieldVector): Dataset                                        = {
    Dataset(vec)
  }
}
