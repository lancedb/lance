package lance
import org.apache.arrow.vector.FieldVector

case class Dataset(vec: FieldVector) {
  def saveToPath(path: String) = {
    JNI.saveToLance(path, vec, vec.getAllocator)
  }
}

object Dataset {
  def fromArrow(vec: FieldVector): Dataset = {
    Dataset(vec)
  }
}
