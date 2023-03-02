import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.IntVector

import scala.util.Using
object Main extends App {
  println("Hello, World!")
  Using(new RootAllocator()) { rootAllocator =>
    Using(new IntVector("fixed-size-primitive-layout", rootAllocator)) { intVector =>
      intVector.allocateNew(3)
      intVector.set(0, 1)
      intVector.setNull(1)
      intVector.set(2, 2)
      intVector.setValueCount(3)
      println("Vector created in memory: " + intVector)
      val path = "/tmp/example.lance"
      lance.Dataset.fromArrow(intVector).saveToPath(path)
    }
  }
}
