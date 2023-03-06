import lance.JNI
import org.apache.arrow.dataset.file.{FileFormat, FileSystemDatasetFactory}
import org.apache.arrow.dataset.jni.{NativeDataset, NativeMemoryPool}
import org.apache.arrow.dataset.scanner.ScanOptions
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.IntVector
import org.apache.arrow.dataset.source.DatasetFactory
import org.apache.arrow.vector.ipc.ArrowStreamReader

import java.io.FileInputStream
import java.nio.file.{Files, Paths}
import scala.util.Using

object Main {
  def main(args: Array[String]): Unit = {
    println("run run run")
    readDataExample()
  }

  private def readDataExample(): Unit = {
    val readPath               = Paths.get("src/main/resources/prime_numbers.csv").toAbsolutePath.toUri.toString
    val writePath              = ""
    println(s"read path: ${readPath}")
    val options                = new ScanOptions( /*batchSize*/ 32768)
    // TODO why would this Using supress exceptions?
    //    Using(new RootAllocator) { allocator =>
    val allocator              = new RootAllocator()
    println("allocator initialized")
    val factory                = new FileSystemDatasetFactory(allocator, NativeMemoryPool.getDefault, FileFormat.CSV, readPath)
    println("factory initialized")
    val dataset: NativeDataset = factory.finish()
    println("dataset initialized")
    val scanner                = dataset.newScan(options)
    println(s"schema: ${scanner.schema()}")
    JNI.saveStreamToLance(writePath, scanner.scanBatches(), allocator)
  }

  def generateDataExample(): Unit = {
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
}
