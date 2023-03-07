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

object Example {
  def main(args: Array[String]): Unit = {
    readCsvSaveLanceExample()
  }

  private def readCsvSaveLanceExample(): Unit = {
    val readPath               = Paths.get("src/main/resources/prime_numbers.csv").toAbsolutePath.toUri.toString
    val writePath              = "/tmp/test.lance"
    println(s"read path: ${readPath}")
    val options                = new ScanOptions( /*batchSize*/ 32768)
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
}
