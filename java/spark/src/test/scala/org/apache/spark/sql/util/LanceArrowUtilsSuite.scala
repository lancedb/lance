package org.apache.spark.sql.util

import com.lancedb.lance.spark.LanceConstant
import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.spark.SparkUnsupportedOperationException
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite

import java.time.ZoneId

// this suite code was fork from apache spark ArrowUtilsSuite and add test("unsigned long")
class LanceArrowUtilsSuite extends AnyFunSuite {
  def roundtrip(dt: DataType, fieldName: String = "value"): Unit = {
    dt match {
      case schema: StructType =>
        assert(LanceArrowUtils.fromArrowSchema(LanceArrowUtils.toArrowSchema(schema, null, true)) === schema)
      case _ =>
        roundtrip(new StructType().add(fieldName, dt))
    }
  }

  test("unsigned long") {
    roundtrip(BooleanType, LanceConstant.ROW_ID)
    val arrowType = LanceArrowUtils.toArrowField(LanceConstant.ROW_ID, LongType, true, "Beijing")
    assert(arrowType.getType.asInstanceOf[ArrowType.Int].getBitWidth === 64)
    assert(!arrowType.getType.asInstanceOf[ArrowType.Int].getIsSigned)
  }

  test("simple") {
    roundtrip(BooleanType)
    roundtrip(ByteType)
    roundtrip(ShortType)
    roundtrip(IntegerType)
    roundtrip(LongType)
    roundtrip(FloatType)
    roundtrip(DoubleType)
    roundtrip(StringType)
    roundtrip(BinaryType)
    roundtrip(DecimalType.SYSTEM_DEFAULT)
    roundtrip(DateType)
    roundtrip(YearMonthIntervalType())
    roundtrip(DayTimeIntervalType())
  }

  test("timestamp") {

    def roundtripWithTz(timeZoneId: String): Unit = {
      val schema = new StructType().add("value", TimestampType)
      val arrowSchema = LanceArrowUtils.toArrowSchema(schema, timeZoneId, true)
      val fieldType = arrowSchema.findField("value").getType.asInstanceOf[ArrowType.Timestamp]
      assert(fieldType.getTimezone() === timeZoneId)
      assert(LanceArrowUtils.fromArrowSchema(arrowSchema) === schema)
    }

    roundtripWithTz(ZoneId.systemDefault().getId)
    roundtripWithTz("Asia/Tokyo")
    roundtripWithTz("UTC")
  }

  test("array") {
    roundtrip(ArrayType(IntegerType, containsNull = true))
    roundtrip(ArrayType(IntegerType, containsNull = false))
    roundtrip(ArrayType(ArrayType(IntegerType, containsNull = true), containsNull = true))
    roundtrip(ArrayType(ArrayType(IntegerType, containsNull = false), containsNull = true))
    roundtrip(ArrayType(ArrayType(IntegerType, containsNull = true), containsNull = false))
    roundtrip(ArrayType(ArrayType(IntegerType, containsNull = false), containsNull = false))
  }

  test("struct") {
    roundtrip(new StructType())
    roundtrip(new StructType().add("i", IntegerType))
    roundtrip(new StructType().add("arr", ArrayType(IntegerType)))
    roundtrip(new StructType().add("i", IntegerType).add("arr", ArrayType(IntegerType)))
    roundtrip(new StructType().add(
      "struct",
      new StructType().add("i", IntegerType).add("arr", ArrayType(IntegerType))))
  }

  test("struct with duplicated field names") {

    def check(dt: DataType, expected: DataType): Unit = {
      val schema = new StructType().add("value", dt)
      intercept[SparkUnsupportedOperationException] {
        LanceArrowUtils.toArrowSchema(schema, null, true)
      }
      assert(LanceArrowUtils.fromArrowSchema(LanceArrowUtils.toArrowSchema(schema, null, false))
        === new StructType().add("value", expected))
    }

    roundtrip(new StructType().add("i", IntegerType).add("i", StringType))

    check(new StructType().add("i", IntegerType).add("i", StringType),
      new StructType().add("i_0", IntegerType).add("i_1", StringType))
    check(ArrayType(new StructType().add("i", IntegerType).add("i", StringType)),
      ArrayType(new StructType().add("i_0", IntegerType).add("i_1", StringType)))
    check(MapType(StringType, new StructType().add("i", IntegerType).add("i", StringType)),
      MapType(StringType, new StructType().add("i_0", IntegerType).add("i_1", StringType)))
  }

}
