/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.util

/*
 * The following code is originally from https://github.com/apache/spark/blob/master/sql/catalyst/src/test/scala/org/apache/spark/sql/util/ArrowUtilsSuite.scala
 * and is licensed under the Apache license:
 *
 * License: Apache License 2.0, Copyright 2014 and onwards The Apache Software Foundation.
 * https://github.com/apache/spark/blob/master/LICENSE
 *
 * It has been modified by the Lance developers to fit the needs of the Lance project.
 */

import com.lancedb.lance.spark.LanceConstant

import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.spark.SparkUnsupportedOperationException
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite

import java.time.ZoneId

class LanceArrowUtilsSuite extends AnyFunSuite {
  def roundtrip(dt: DataType, fieldName: String = "value"): Unit = {
    dt match {
      case schema: StructType =>
        assert(LanceArrowUtils.fromArrowSchema(
          LanceArrowUtils.toArrowSchema(schema, null, true)) === schema)
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

    check(
      new StructType().add("i", IntegerType).add("i", StringType),
      new StructType().add("i_0", IntegerType).add("i_1", StringType))
    check(
      ArrayType(new StructType().add("i", IntegerType).add("i", StringType)),
      ArrayType(new StructType().add("i_0", IntegerType).add("i_1", StringType)))
    check(
      MapType(StringType, new StructType().add("i", IntegerType).add("i", StringType)),
      MapType(StringType, new StructType().add("i_0", IntegerType).add("i_1", StringType)))
  }

}
