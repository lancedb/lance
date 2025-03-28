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
package org.apache.spark.sql.vectorized;

import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.ValueVector;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.util.LanceArrowUtils;
import org.apache.spark.unsafe.types.UTF8String;

public class LanceArrowColumnVector extends ColumnVector {
  private UInt8Accessor uInt8Accessor;
  private ArrowColumnVector arrowColumnVector;

  public LanceArrowColumnVector(ValueVector vector) {
    super(LanceArrowUtils.fromArrowField(vector.getField()));
    if (vector instanceof UInt8Vector) {
      uInt8Accessor = new UInt8Accessor((UInt8Vector) vector);
    } else {
      arrowColumnVector = new ArrowColumnVector(vector);
    }
  }

  @Override
  public void close() {
    if (uInt8Accessor != null) {
      uInt8Accessor.close();
    }
    if (arrowColumnVector != null) {
      arrowColumnVector.close();
    }
  }

  @Override
  public boolean hasNull() {
    if (uInt8Accessor != null) {
      return uInt8Accessor.getNullCount() > 0;
    }
    if (arrowColumnVector != null) {
      return arrowColumnVector.hasNull();
    }
    return false;
  }

  @Override
  public int numNulls() {
    if (uInt8Accessor != null) {
      return uInt8Accessor.getNullCount();
    }
    if (arrowColumnVector != null) {
      return arrowColumnVector.numNulls();
    }
    return 0;
  }

  @Override
  public boolean isNullAt(int rowId) {
    if (uInt8Accessor != null) {
      return uInt8Accessor.isNullAt(rowId);
    }
    if (arrowColumnVector != null) {
      return arrowColumnVector.isNullAt(rowId);
    }
    return false;
  }

  @Override
  public boolean getBoolean(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getBoolean(rowId);
    }
    return false;
  }

  @Override
  public byte getByte(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getByte(rowId);
    }
    return 0;
  }

  @Override
  public short getShort(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getShort(rowId);
    }
    return 0;
  }

  @Override
  public int getInt(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getInt(rowId);
    }
    return 0;
  }

  @Override
  public long getLong(int rowId) {
    if (uInt8Accessor != null) {
      return uInt8Accessor.getLong(rowId);
    }
    if (arrowColumnVector != null) {
      return arrowColumnVector.getLong(rowId);
    }
    return 0L;
  }

  @Override
  public float getFloat(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getFloat(rowId);
    }
    return 0;
  }

  @Override
  public double getDouble(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getDouble(rowId);
    }
    return 0;
  }

  @Override
  public ColumnarArray getArray(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getArray(rowId);
    }
    return null;
  }

  @Override
  public ColumnarMap getMap(int ordinal) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getMap(ordinal);
    }
    return null;
  }

  @Override
  public Decimal getDecimal(int rowId, int precision, int scale) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getDecimal(rowId, precision, scale);
    }
    return null;
  }

  @Override
  public UTF8String getUTF8String(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getUTF8String(rowId);
    }
    return null;
  }

  @Override
  public byte[] getBinary(int rowId) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getBinary(rowId);
    }
    return new byte[0];
  }

  @Override
  public ColumnVector getChild(int ordinal) {
    if (arrowColumnVector != null) {
      return arrowColumnVector.getChild(ordinal);
    }
    return null;
  }
}
