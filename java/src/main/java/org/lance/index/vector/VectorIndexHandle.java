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
package org.lance.index.vector;

import com.google.common.base.Preconditions;

import java.util.Optional;

/**
 * Snapshot handle for a committed vector index, returned by {@code
 * Dataset.openVectorIndexHandle(String)}.
 *
 * <p>The handle is a fully-materialized snapshot at open time and is decoupled from the source
 * {@code Dataset} thereafter — subsequent dataset commits or drops do not affect this handle. The
 * handle owns native resources and must be closed; use try-with-resources or call {@link #close()}.
 */
public final class VectorIndexHandle implements AutoCloseable {
  private long nativeHandle; // 0 == closed
  private final String name;
  private final String column;
  private final int numSegments;

  // Invoked from JNI via reflection.
  private VectorIndexHandle(long nativeHandle, String name, String column, int numSegments) {
    Preconditions.checkArgument(nativeHandle != 0, "nativeHandle must be non-zero");
    this.nativeHandle = nativeHandle;
    this.name = name;
    this.column = column;
    this.numSegments = numSegments;
  }

  /** Logical name of the indexed vector. */
  public String getName() {
    return name;
  }

  /** Vector column the index was built on. */
  public String getColumn() {
    return column;
  }

  /** Number of physical segments composing this logical index. */
  public int getNumSegments() {
    return numSegments;
  }

  /**
   * Open the IVF view on this handle.
   *
   * <p>Today every committed vector index is IVF-backed; the {@link Optional} return type is
   * reserved for future non-IVF families.
   *
   * @throws IllegalStateException if this handle is closed
   */
  public Optional<IvfHandle> asIvf() {
    checkNotClosed();
    return Optional.of(new IvfHandle(this));
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      nativeRelease(nativeHandle);
      nativeHandle = 0;
    }
  }

  /** Internal: pointer accessor used by {@link IvfHandle} to dispatch native calls. */
  long nativeHandlePtr() {
    checkNotClosed();
    return nativeHandle;
  }

  private void checkNotClosed() {
    if (nativeHandle == 0) {
      throw new IllegalStateException("VectorIndexHandle is closed");
    }
  }

  private static native void nativeRelease(long handle);
}
