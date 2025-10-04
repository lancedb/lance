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
package com.lancedb.lance.blob;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.CompletableFuture;

/**
 * A file-like object that represents a blob in a Lance dataset.
 *
 * BlobFile provides both synchronous and asynchronous access to blob data stored
 * in Lance datasets. It supports standard file operations like read, seek, and tell,
 * with proper resource management through the Closeable interface.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (BlobFile blobFile = dataset.takeBlobs(Arrays.asList(0L), "blob_column").get(0)) {
 *     // Read entire blob
 *     byte[] data = blobFile.read();
 *
 *     // Read specific amount
 *     blobFile.seek(100);
 *     byte[] chunk = blobFile.readUpTo(50);
 *
 *     // Get file info
 *     long position = blobFile.tell();
 *     long size = blobFile.size();
 * }
 * }</pre>
 */
public class BlobFile implements Closeable {

  private long nativeBlobFileHandle;
  private boolean closed = false;

  // Package-private constructor - should only be created by Dataset
  BlobFile(long nativeBlobFileHandle) {
    this.nativeBlobFileHandle = nativeBlobFileHandle;
  }

  /**
   * Read the entire blob file from the current cursor position to the end of the file.
   * After this call, the cursor will be pointing to the end of the file.
   *
   * @return the blob data as a byte array
   * @throws IOException if the blob file is closed or an I/O error occurs
   */
  public byte[] read() throws IOException {
    checkNotClosed();
    return nativeRead(nativeBlobFileHandle);
  }

  /**
   * Read the entire blob file asynchronously from the current cursor position to the end.
   *
   * @return a CompletableFuture containing the blob data as a byte array
   */
  public CompletableFuture<byte[]> readAsync() {
    if (closed) {
      CompletableFuture<byte[]> future = new CompletableFuture<>();
      future.completeExceptionally(new IOException("BlobFile is closed"));
      return future;
    }
    return CompletableFuture.supplyAsync(() -> {
      try {
        return nativeReadAsync(nativeBlobFileHandle);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Read up to the specified number of bytes from the current cursor position.
   * After this call, the cursor will be pointing to the end of the read data.
   *
   * @param length the maximum number of bytes to read
   * @return the blob data as a byte array (may be shorter than requested if EOF is reached)
   * @throws IOException if the blob file is closed or an I/O error occurs
   * @throws IllegalArgumentException if length is negative
   */
  public byte[] readUpTo(int length) throws IOException {
    if (length < 0) {
      throw new IllegalArgumentException("Length cannot be negative");
    }
    checkNotClosed();
    return nativeReadUpTo(nativeBlobFileHandle, length);
  }

  /**
   * Read up to the specified number of bytes asynchronously.
   *
   * @param length the maximum number of bytes to read
   * @return a CompletableFuture containing the blob data as a byte array
   */
  public CompletableFuture<byte[]> readUpToAsync(int length) {
    if (length < 0) {
      CompletableFuture<byte[]> future = new CompletableFuture<>();
      future.completeExceptionally(new IllegalArgumentException("Length cannot be negative"));
      return future;
    }
    if (closed) {
      CompletableFuture<byte[]> future = new CompletableFuture<>();
      future.completeExceptionally(new IOException("BlobFile is closed"));
      return future;
    }
    return CompletableFuture.supplyAsync(() -> {
      try {
        return nativeReadUpToAsync(nativeBlobFileHandle, length);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Seek to a new cursor position in the file.
   *
   * @param position the new cursor position (0-based)
   * @throws IOException if the blob file is closed or an I/O error occurs
   * @throws IllegalArgumentException if position is negative or beyond file size
   */
  public void seek(long position) throws IOException {
    if (position < 0) {
      throw new IllegalArgumentException("Position cannot be negative");
    }
    checkNotClosed();
    nativeSeek(nativeBlobFileHandle, position);
  }

  /**
   * Seek to a new cursor position asynchronously.
   *
   * @param position the new cursor position (0-based)
   * @return a CompletableFuture that completes when the seek operation is done
   */
  public CompletableFuture<Void> seekAsync(long position) {
    if (position < 0) {
      CompletableFuture<Void> future = new CompletableFuture<>();
      future.completeExceptionally(new IllegalArgumentException("Position cannot be negative"));
      return future;
    }
    if (closed) {
      CompletableFuture<Void> future = new CompletableFuture<>();
      future.completeExceptionally(new IOException("BlobFile is closed"));
      return future;
    }
    return CompletableFuture.runAsync(() -> {
      try {
        nativeSeekAsync(nativeBlobFileHandle, position);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Return the current cursor position in the file.
   *
   * @return the current cursor position (0-based)
   * @throws IOException if the blob file is closed or an I/O error occurs
   */
  public long tell() throws IOException {
    checkNotClosed();
    return nativeTell(nativeBlobFileHandle);
  }

  /**
   * Return the current cursor position asynchronously.
   *
   * @return a CompletableFuture containing the current cursor position
   */
  public CompletableFuture<Long> tellAsync() {
    if (closed) {
      CompletableFuture<Long> future = new CompletableFuture<>();
      future.completeExceptionally(new IOException("BlobFile is closed"));
      return future;
    }
    return CompletableFuture.supplyAsync(() -> {
      try {
        return nativeTellAsync(nativeBlobFileHandle);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Return the size of the blob file in bytes.
   *
   * @return the size of the blob file in bytes
   */
  public long size() {
    if (closed) {
      return 0;
    }
    return nativeSize(nativeBlobFileHandle);
  }

  /**
   * Check if the blob file is closed.
   *
   * @return true if the blob file is closed, false otherwise
   */
  public boolean isClosed() {
    return closed;
  }

  /**
   * Close the blob file and release any associated resources.
   * This method is idempotent - calling it multiple times has no effect.
   */
  @Override
  public void close() throws IOException {
    if (!closed) {
      try {
        nativeClose(nativeBlobFileHandle);
      } finally {
        closed = true;
        nativeBlobFileHandle = 0;
      }
    }
  }

  /**
   * Close the blob file asynchronously.
   *
   * @return a CompletableFuture that completes when the close operation is done
   */
  public CompletableFuture<Void> closeAsync() {
    if (closed) {
      return CompletableFuture.completedFuture(null);
    }
    return CompletableFuture.runAsync(() -> {
      try {
        close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  private void checkNotClosed() throws IOException {
    if (closed) {
      throw new IOException("BlobFile is closed");
    }
  }

  @Override
  protected void finalize() throws Throwable {
    try {
      if (!closed) {
        close();
      }
    } finally {
      super.finalize();
    }
  }

  // Native method declarations
  private static native byte[] nativeRead(long handle) throws IOException;
  private static native byte[] nativeReadAsync(long handle) throws IOException;
  private static native byte[] nativeReadUpTo(long handle, int length) throws IOException;
  private static native byte[] nativeReadUpToAsync(long handle, int length) throws IOException;
  private static native void nativeSeek(long handle, long position) throws IOException;
  private static native void nativeSeekAsync(long handle, long position) throws IOException;
  private static native long nativeTell(long handle) throws IOException;
  private static native long nativeTellAsync(long handle) throws IOException;
  private static native long nativeSize(long handle);
  private static native void nativeClose(long handle) throws IOException;
}