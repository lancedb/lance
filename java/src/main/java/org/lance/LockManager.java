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
package org.lance;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** The LockManager class provides a way to manage read and write locks. */
public class LockManager {
  private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
  private final AtomicBoolean rejectWriteLocks;

  public LockManager() {
    this(new AtomicBoolean(false));
  }

  LockManager(AtomicBoolean rejectWriteLocks) {
    this.rejectWriteLocks = rejectWriteLocks;
  }

  /**
   * Represents a read lock for the LockManager. This lock allows multiple threads to read
   * concurrently, but prevents write access.
   */
  public class ReadLock implements AutoCloseable {
    /** Acquires a read lock on the lock manager. */
    public ReadLock() {
      lock.readLock().lock();
    }

    @Override
    public void close() {
      lock.readLock().unlock();
    }
  }

  /** Represents a write lock that can be acquired and released. */
  public class WriteLock implements AutoCloseable {
    /** Constructs a new WriteLock and acquires the write lock. */
    public WriteLock() {
      if (rejectWriteLocks.get()) {
        throw new IllegalStateException("Dataset is already creating an index");
      }
      lock.writeLock().lock();
    }

    @Override
    public void close() {
      lock.writeLock().unlock();
    }
  }

  /**
   * Acquires a read lock on the LockManager.
   *
   * @return the acquired ReadLock object
   */
  public ReadLock acquireReadLock() {
    return new ReadLock();
  }

  /**
   * Acquires a write lock on the LockManager.
   *
   * @return the acquired WriteLock object
   */
  public WriteLock acquireWriteLock() {
    return new WriteLock();
  }
}
