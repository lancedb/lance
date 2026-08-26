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

/**
 * Thrown when a Lance operation exceeds its configured timeout.
 *
 * <p>For example, a commit issued via {@link CommitBuilder} that does not complete within the
 * duration set by {@link CommitBuilder#commitTimeout(java.time.Duration)} fails with this
 * exception. It is unchecked so it does not change existing method signatures; catch it
 * specifically to distinguish a timeout from other failures.
 */
public class LanceTimeoutException extends RuntimeException {
  public LanceTimeoutException(String message) {
    super(message);
  }
}
