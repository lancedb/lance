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

package com.lancedb.lance;

import java.util.Arrays;
import java.util.Optional;

import org.junit.jupiter.api.Test;

import com.lancedb.lance.test.JniTestHelper;

public class JNITest {
  @Test
  public void testInts() {
    JniTestHelper.parseInts(Arrays.asList(1, 2, 3));
  }

  @Test
  public void testIntsOpt() {
    JniTestHelper.parseIntsOpt(Optional.of(Arrays.asList(1, 2, 3)));
  }
}
