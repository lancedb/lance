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
package org.lance.namespace;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for CustomNamespace implementation that wraps DirectoryNamespace.
 *
 * <p>This test class extends DirectoryNamespaceTest to verify that all tests pass when using a
 * CustomNamespace wrapper around DirectoryNamespace. This validates that the Java-Rust binding
 * works correctly for custom namespace implementations.
 */
public class CustomNamespaceTest extends DirectoryNamespaceTest {

  @Override
  protected LanceNamespace wrapNamespace(DirectoryNamespace inner) {
    return new CustomNamespace(inner);
  }

  @Test
  void testCustomNamespaceId() {
    String namespaceId = namespaceClient.namespaceId();
    assertNotNull(namespaceId);
    assertTrue(
        namespaceId.startsWith("CustomNamespace["),
        "namespaceId should start with 'CustomNamespace[', got: " + namespaceId);
    assertTrue(
        namespaceId.contains("DirectoryNamespace"),
        "namespaceId should contain 'DirectoryNamespace', got: " + namespaceId);
  }

  @Test
  void testCustomNamespaceInnerAccess() {
    assertTrue(
        namespaceClient instanceof CustomNamespace, "namespaceClient should be a CustomNamespace");
    CustomNamespace customNs = (CustomNamespace) namespaceClient;
    assertNotNull(customNs.getInner(), "inner namespace should not be null");
    assertSame(
        innerNamespaceClient, customNs.getInner(), "inner namespace should be the same instance");
  }
}
