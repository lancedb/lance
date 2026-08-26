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

import org.lance.namespace.CustomNamespace;
import org.lance.namespace.DirectoryNamespace;
import org.lance.namespace.LanceNamespace;

import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/**
 * Integration tests for CustomNamespace wrapper around DirectoryNamespace.
 *
 * <p>This test class extends DirectoryNamespaceIntegrationTest to verify that all tests pass when
 * using a CustomNamespace wrapper around DirectoryNamespace. This validates that the Java-Rust
 * binding works correctly for custom namespace implementations in integration scenarios.
 *
 * <p>These tests require LocalStack to be running. Run with: docker compose up -d
 *
 * <p>Set LANCE_INTEGRATION_TEST=1 environment variable to enable these tests.
 */
@EnabledIfEnvironmentVariable(named = "LANCE_INTEGRATION_TEST", matches = "1")
public class CustomNamespaceIntegrationTest extends DirectoryNamespaceIntegrationTest {

  @Override
  protected LanceNamespace wrapNamespace(DirectoryNamespace inner) {
    return new CustomNamespace(inner);
  }
}
