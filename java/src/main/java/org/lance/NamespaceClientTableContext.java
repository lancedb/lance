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

import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DescribeTableResponse;

import org.apache.arrow.util.Preconditions;

import java.util.Map;

/**
 * Cached context from a namespace client's {@code describeTable} or {@code declareTable} response.
 *
 * <p>Contains only the resolved table metadata (location, storage options, managed-versioning
 * flag). The namespace client and table ID are <b>not</b> part of this class — they are still
 * passed separately.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * DescribeTableResponse response = namespaceClient.describeTable(request);
 * NamespaceClientTableContext namespaceClientTableContext =
 *     NamespaceClientTableContext.fromDescribeTableResponse(response);
 *
 * Dataset dataset = Dataset.open()
 *     .namespaceClient(namespaceClient)
 *     .tableId(tableId)
 *     .namespaceClientTableContext(namespaceClientTableContext)
 *     .build();
 * }</pre>
 */
public class NamespaceClientTableContext {

  private final String location;
  private final Map<String, String> storageOptions;
  private final boolean managedVersioning;

  /**
   * Creates a new NamespaceClientTableContext.
   *
   * @param location the table's storage location (URI)
   * @param storageOptions storage options returned by the namespace (e.g. temporary credentials),
   *     may be null
   * @param managedVersioning whether commits should go through the namespace's version API
   */
  public NamespaceClientTableContext(
      String location, Map<String, String> storageOptions, boolean managedVersioning) {
    Preconditions.checkNotNull(location, "location must not be null");
    this.location = location;
    this.storageOptions = storageOptions;
    this.managedVersioning = managedVersioning;
  }

  /**
   * Build a context from a {@link DescribeTableResponse}.
   *
   * @param response the describe table response
   * @return a new NamespaceClientTableContext
   * @throws IllegalArgumentException if the response does not contain a location
   */
  public static NamespaceClientTableContext fromDescribeTableResponse(
      DescribeTableResponse response) {
    String location = response.getLocation();
    Preconditions.checkArgument(
        location != null && !location.isEmpty(), "DescribeTableResponse missing location");
    return new NamespaceClientTableContext(
        location,
        response.getStorageOptions(),
        Boolean.TRUE.equals(response.getManagedVersioning()));
  }

  /**
   * Build a context from a {@link DeclareTableResponse}.
   *
   * @param response the declare table response
   * @return a new NamespaceClientTableContext
   * @throws IllegalArgumentException if the response does not contain a location
   */
  public static NamespaceClientTableContext fromDeclareTableResponse(
      DeclareTableResponse response) {
    String location = response.getLocation();
    Preconditions.checkArgument(
        location != null && !location.isEmpty(), "DeclareTableResponse missing location");
    return new NamespaceClientTableContext(
        location,
        response.getStorageOptions(),
        Boolean.TRUE.equals(response.getManagedVersioning()));
  }

  /** Returns the table's storage location (URI). */
  public String getLocation() {
    return location;
  }

  /** Returns the storage options, or null if none were provided. */
  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  /** Returns whether commits should go through the namespace's version API. */
  public boolean isManagedVersioning() {
    return managedVersioning;
  }
}
