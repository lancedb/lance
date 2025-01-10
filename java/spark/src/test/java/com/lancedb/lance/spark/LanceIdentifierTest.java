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
package com.lancedb.lance.spark;

import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LanceIdentifierTest {
  @Test
  public void testLanceIdentifierFullyConstruct() {
    String url = "/tmp/data.lance";
    LanceIdentifier identifier = new LanceIdentifier(url);
    assertEquals(url, identifier.name());
    assertEquals(1, identifier.namespace().length);
    assertEquals("default", identifier.namespace()[0]);

    identifier = new LanceIdentifier(url, "catalog", "ods");
    assertEquals(2, identifier.namespace().length);
    assertEquals("catalog", identifier.namespace()[0]);
    assertEquals("ods", identifier.namespace()[1]);

    Map<String, String> options = new HashMap<>();
    options.put("key1", "value1");
    options.put("key2", "value2");
    identifier = new LanceIdentifier(url, options);
    assertEquals(4, identifier.namespace().length);
    assertEquals("default", identifier.namespace()[0]);
    assertEquals(LanceIdentifier.SEPARATOR, identifier.namespace()[1]);
    assertEquals("key1=value1", identifier.namespace()[2]);
    assertEquals("key2=value2", identifier.namespace()[3]);

    String[] namespace = new String[] {"spark_catalog", "default"};
    identifier = new LanceIdentifier(url, namespace, options);
    assertEquals(url, identifier.name());
    assertEquals(5, identifier.namespace().length);
    assertEquals("spark_catalog", identifier.namespace()[0]);
    assertEquals("default", identifier.namespace()[1]);
    assertEquals(LanceIdentifier.SEPARATOR, identifier.namespace()[2]);
    assertEquals("key1=value1", identifier.namespace()[3]);
    assertEquals("key2=value2", identifier.namespace()[4]);
  }

  @Test
  public void testGenLanceConfig() {
    Map<String, String> pros = new HashMap<>();
    pros.put("key1", "value11");
    pros.put("key3", "value3");
    CaseInsensitiveStringMap map = new CaseInsensitiveStringMap(pros);

    Map<String, String> options = new HashMap<>();
    options.put("key1", "value1");
    options.put("key2", "value2");
    LanceIdentifier identifier = new LanceIdentifier("/tmp/data.lance", options);

    LanceConfig config = identifier.genLanceConfig(map);
    assertEquals("data", config.getDatasetName());
    assertEquals("/tmp/data.lance", config.getDatasetUri());
    assertEquals("/tmp/", config.getDbPath());
    assertEquals(3, config.getOptions().size());
    assertEquals("value1", config.getOptions().get("key1"));
    assertEquals("value2", config.getOptions().get("key2"));
    assertEquals("value3", config.getOptions().get("key3"));
  }

  @Test
  public void testLanceIdentifierOf() {
    String url = "/tmp/data.lance";
    Map<String, String> options = new HashMap<>();
    options.put("key1", "value1");
    options.put("key2", "value2");
    String[] namespace = new String[] {"spark_catalog", "default"};
    LanceIdentifier identifier = new LanceIdentifier(url, namespace, options);

    CaseInsensitiveStringMap emptyOptions = new CaseInsensitiveStringMap(new HashMap<>());
    LanceIdentifier lanceIdentifier = LanceIdentifier.of(identifier);
    assertEquals(options, lanceIdentifier.genLanceConfig(emptyOptions).getOptions());

    Identifier identifierImpl = Identifier.of(identifier.namespace(), identifier.name());
    lanceIdentifier = LanceIdentifier.of(identifierImpl);
    assertEquals(options, lanceIdentifier.genLanceConfig(emptyOptions).getOptions());

    identifierImpl = Identifier.of(namespace, identifier.name());
    lanceIdentifier = LanceIdentifier.of(identifierImpl);
    assertEquals(0, lanceIdentifier.genLanceConfig(emptyOptions).getOptions().size());
    assertEquals(namespace, lanceIdentifier.namespace());
  }
}
