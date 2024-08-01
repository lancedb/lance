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

package com.lancedb.lance.spark.utils;

import java.io.Serializable;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Consumer;

public class Optional<T> implements Serializable {
  private static final long serialVersionUID = 1L;
  private final T value;

  private Optional(T value) {
    this.value = value;
  }

  public static <T> Optional<T> of(T value) {
    return new Optional<>(Objects.requireNonNull(value));
  }

  public static <T> Optional<T> ofNullable(T value) {
    return new Optional<>(value);
  }

  public static <T> Optional<T> empty() {
    return new Optional<>(null);
  }

  public java.util.Optional<T> toStandardOptional() {
    return java.util.Optional.ofNullable(value);
  }

  public boolean isEmpty() {
    return value == null;
  }

  public boolean isPresent() {
    return value != null;
  }

  public T get() {
    if (value == null) {
      throw new NoSuchElementException("No value present");
    }
    return value;
  }

  public void ifPresent(Consumer<? super T> action) {
    if (value != null) {
      action.accept(value);
    }
  }

  @Override
  public String toString() {
    return value != null ? String.format("Optional[%s]", value) : "Optional.empty";
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(value);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null || getClass() != obj.getClass()) return false;
    Optional<?> other = (Optional<?>) obj;
    return Objects.equals(value, other.value);
  }
}
