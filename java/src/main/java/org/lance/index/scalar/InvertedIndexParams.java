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
package org.lance.index.scalar;

import org.lance.util.JsonUtils;

import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Builder-style configuration for inverted (full-text) scalar index parameters. */
public final class InvertedIndexParams {

  private static final String INDEX_TYPE = "inverted";

  private InvertedIndexParams() {}

  /**
   * Create a new builder for inverted index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for inverted scalar index parameters. */
  public static final class Builder {
    private String baseTokenizer;
    private String language;
    private Boolean withPosition;
    private Integer maxTokenLength;
    private Boolean lowerCase;
    private Boolean stem;
    private Boolean removeStopWords;
    private List<String> customStopWords;
    private Boolean asciiFolding;
    private Integer minNgramLength;
    private Integer maxNgramLength;
    private Boolean prefixOnly;
    private Boolean skipMerge;

    /**
     * Configure the base tokenizer.
     *
     * <p>Supported values include:
     *
     * <ul>
     *   <li>{@code "simple"} (default): splits tokens on whitespace and punctuation
     *   <li>{@code "whitespace"}: splits tokens on whitespace
     *   <li>{@code "raw"}: no tokenization
     *   <li>{@code "ngram"}: N-Gram tokenizer
     *   <li>{@code "lindera/*"}: Lindera tokenizer
     *   <li>{@code "jieba/*"}: Jieba tokenizer
     * </ul>
     *
     * @param baseTokenizer tokenizer identifier string
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder baseTokenizer(String baseTokenizer) {
      Objects.requireNonNull(baseTokenizer, "baseTokenizer must not be null");
      if (baseTokenizer.isEmpty()) {
        throw new IllegalArgumentException("baseTokenizer must not be empty");
      }
      this.baseTokenizer = baseTokenizer;
      return this;
    }

    /**
     * Configure the language used for stemming and stop words.
     *
     * @param language language name understood by Tantivy, for example {@code "English"}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder language(String language) {
      Objects.requireNonNull(language, "language must not be null");
      if (language.isEmpty()) {
        throw new IllegalArgumentException("language must not be empty");
      }
      this.language = language;
      return this;
    }

    /**
     * Configure whether to store token positions in the index.
     *
     * @param withPosition whether to store term positions
     * @return this builder
     */
    public Builder withPosition(boolean withPosition) {
      this.withPosition = withPosition;
      return this;
    }

    /**
     * Configure the maximum token length.
     *
     * @param maxTokenLength maximum token length, must be positive
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder maxTokenLength(Integer maxTokenLength) {
      if (maxTokenLength == null || maxTokenLength <= 0) {
        throw new IllegalArgumentException("maxTokenLength must be positive when specified");
      }
      this.maxTokenLength = maxTokenLength;
      return this;
    }

    /**
     * Configure whether to lower case tokens.
     *
     * @param lowerCase whether to lower case tokens
     * @return this builder
     */
    public Builder lowerCase(boolean lowerCase) {
      this.lowerCase = lowerCase;
      return this;
    }

    /**
     * Configure whether to apply stemming.
     *
     * @param stem whether to apply stemming
     * @return this builder
     */
    public Builder stem(boolean stem) {
      this.stem = stem;
      return this;
    }

    /**
     * Configure whether to remove stop words.
     *
     * @param removeStopWords whether to remove stop words
     * @return this builder
     */
    public Builder removeStopWords(boolean removeStopWords) {
      this.removeStopWords = removeStopWords;
      return this;
    }

    /**
     * Configure custom stop words. When set, these override the built-in stop word list for the
     * configured language.
     *
     * @param customStopWords list of stop words
     * @return this builder
     */
    public Builder customStopWords(List<String> customStopWords) {
      Objects.requireNonNull(customStopWords, "customStopWords must not be null");
      this.customStopWords = new ArrayList<>(customStopWords);
      return this;
    }

    /**
     * Configure whether to apply ASCII folding
     *
     * @param asciiFolding whether to enable ASCII folding
     * @return this builder
     */
    public Builder asciiFolding(boolean asciiFolding) {
      this.asciiFolding = asciiFolding;
      return this;
    }

    /**
     * Configure the minimum N-gram length (only used when {@code baseTokenizer = "ngram"}).
     *
     * @param minNgramLength minimum N-gram length, must be &gt; 0 and &lt;= {@code maxNgramLength}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder minNgramLength(int minNgramLength) {
      if (minNgramLength <= 0) {
        throw new IllegalArgumentException("minNgramLength must be positive");
      }
      this.minNgramLength = minNgramLength;
      return this;
    }

    /**
     * Configure the maximum N-gram length (only used when {@code baseTokenizer = "ngram"}).
     *
     * @param maxNgramLength maximum N-gram length, must be &gt; 0 and &gt;= {@code minNgramLength}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder maxNgramLength(int maxNgramLength) {
      if (maxNgramLength <= 0) {
        throw new IllegalArgumentException("maxNgramLength must be positive");
      }
      this.maxNgramLength = maxNgramLength;
      return this;
    }

    /**
     * Configure whether only prefix N-grams are generated (only used when {@code baseTokenizer =
     * "ngram"}).
     *
     * @param prefixOnly whether to generate only prefix N-grams
     * @return this builder
     */
    public Builder prefixOnly(boolean prefixOnly) {
      this.prefixOnly = prefixOnly;
      return this;
    }

    /**
     * Configure whether to skip the partition merge stage after indexing. If true, skip the
     * partition merge stage after indexing. This can be useful for distributed indexing where merge
     * is handled separately.
     *
     * @param skipMerge whether to skip partition merge
     * @return this builder
     */
    public Builder skipMerge(boolean skipMerge) {
      this.skipMerge = skipMerge;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for an inverted index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (baseTokenizer != null) {
        params.put("base_tokenizer", baseTokenizer);
      }
      if (language != null) {
        params.put("language", language);
      }
      if (withPosition != null) {
        params.put("with_position", withPosition);
      }
      if (maxTokenLength != null) {
        params.put("max_token_length", maxTokenLength);
      }
      if (lowerCase != null) {
        params.put("lower_case", lowerCase);
      }
      if (stem != null) {
        params.put("stem", stem);
      }
      if (removeStopWords != null) {
        params.put("remove_stop_words", removeStopWords);
      }
      if (customStopWords != null) {
        params.put("custom_stop_words", new ArrayList<>(customStopWords));
      }
      if (asciiFolding != null) {
        params.put("ascii_folding", asciiFolding);
      }
      if (minNgramLength != null) {
        params.put("min_ngram_length", minNgramLength);
      }
      if (maxNgramLength != null) {
        Preconditions.checkArgument(
            minNgramLength == null || maxNgramLength >= minNgramLength,
            "maxNgramLength {} shouldn't less than minNgramLength {}",
            maxNgramLength,
            minNgramLength);
        params.put("max_ngram_length", maxNgramLength);
      }
      if (prefixOnly != null) {
        params.put("prefix_only", prefixOnly);
      }
      if (skipMerge != null) {
        params.put("skip_merge", skipMerge);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
