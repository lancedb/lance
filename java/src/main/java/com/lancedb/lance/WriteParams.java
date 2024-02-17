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

public class WriteParams {
    public enum WriteMode {
        CREATE,
        APPEND,
        OVERWRITE
    }

    private final int maxRowsPerFile;
    private final int maxRowsPerGroup;
    private final long maxBytesPerFile;
    private final WriteMode mode;

    private WriteParams(int maxRowsPerFile, int maxRowsPerGroup, long maxBytesPerFile, WriteMode mode) {
        this.maxRowsPerFile = maxRowsPerFile;
        this.maxRowsPerGroup = maxRowsPerGroup;
        this.maxBytesPerFile = maxBytesPerFile;
        this.mode = mode;
    }

    public int getMaxRowsPerFile() { return maxRowsPerFile; }
    public int getMaxRowsPerGroup() { return maxRowsPerGroup; }
    public long getMaxBytesPerFile() { return maxBytesPerFile; }
    public WriteMode getMode() { return mode; }

    public static class Builder {
        private int maxRowsPerFile = 1024 * 1024; // 1 million
        private int maxRowsPerGroup = 1024;
        private long maxBytesPerFile = 90L * 1024 * 1024 * 1024; // 90 GB
        private WriteMode mode = WriteMode.CREATE;

        public Builder withMaxRowsPerFile(int maxRowsPerFile) {
            this.maxRowsPerFile = maxRowsPerFile;
            return this;
        }

        public Builder withMaxRowsPerGroup(int maxRowsPerGroup) {
            this.maxRowsPerGroup = maxRowsPerGroup;
            return this;
        }

        public Builder withMaxBytesPerFile(long maxBytesPerFile) {
            this.maxBytesPerFile = maxBytesPerFile;
            return this;
        }

        public Builder withMode(WriteMode mode) {
            this.mode = mode;
            return this;
        }

        public WriteParams build() {
            return new WriteParams(this.maxRowsPerFile, this.maxRowsPerGroup, this.maxBytesPerFile, this.mode);
        }
    }
}