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
package org.lance.ipc;

import org.lance.Dataset;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Distributed filtered read support for Lance datasets.
 *
 * <p>Enables plan/execute separation for distributed engines (e.g., Spark):
 *
 * <ul>
 *   <li>Master node: calls {@link #planFilteredRead} to get a {@code FilteredRead} plan
 *   <li>Plan is serialized and distributed to workers
 *   <li>Workers: call {@link #executeFilteredRead} to execute tasks
 * </ul>
 */
public class FilteredRead implements Serializable {
  private static final long serialVersionUID = 1L;

  private final byte[] filteredReadExecProto;
  private final transient List<byte[]> splitProtos;
  private final transient int[] fragmentIds;
  private final transient long[] rowsPerFragment;

  /**
   * Package-private constructor called from JNI ({@code inner_create_plan} in {@code
   * blocking_scanner.rs}) to create a fully populated plan.
   *
   * @param filteredReadExecProto Prost-serialized bytes of the protobuf message {@code
   *     lance.datafusion.FilteredReadExecProto} (defined in {@code protos/filtered_read.proto}).
   *     This is the <b>complete</b> execution plan covering all fragments. It is produced by the
   *     following Rust serialization chain:
   *     <ol>
   *       <li>{@code Scanner::plan_filtered_read} (scanner.rs) builds a {@code FilteredReadExec}
   *           and calls {@code FilteredReadExec::get_or_create_plan} to compute the internal plan
   *           (fragment row ranges, per-fragment filters, scalar-index results).
   *       <li>{@code filtered_read_exec_to_proto} (filtered_read_proto.rs) converts the {@code
   *           FilteredReadExec} into a {@code FilteredReadExecProto}, which contains three parts:
   *           <ul>
   *             <li>{@code TableIdentifier} — dataset URI, version, manifest etag, and storage
   *                 options (via {@code table_identifier_from_dataset}).
   *             <li>{@code FilteredReadOptionsProto} — scan ranges, Substrait-encoded filter
   *                 expressions, projection (field IDs), batch size, fragment readahead, threading
   *                 mode, and IO buffer size (via {@code fr_options_to_proto}).
   *             <li>{@code FilteredReadPlanProto} — serialized {@code RowAddrTreeMap} (bitmap of
   *                 rows to read per fragment), optional {@code scan_range_after_filter}, and
   *                 deduplicated Substrait filter expressions mapped per fragment (via {@code
   *                 plan_to_proto}).
   *           </ul>
   *       <li>{@code proto.encode_to_vec()} (prost) encodes the protobuf message into the byte
   *           array passed here.
   *     </ol>
   *     This byte array is {@link Serializable} and can be sent to a remote executor. On the
   *     execution side, {@link #executeFilteredRead} passes it to {@code
   *     execute_filtered_read_from_bytes}, which decodes the proto, reconstructs a {@code
   *     FilteredReadExec} via {@code filtered_read_exec_from_proto}, and executes the scan.
   * @param splitProtos Per-fragment task protos for distributed execution. Each element is a
   *     Prost-serialized {@code FilteredReadExecProto} that contains the same {@code
   *     TableIdentifier} and {@code FilteredReadOptionsProto} as the full plan, but its {@code
   *     FilteredReadPlanProto} is scoped to a single fragment — the {@code RowAddrTreeMap} holds
   *     only that fragment's row bitmap, and the {@code fragment_filter_ids} / {@code
   *     filter_expressions} include only that fragment's filter. The global {@code
   *     scan_range_after_filter} is dropped from per-fragment plans because it can only be applied
   *     after aggregating results across all fragments. Produced by {@code
   *     split_and_inspect_plan_proto} (filtered_read_proto.rs), which decodes the full proto once,
   *     iterates over the {@code RowAddrTreeMap}, and re-serializes a scoped proto per fragment.
   *     This list is parallel to {@code fragmentIds} and {@code rowsPerFragment}. Marked {@code
   *     transient} — not included in Java serialization; the receiver should call {@code
   *     split_and_inspect_plan_proto} on the deserialized {@code filteredReadExecProto} to
   *     reconstruct them.
   * @param fragmentIds Array of Lance fragment IDs present in the plan, in {@code RowAddrTreeMap}
   *     iteration order (ascending by fragment ID). Each entry corresponds to the same index in
   *     {@code splitProtos} and {@code rowsPerFragment}. Produced by {@code
   *     split_and_inspect_plan_proto} which collects the keys of the deserialized {@code
   *     RowAddrTreeMap}. Marked {@code transient}.
   * @param rowsPerFragment Array of row counts parallel to {@code fragmentIds}. A value of {@code
   *     -1} means the entire fragment will be read ({@code RowAddrSelection::Full}); a non-negative
   *     value is the number of selected rows in the fragment's bitmap ({@code
   *     RowAddrSelection::Partial}). This allows the coordinator to estimate task sizes for
   *     load-balanced scheduling without deserializing each split proto. Produced by {@code
   *     split_and_inspect_plan_proto}. Marked {@code transient}.
   */
  FilteredRead(
      byte[] filteredReadExecProto,
      List<byte[]> splitProtos,
      int[] fragmentIds,
      long[] rowsPerFragment) {
    this.filteredReadExecProto = filteredReadExecProto;
    this.splitProtos = splitProtos;
    this.fragmentIds = fragmentIds;
    this.rowsPerFragment = rowsPerFragment;
  }

  /**
   * Plan a filtered read using the given scanner's settings. The scanner must be configured with
   * the desired filter, projection, etc.
   *
   * @param scanner a configured {@link LanceScanner}
   * @return a {@link FilteredRead} containing the serialized execution plan
   */
  public static FilteredRead planFilteredRead(LanceScanner scanner) {
    Preconditions.checkNotNull(scanner);
    return nativeCreatePlan(scanner);
  }

  /**
   * Execute a filtered read task, returning an {@link ArrowReader} over the results.
   *
   * @param dataset the dataset to read from
   * @param taskProto the serialized FilteredReadExecProto bytes for this task
   * @param allocator the buffer allocator
   * @return an {@link ArrowReader} over the scan results
   */
  public static ArrowReader executeFilteredRead(
      Dataset dataset, byte[] taskProto, BufferAllocator allocator) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(taskProto);
    Preconditions.checkNotNull(allocator);
    try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
      nativeExecuteFilteredRead(dataset, taskProto, s.memoryAddress());
      return Data.importArrayStream(allocator, s);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Get the serialized proto bytes. This is the opaque payload that can be sent to workers for
   * execution.
   *
   * @return the serialized FilteredReadExecProto bytes
   */
  public byte[] getFilteredReadExecProto() {
    return filteredReadExecProto;
  }

  /**
   * Split this plan into per-fragment task protos for distributed execution.
   *
   * <p>Each returned byte array contains a complete serialized {@code FilteredReadExecProto} for a
   * single fragment. The {@code scan_range_after_filter} is dropped from per-fragment plans because
   * it can only be applied globally by the coordinator after aggregating results from all workers.
   *
   * @return list of serialized task protos, one per fragment in the plan
   */
  public List<byte[]> getTasks() {
    return new ArrayList<>(splitProtos);
  }

  /**
   * Get the number of fragments in this plan.
   *
   * @return the number of fragments
   */
  public int getNumFragments() {
    return fragmentIds.length;
  }

  /**
   * Get the fragment IDs in this plan.
   *
   * @return array of fragment IDs
   */
  public int[] getFragmentIds() {
    return fragmentIds;
  }

  /**
   * Get the number of rows planned per fragment. A value of -1 indicates all rows in the fragment
   * will be read.
   *
   * @return array of row counts, parallel to {@link #getFragmentIds()}
   */
  public long[] getRowsPerFragment() {
    return rowsPerFragment;
  }

  /**
   * Native method to plan and split a filtered read from a configured scanner in a single call.
   *
   * @param scanner the configured LanceScanner
   * @return a {@link FilteredRead} containing the full proto, split protos, fragment IDs, and
   *     rows-per-fragment
   */
  static native FilteredRead nativeCreatePlan(LanceScanner scanner);

  /**
   * Native method to execute a filtered read task and write results to an Arrow stream.
   *
   * @param dataset the dataset to read from
   * @param filteredReadExecProto the serialized execution plan bytes for this task
   * @param streamAddress the memory address of an allocated ArrowArrayStream to receive results
   * @throws IOException if an error occurs during execution or stream export
   */
  static native void nativeExecuteFilteredRead(
      Dataset dataset, byte[] filteredReadExecProto, long streamAddress) throws IOException;
}
