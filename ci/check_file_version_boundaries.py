#!/usr/bin/env python3

"""Enforce the exact Lance file-version composition boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EXACT_VERSION_MODULES = ("v1", "v2_0", "v2_1", "v2_2", "v2_3")

VERSION_SELECTOR_ALLOWLIST = {
    Path("rust/lance-file/src/version.rs"),
    Path("rust/lance/src/dataset/write.rs"),
    Path("rust/lance/src/dataset/write/insert.rs"),
    Path("rust/lance/src/dataset/write/commit.rs"),
}

EXACT_COMPARISON_ALLOWLIST = {
    Path("rust/lance-file/src/reader.rs"),
    Path("rust/lance-file/src/version.rs"),
    Path("rust/lance-table/src/format/fragment.rs"),
    Path("rust/lance-table/src/format/manifest.rs"),
    Path("rust/lance/src/dataset/transaction.rs"),
}

EXACT_IDENTITY_ALLOWLIST = {
    # Persisted identity codecs and file open/create boundaries.
    Path("rust/lance-file/src/lib.rs"),
    Path("rust/lance-file/src/reader.rs"),
    Path("rust/lance-file/src/version.rs"),
    # Persisted manifest and DataFile identity ownership.
    Path("rust/lance-table/src/format/fragment.rs"),
    Path("rust/lance-table/src/format/manifest.rs"),
    Path("rust/lance-table/src/io/manifest.rs"),
    # Dataset API, persisted validation, and explicit physical-file boundaries.
    Path("rust/lance/src/dataset/index.rs"),
    Path("rust/lance/src/dataset/mem_wal/memtable/flush.rs"),
    Path("rust/lance/src/dataset/optimize/binary_copy.rs"),
    Path("rust/lance/src/dataset/transaction.rs"),
    Path("rust/lance/src/dataset/write.rs"),
    Path("rust/lance/src/dataset/write/commit.rs"),
    Path("rust/lance/src/dataset/write/insert.rs"),
    Path("rust/lance/src/index/vector/builder.rs"),
    # lance-index owns explicit physical Lance files, not dataset format policy.
    Path("rust/lance-index/src/scalar/lance_format.rs"),
    Path("rust/lance-index/src/vector/distributed/index_merger.rs"),
    Path("rust/lance-index/src/vector/ivf/shuffler.rs"),
    Path("rust/lance-index/src/vector/v3/shuffler.rs"),
}

LEGACY_READER_ESCAPE_HATCHES = (
    "as_legacy_opt",
    "as_legacy_opt_mut",
    "legacy_read_page_stats",
    "legacy_read_batch_projected",
    "legacy_row_group_size",
)

PREVIOUS_FILE_IDENTITY_ALIASES = (
    "PreviousFileReader",
    "PreviousFileWriter",
    "PreviousFileWriterOptions",
    "PreviousManifestProvider",
    "PreviousIndexReader",
    "previous_read_batch",
    "previous_write_batch",
)

VERSION_CAPABILITY_ESCAPE_HATCHES = (
    "BINARY_COPY_SUPPORTED",
    "PhysicalColumnLayout",
    "binary_copy_supported",
    "has_legacy_files",
    "is_legacy_file",
    "is_external_metadata_structural_header",
    "lance_supports_nulls",
    "nulls_supported",
    "physical_column_layout",
    "scan_strategy",
    "should_load_indexed_metadata",
    "should_try_indexed_metadata",
    "should_use_legacy_format",
    "supports_indexed_metadata",
    "supports_indexed_projection",
)

FORBIDDEN_VERSION_POLICY_PROXIES = (
    "BlobWrite",
    "IndexedMetadataSupport",
    "ScanStrategy",
    "WriteBatching",
)

FORBIDDEN_SHARED_WRITER_PROFILE_FIELDS = (
    "format_version",
    "version",
    "encoding_strategy",
    "footer_numbers",
    "major_version",
    "minor_version",
    "column_layout",
    "page_encoding",
    "buffer_alignment",
)

FORBIDDEN_SHARED_READER_PROFILE_FIELDS = (
    "version",
    "encoding_strategy",
    "footer_numbers",
    "major_version",
    "minor_version",
    "column_layout",
    "page_encoding",
    "buffer_alignment",
    "accepted_grammar",
)

FORBIDDEN_EXACT_CONSTRUCTOR_INPUTS = (
    "LanceFileVersion",
    "ConcreteFileVersion",
    "footer_numbers",
    "major_version",
    "minor_version",
    "column_layout",
    "page_encoding",
    "buffer_alignment",
    "encoding_strategy",
)

FORBIDDEN_ENCODING_PROFILE_TYPES = (
    "BlobEncoding",
    "BlockCompression",
    "ConstantEncoding",
    "DefaultCompressionStrategy",
    "FixedSizeListEncoding",
    "MapEncoding",
    "MinichunkSize",
    "PackedStructEncoding",
    "PrimitiveEncodingOptions",
    "RleEncoding",
    "SparseEncoding",
)

FORBIDDEN_ENCODING_VERSION_MIRRORS = (
    "BenchFileVersion",
    "TestFileVersion",
)

FORBIDDEN_ORDERED_ENCODING_TEST_APIS = (
    "with_max_file_version",
    "with_min_file_version",
)

FORBIDDEN_COMPRESSION_SELECTION_ABSTRACTIONS = (
    "CompressionAtom",
    "CompressionPipeline",
)

FORBIDDEN_FIELD_SELECTION_ABSTRACTIONS = (
    "FieldEncodingAtom",
    "StructuralEncodingStrategy",
)

ENCODING_COMPOSITION_CONTAINERS = {
    Path("rust/lance-encoding/src/encoder/structural.rs"): {
        "PrimitiveFieldEncoding": {"page_encodings"},
    },
    Path("rust/lance-encoding/src/encodings/logical/primitive.rs"): {
        "PrimitivePageEncoding": {"behavior"},
    },
    Path("rust/lance-file/src/versions/v2_1/compression.rs"): {
        "Strategy": {"params"},
    },
    Path("rust/lance-file/src/versions/v2_2/compression.rs"): {
        "Strategy": {"params"},
    },
    Path("rust/lance-file/src/versions/v2_3/compression.rs"): {
        "Strategy": {"params"},
    },
}


def is_test_source(path: Path) -> bool:
    return (
        "tests" in path.parts
        or "benches" in path.parts
        or path.name in {"test.rs", "testing.rs"}
    )


def mask_span(source: str, start: int, end: int) -> str:
    masked = "".join("\n" if char == "\n" else " " for char in source[start:end])
    return source[:start] + masked + source[end:]


def strip_cfg_test_items(source: str) -> str:
    pattern = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")
    while match := pattern.search(source):
        item_start = match.start()
        cursor = match.end()
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1

        brace = source.find("{", cursor)
        semicolon = source.find(";", cursor)
        if semicolon != -1 and (brace == -1 or semicolon < brace):
            item_end = semicolon + 1
        elif brace != -1:
            depth = 0
            item_end = len(source)
            for index in range(brace, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        item_end = index + 1
                        break
        else:
            item_end = len(source)
        source = mask_span(source, item_start, item_end)
    return source


def strip_comments_and_strings(source: str) -> str:
    output = list(source)
    index = 0
    block_depth = 0
    in_string = False
    while index < len(source):
        if block_depth:
            if source.startswith("/*", index):
                block_depth += 1
                output[index : index + 2] = "  "
                index += 2
            elif source.startswith("*/", index):
                block_depth -= 1
                output[index : index + 2] = "  "
                index += 2
            else:
                if source[index] != "\n":
                    output[index] = " "
                index += 1
            continue

        if in_string:
            if source[index] == "\\":
                output[index] = " "
                if index + 1 < len(source):
                    output[index + 1] = " "
                index += 2
            else:
                if source[index] == '"':
                    in_string = False
                if source[index] != "\n":
                    output[index] = " "
                index += 1
            continue

        if source.startswith("//", index):
            line_end = source.find("\n", index)
            if line_end == -1:
                line_end = len(source)
            output[index:line_end] = " " * (line_end - index)
            index = line_end
        elif source.startswith("/*", index):
            block_depth = 1
            output[index : index + 2] = "  "
            index += 2
        elif source[index] == '"':
            in_string = True
            output[index] = " "
            index += 1
        else:
            index += 1
    return "".join(output)


def production_source(source: str) -> str:
    return strip_comments_and_strings(strip_cfg_test_items(source))


def line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def enum_derives(source: str, enum_name: str) -> set[str]:
    match = re.search(
        rf"#\s*\[\s*derive\s*\((?P<traits>[^)]*)\)\s*\]\s*pub\s+enum\s+{enum_name}\b",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        return set()
    return {trait.strip().split("::")[-1] for trait in match.group("traits").split(",")}


def struct_body(source: str, struct_name: str) -> str:
    match = re.search(rf"\bstruct\s+{struct_name}\b[^{{]*{{", source)
    if match is None:
        return ""
    brace = source.find("{", match.start())
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    return source[brace + 1 :]


def struct_fields(source: str, struct_name: str) -> set[str]:
    body = struct_body(source, struct_name)
    return {
        match.group("name")
        for match in re.finditer(
            r"(?:^|,)\s*(?:pub(?:\s*\([^)]*\))?\s+)?"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:",
            body,
            flags=re.MULTILINE,
        )
    }


def public_function_parameters(source: str) -> list[tuple[str, str, int]]:
    functions = []
    pattern = re.compile(
        r"\bpub(?:\s*\([^)]*\))?\s+(?:async\s+)?fn\s+"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    )
    for match in pattern.finditer(source):
        opening = source.find("(", match.end())
        if opening == -1:
            continue
        depth = 0
        for index in range(opening, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    functions.append(
                        (
                            match.group("name"),
                            source[opening + 1 : index],
                            line_number(source, match.start()),
                        )
                    )
                    break
    return functions


def exact_version_module(relative: Path) -> str | None:
    prefix = ("rust", "lance-file", "src", "versions")
    if relative.parts[: len(prefix)] != prefix or len(relative.parts) <= len(prefix):
        return None
    candidate = relative.parts[len(prefix)]
    return candidate if candidate in EXACT_VERSION_MODULES else None


def sibling_version_references(source: str, current: str) -> list[tuple[int, str]]:
    references = []
    version_pattern = "|".join(re.escape(version) for version in EXACT_VERSION_MODULES)
    patterns = (
        re.compile(rf"\b(?:crate::)?versions::(?P<version>{version_pattern})\b"),
        re.compile(rf"\b(?:super::)+(?P<version>{version_pattern})\b"),
        re.compile(
            r"\b(?:pub(?:\s*\([^)]*\))?\s+)?use\s+"
            r"(?P<statement>[^;]*\bversions\b[^;]*);",
            flags=re.DOTALL,
        ),
    )
    seen: set[tuple[int, str]] = set()
    for pattern in patterns:
        for match in pattern.finditer(source):
            if "statement" in match.groupdict():
                for version_match in re.finditer(
                    rf"\b(?P<version>{version_pattern})\b",
                    match.group("statement"),
                ):
                    version = version_match.group("version")
                    offset = match.start("statement") + version_match.start()
                    if version != current:
                        seen.add((offset, version))
            else:
                version = match.group("version")
                if version != current:
                    seen.add((match.start("version"), version))
    for offset, version in sorted(seen):
        references.append((line_number(source, offset), version))
    return references


def comparison_violations(source: str) -> list[tuple[int, str]]:
    violations = []
    patterns = (
        re.compile(
            r"(?:ConcreteFileVersion|LanceFileVersion)::[A-Za-z0-9_]+"
            r"[^\n;]*(?:==|!=|>=|<=|\.max\s*\(|\.min\s*\()"
        ),
        re.compile(
            r"(?:==|!=|>=|<=|\.max\s*\(|\.min\s*\()[^\n;]*"
            r"(?:ConcreteFileVersion|LanceFileVersion)::[A-Za-z0-9_]+"
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(source):
            violations.append(
                (line_number(source, match.start()), match.group(0).strip())
            )

    aliases = {"ConcreteFileVersion", "LanceFileVersion"}
    aliases.update(
        match.group("alias")
        for match in re.finditer(
            r"\b(?:ConcreteFileVersion|LanceFileVersion)\s+as\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)",
            source,
        )
    )
    type_pattern = "|".join(re.escape(alias) for alias in sorted(aliases))
    typed_variables = {
        match.group("name")
        for match in re.finditer(
            rf"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:&\s*)?(?:Option\s*<\s*)?(?:{type_pattern})\b",
            source,
        )
    }
    typed_variables.update(
        match.group("name")
        for match in re.finditer(
            r"\blet\s+(?:mut\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\s*:\s*[^=;]+)?\s*=\s*[^;]*"
            r"(?:"
            r"\.lance_file_format\s*\("
            r"|\.file_version\s*\("
            r"|\bdetermine_file_version\s*\("
            r"|\bstable_file_version\s*\("
            r"|(?:version|file_version|format_version|storage_version)\s*\.resolve\s*\("
            r")",
            source,
        )
    )
    for variable in typed_variables:
        standalone_variable = rf"(?<![A-Za-z0-9_.]){re.escape(variable)}\b"
        variable_patterns = (
            re.compile(rf"{standalone_variable}\s*(?:==|!=|>=|<=)"),
            re.compile(rf"(?:==|!=|>=|<=)\s*\b{re.escape(variable)}\b"),
            re.compile(rf"{standalone_variable}\s*\.\s*(?:max|min)\s*\("),
            re.compile(rf"\bmatch\s+{re.escape(variable)}\b"),
        )
        for pattern in variable_patterns:
            for match in pattern.finditer(source):
                violations.append(
                    (line_number(source, match.start()), match.group(0).strip())
                )
    return violations


def exact_identity_allowed(relative: Path) -> bool:
    return (
        relative in EXACT_IDENTITY_ALLOWLIST
        or relative.is_relative_to("rust/lance-file/src/versions")
        or relative.is_relative_to("rust/lance/src/dataset/versions")
    )


def check_source(path: Path, source: str) -> list[str]:
    relative = path if not path.is_absolute() else path.relative_to(REPO_ROOT)
    source_without_comments = strip_comments_and_strings(source)
    errors: list[str] = []

    if (
        relative == Path("rust/lance-io/src/encodings.rs")
        or relative.is_relative_to("rust/lance-io/src/encodings")
        or re.search(r"\blance_io\s*::\s*encodings\b", source_without_comments)
        or re.search(
            r"\blance_io\s*::\s*\{[^}]*\bencodings\b",
            source_without_comments,
            flags=re.DOTALL,
        )
        or (
            relative.is_relative_to("rust/lance-io/src")
            and re.search(
                r"\b(?:pub\s+)?mod\s+encodings\b"
                r"|\b(?:crate|super)::encodings\b",
                source_without_comments,
            )
        )
    ):
        errors.append(
            f"{relative}: v1 encoding grammar leaked into lance-io; "
            "use lance-file::versions::v1::encoding"
        )

    if (
        relative == Path("rust/lance-encoding/src/previous.rs")
        or relative.is_relative_to("rust/lance-encoding/src/previous")
        or re.search(r"\blance_encoding\s*::\s*previous\b", source_without_comments)
        or re.search(
            r"\blance_encoding\s*::\s*\{[^}]*\bprevious\b",
            source_without_comments,
            flags=re.DOTALL,
        )
        or (
            relative.is_relative_to("rust/lance-encoding/src")
            and re.search(
                r"\b(?:pub\s+)?mod\s+previous\b"
                r"|\b(?:crate|super)::previous\b",
                source_without_comments,
            )
        )
    ):
        errors.append(
            f"{relative}: time-relative lance-encoding facade remains; "
            "use array_encoding or the shared encoder/decoder runtime"
        )

    if relative.is_relative_to("rust/lance-encoding"):
        for name in FORBIDDEN_ENCODING_VERSION_MIRRORS:
            if re.search(rf"\b{re.escape(name)}\b", source_without_comments):
                errors.append(
                    f"{relative}: encoding test/bench mirrors file identity as {name}; "
                    "name the executable encoding semantics instead"
                )
        for name in FORBIDDEN_ORDERED_ENCODING_TEST_APIS:
            if re.search(rf"\b{re.escape(name)}\b", source_without_comments):
                errors.append(
                    f"{relative}: ordered encoding test gate remains: {name}; "
                    "list the intended semantic encodings explicitly"
                )
        for enum_name in ("BenchEncoding", "TestEncoding"):
            forbidden = enum_derives(source_without_comments, enum_name) & {
                "Ord",
                "PartialOrd",
            }
            if forbidden:
                errors.append(
                    f"{relative}: {enum_name} derives forbidden ordering trait(s): "
                    + ", ".join(sorted(forbidden))
                )

    if is_test_source(relative):
        return errors

    production = production_source(source)

    if (
        "ArrayFieldEncodingStrategy" in production
        and not relative.is_relative_to("rust/lance-encoding")
        and not relative.is_relative_to("rust/lance-file/src/versions/v2_0")
    ):
        errors.append(
            f"{relative}: pb::ArrayEncoding field composition escaped the v2.0 "
            "file-version owner"
        )

    for name in FORBIDDEN_COMPRESSION_SELECTION_ABSTRACTIONS:
        if re.search(rf"\b{re.escape(name)}\b", production):
            errors.append(
                f"{relative}: redundant compression selection abstraction remains: {name}; "
                "exact versions must implement CompressionStrategy directly"
            )

    for name in FORBIDDEN_FIELD_SELECTION_ABSTRACTIONS:
        if re.search(rf"\b{re.escape(name)}\b", production):
            errors.append(
                f"{relative}: redundant field selection abstraction remains: {name}; "
                "exact versions must implement FieldEncodingStrategy directly"
            )

    for name in FORBIDDEN_VERSION_POLICY_PROXIES:
        if re.search(rf"\b{re.escape(name)}\b", production):
            errors.append(
                f"{relative}: version policy proxy remains: {name}; "
                "dispatch the complete operation at the exact-version boundary"
            )

    if relative.parts[:3] == ("rust", "lance-encoding", "src"):
        for name in FORBIDDEN_ENCODING_PROFILE_TYPES:
            if re.search(rf"\b{re.escape(name)}\b", production):
                errors.append(
                    f"{relative}: encoding capability/profile type remains: {name}; "
                    "compose exact executable strategies instead"
                )

    for struct_name, allowed_fields in ENCODING_COMPOSITION_CONTAINERS.get(
        relative, {}
    ).items():
        fields = struct_fields(production, struct_name)
        if fields and fields != allowed_fields:
            errors.append(
                f"{relative}: {struct_name} must contain only "
                f"{', '.join(sorted(allowed_fields))}; found "
                f"{', '.join(sorted(fields))}"
            )

    dataset_versions = Path("rust/lance/src/dataset/versions")
    if relative.parent == dataset_versions and relative.name != "mod.rs":
        errors.append(
            f"{relative}: dataset per-version mirror modules are forbidden; "
            "dispatch each real operation-level difference from versions/mod.rs"
        )

    if relative.is_relative_to("rust/lance-file/src/previous"):
        errors.append(
            f"{relative}: the legacy file implementation must live under versions/v1"
        )

    if re.search(r"\blance_file::previous\b", production) or (
        relative.is_relative_to("rust/lance-file/src")
        and re.search(
            r"\b(?:pub\s+)?mod\s+previous\b|\b(?:crate|super)::previous\b",
            production,
        )
    ):
        errors.append(
            f"{relative}: legacy lance-file facade remains; use versions::v1 directly"
        )

    if relative.parts[:3] == ("rust", "lance-encoding", "src"):
        for version_type in ("LanceFileVersion", "ConcreteFileVersion"):
            if version_type in production:
                errors.append(
                    f"{relative}: lance-encoding production code imports {version_type}"
                )

    if relative == Path("rust/lance-file/src/version.rs"):
        for enum_name in ("LanceFileVersion", "ConcreteFileVersion"):
            forbidden = enum_derives(production, enum_name) & {"Ord", "PartialOrd"}
            if forbidden:
                errors.append(
                    f"{relative}: {enum_name} derives forbidden ordering trait(s): "
                    + ", ".join(sorted(forbidden))
                )

    if (
        "LanceFileVersion" in production
        and relative not in VERSION_SELECTOR_ALLOWLIST
        and not relative.is_relative_to("rust/lance-file/src/versions")
    ):
        errors.append(
            f"{relative}: public selector leaked past an API or release-policy boundary"
        )

    if "ConcreteFileVersion" in production and not exact_identity_allowed(relative):
        errors.append(
            f"{relative}: exact file identity leaked past a persisted or dispatch boundary"
        )

    comparison_allowed = (
        relative in EXACT_COMPARISON_ALLOWLIST
        or relative.is_relative_to("rust/lance-file/src/versions")
        or relative.is_relative_to("rust/lance/src/dataset/versions")
    )
    if not comparison_allowed:
        for line, expression in comparison_violations(production):
            errors.append(
                f"{relative}:{line}: file-version comparison outside a dispatch boundary: "
                f"{expression}"
            )

    for name in LEGACY_READER_ESCAPE_HATCHES:
        if name in production:
            errors.append(
                f"{relative}: legacy GenericFileReader escape hatch remains: {name}"
            )

    for name in PREVIOUS_FILE_IDENTITY_ALIASES:
        if re.search(rf"\b{re.escape(name)}\b", production):
            errors.append(
                f"{relative}: legacy file identity alias remains: {name}; use a v1 name"
            )

    if not (
        relative.is_relative_to("rust/lance-file/src/versions")
        or relative.is_relative_to("rust/lance/src/dataset/versions")
    ):
        for name in VERSION_CAPABILITY_ESCAPE_HATCHES:
            pattern = (
                rf"\b{re.escape(name)}\b"
                if name
                in {"BINARY_COPY_SUPPORTED", "PhysicalColumnLayout", "ScanStrategy"}
                else rf"\b{re.escape(name)}\s*\("
            )
            if re.search(pattern, production):
                errors.append(
                    f"{relative}: file-version capability escape hatch remains: {name}"
                )

    if current_version := exact_version_module(relative):
        for line, referenced_version in sibling_version_references(
            production, current_version
        ):
            errors.append(
                f"{relative}:{line}: {current_version} implementation references "
                f"sibling version module {referenced_version}"
            )

        if current_version != "v1":
            for reader_type in ("Reader", "ProjectedReader"):
                if re.search(
                    rf"\b(?:pub(?:\s*\([^)]*\))?\s+)?struct\s+{reader_type}\b",
                    production,
                ):
                    errors.append(
                        f"{relative}: exact {current_version} reader wrapper "
                        f"{reader_type} is forbidden; return the shared runtime "
                        "after exact grammar selection"
                    )

        for function_name, parameters, line in public_function_parameters(production):
            if "FieldEncodingStrategy" in parameters:
                errors.append(
                    f"{relative}:{line}: public exact-version API {function_name} "
                    "accepts a complete FieldEncodingStrategy override"
                )
            if function_name.startswith(
                ("create_writer", "create_lazy_writer", "try_new", "new_lazy")
            ):
                for forbidden in FORBIDDEN_EXACT_CONSTRUCTOR_INPUTS:
                    if re.search(rf"\b{re.escape(forbidden)}\b", parameters):
                        errors.append(
                            f"{relative}:{line}: exact-version constructor "
                            f"{function_name} accepts forbidden format input {forbidden}"
                        )

    shared_writer_structs: tuple[str, ...] = ()
    if relative == Path("rust/lance-file/src/writer.rs"):
        shared_writer_structs = ("FileWriterOptions", "FileWriter")
    elif relative.is_relative_to("rust/lance-file/src/writer"):
        shared_writer_structs = (
            "StructuralFileSink",
            "EncodingPipeline",
            "EncodedBatchBody",
        )
    if shared_writer_structs:
        for struct_name in shared_writer_structs:
            body = struct_body(production, struct_name)
            for field in FORBIDDEN_SHARED_WRITER_PROFILE_FIELDS:
                if re.search(rf"\b{re.escape(field)}\s*:", body):
                    errors.append(
                        f"{relative}: shared {struct_name} carries format-profile "
                        f"field {field}"
                    )
        for function_name, parameters, line in public_function_parameters(production):
            if "FieldEncodingStrategy" in parameters:
                errors.append(
                    f"{relative}:{line}: shared writer API {function_name} accepts "
                    "a complete FieldEncodingStrategy override"
                )
            if relative.is_relative_to("rust/lance-file/src/writer"):
                for field in FORBIDDEN_SHARED_WRITER_PROFILE_FIELDS:
                    if re.search(rf"\b{re.escape(field)}\b", parameters):
                        errors.append(
                            f"{relative}:{line}: shared writer mechanism "
                            f"{function_name} accepts format-profile input {field}"
                        )

    if relative == Path("rust/lance-file/src/reader.rs"):
        for reader_type in ("FileReader", "ProjectedFileReader"):
            if re.search(rf"\bpub\s+enum\s+{reader_type}\b", production):
                errors.append(
                    f"{relative}: shared {reader_type} must not dispatch every "
                    "runtime method through exact-version variants"
                )
        body = struct_body(production, "DecodeEngine")
        for field in FORBIDDEN_SHARED_READER_PROFILE_FIELDS:
            if re.search(rf"\b{re.escape(field)}\s*:", body):
                errors.append(
                    f"{relative}: shared DecodeEngine carries format-profile field {field}"
                )
        if body and not re.search(
            r"\bread_projection\s*:\s*Arc\s*<\s*dyn\s+ReadProjection\s*>",
            body,
        ):
            errors.append(
                f"{relative}: shared DecodeEngine must carry only the executable "
                "ReadProjection selected at the exact-version boundary"
            )
    elif relative.is_relative_to("rust/lance-file/src/reader"):
        for function_name, parameters, line in public_function_parameters(production):
            for field in FORBIDDEN_SHARED_READER_PROFILE_FIELDS:
                if re.search(rf"\b{re.escape(field)}\b", parameters):
                    errors.append(
                        f"{relative}:{line}: shared reader mechanism "
                        f"{function_name} accepts format-profile input {field}"
                    )

    if relative == Path("rust/lance-encoding/src/encoder.rs"):
        body = struct_body(production, "EncodingOptions")
        if re.search(r"\bversion\s*:", body):
            errors.append(f"{relative}: EncodingOptions still carries a file version")

    if relative == Path("rust/lance/src/dataset/write.rs"):
        body = struct_body(production, "WriterGenerator")
        if re.search(r"\b(?:ConcreteFileVersion|LanceFileVersion)\b", body):
            errors.append(
                f"{relative}: shared WriterGenerator stores file identity instead of "
                "an already selected writer factory"
            )
        for function_name, parameters, line in public_function_parameters(production):
            if function_name == "do_write_fragments_impl" and re.search(
                r"\b(?:ConcreteFileVersion|LanceFileVersion)\b", parameters
            ):
                errors.append(
                    f"{relative}:{line}: shared write loop accepts file identity instead "
                    "of already selected batching and writer behavior"
                )

    return errors


def run_checks() -> list[str]:
    errors = []
    for path in sorted((REPO_ROOT / "rust").rglob("*.rs")):
        errors.extend(check_source(path, path.read_text(encoding="utf-8")))
    return errors


def run_self_test() -> None:
    illegal_comparison = """
use lance_file::version::ConcreteFileVersion;
fn leaked(version: ConcreteFileVersion) {
    if version >= ConcreteFileVersion::V2_2 {}
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), illegal_comparison)
    if not any("comparison outside a dispatch boundary" in error for error in errors):
        raise AssertionError("illegal file-version comparison was not rejected")

    variable_comparison = """
use lance_file::version::ConcreteFileVersion;
fn leaked(left: ConcreteFileVersion, right: ConcreteFileVersion) {
    if left != right {}
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), variable_comparison)
    if not any("comparison outside a dispatch boundary" in error for error in errors):
        raise AssertionError(
            "variable-to-variable file-version comparison was not rejected"
        )

    inferred_variable_comparison = """
fn leaked(manifest: &Manifest) {
    let actual = determine_file_version();
    let expected = manifest.data_storage_format.lance_file_format();
    if actual != expected {}
}
"""
    errors = check_source(
        Path("rust/example/src/leak.rs"), inferred_variable_comparison
    )
    if not any("comparison outside a dispatch boundary" in error for error in errors):
        raise AssertionError("inferred file-version comparison was not rejected")

    unrelated_resolve = """
fn unrelated(segment_selection: SegmentSelection) {
    let segments = segment_selection.resolve();
    match segments {
        Some(segments) => consume(segments),
        None => {}
    }
}
"""
    errors = check_source(Path("rust/example/src/fts.rs"), unrelated_resolve)
    if any("comparison outside a dispatch boundary" in error for error in errors):
        raise AssertionError("unrelated resolve result was treated as a file version")

    typed_identity = """
use lance_file::version::ConcreteFileVersion;
fn leaked(_version: ConcreteFileVersion) {}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), typed_identity)
    if not any("exact file identity leaked" in error for error in errors):
        raise AssertionError("deep exact file identity was not rejected")

    aliased_identity = """
use lance_file::version::ConcreteFileVersion as FileVersion;
fn leaked(_version: FileVersion) {}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), aliased_identity)
    if not any("exact file identity leaked" in error for error in errors):
        raise AssertionError("aliased exact file identity was not rejected")

    exhaustive_match = """
use lance_file::version::ConcreteFileVersion;
fn leaked(version: ConcreteFileVersion) {
    match version {
        ConcreteFileVersion::V1 => {}
        _ => {}
    }
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), exhaustive_match)
    if not any("comparison outside a dispatch boundary" in error for error in errors):
        raise AssertionError("deep exact-version match was not rejected")

    illegal_selector = """
use lance_file::version::LanceFileVersion;
fn leaked(_version: LanceFileVersion) {}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), illegal_selector)
    if not any("public selector leaked" in error for error in errors):
        raise AssertionError("deep public selector use was not rejected")

    test_only_selector = """
#[cfg(test)]
mod tests {
    use lance_file::version::LanceFileVersion;
}
"""
    if check_source(Path("rust/example/src/lib.rs"), test_only_selector):
        raise AssertionError("test-only selector use was rejected")

    encoding_profile = """
pub struct DefaultCompressionStrategy {
    block_compression: BlockCompression,
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/compression.rs"), encoding_profile
    )
    if not any("encoding capability/profile type remains" in error for error in errors):
        raise AssertionError("encoding capability profile was not rejected")

    test_version_mirror = """
enum TestFileVersion {
    V2_1,
    V2_2,
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/testing.rs"), test_version_mirror
    )
    if not any("mirrors file identity" in error for error in errors):
        raise AssertionError("encoding test file-version mirror was not rejected")

    ordered_test_encoding = """
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum TestEncoding {
    Legacy,
    Structural,
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/testing.rs"), ordered_test_encoding
    )
    if not any("forbidden ordering trait" in error for error in errors):
        raise AssertionError("ordered semantic test encoding was not rejected")

    ordered_test_gate = """
fn with_min_file_version(version: TestEncoding) {}
"""
    errors = check_source(Path("rust/lance-encoding/src/testing.rs"), ordered_test_gate)
    if not any("ordered encoding test gate" in error for error in errors):
        raise AssertionError("ordered encoding test API was not rejected")

    redundant_field_selection = """
pub trait FieldEncodingAtom {
    fn try_create_field_encoder(&self);
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/encoder/structural.rs"),
        redundant_field_selection,
    )
    if not any("redundant field selection abstraction" in error for error in errors):
        raise AssertionError("redundant field selection abstraction was not rejected")

    primitive_capability_bag = """
struct PrimitivePageEncoding {
    behavior: Arc<dyn PrimitivePageEncodingBehavior>,
    supports_constant: bool,
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/encodings/logical/primitive.rs"),
        primitive_capability_bag,
    )
    if not any("must contain only behavior" in error for error in errors):
        raise AssertionError("primitive encoding capability bag was not rejected")

    redundant_compression_atom = """
pub trait CompressionAtom {
    fn try_create_block(&self);
}
"""
    errors = check_source(
        Path("rust/lance-encoding/src/compression.rs"),
        redundant_compression_atom,
    )
    if not any(
        "redundant compression selection abstraction" in error for error in errors
    ):
        raise AssertionError("redundant compression atom was not rejected")

    exact_compression_capability_bag = """
struct Strategy {
    params: CompressionParams,
    supports_block: bool,
}
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_2/compression.rs"),
        exact_compression_capability_bag,
    )
    if not any("must contain only params" in error for error in errors):
        raise AssertionError("exact compression capability bag was not rejected")

    legacy_escape_hatch = """
fn leaked(reader: &GenericFileReader) {
    reader.as_legacy_opt();
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), legacy_escape_hatch)
    if not any("legacy GenericFileReader escape hatch" in error for error in errors):
        raise AssertionError("legacy reader escape hatch was not rejected")

    capability_lookup = """
fn leaked(version: Version) {
    let strategy = versions::scan_strategy(version);
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), capability_lookup)
    if not any("file-version capability escape hatch" in error for error in errors):
        raise AssertionError("deep version capability lookup was not rejected")

    indexed_metadata_capability = """
fn leaked(version: Version) {
    if versions::supports_indexed_metadata(version) {}
}
"""
    errors = check_source(Path("rust/example/src/leak.rs"), indexed_metadata_capability)
    if not any("file-version capability escape hatch" in error for error in errors):
        raise AssertionError("indexed metadata capability was not rejected")

    write_policy_proxy = """
enum WriteBatching {
    RowGroup,
    File,
}
"""
    errors = check_source(
        Path("rust/lance/src/dataset/versions/mod.rs"), write_policy_proxy
    )
    if not any("version policy proxy remains" in error for error in errors):
        raise AssertionError("dataset write policy proxy was not rejected")

    legal_persisted_validation = """
use lance_file::version::ConcreteFileVersion;
fn validate(actual: ConcreteFileVersion, expected: ConcreteFileVersion) -> bool {
    actual == expected
}
"""
    if check_source(
        Path("rust/lance-table/src/format/fragment.rs"), legal_persisted_validation
    ):
        raise AssertionError("persisted identity validation was rejected")

    shared_writer_identity = """
use lance_file::version::ConcreteFileVersion;
struct FileWriter {
    version: ConcreteFileVersion,
}
"""
    errors = check_source(Path("rust/lance-file/src/writer.rs"), shared_writer_identity)
    if not any("exact file identity leaked" in error for error in errors):
        raise AssertionError("shared writer exact identity was not rejected")

    shared_writer_generator_identity = """
use lance_file::version::ConcreteFileVersion;
struct WriterGenerator {
    version: ConcreteFileVersion,
}
"""
    errors = check_source(
        Path("rust/lance/src/dataset/write.rs"), shared_writer_generator_identity
    )
    if not any(
        "shared WriterGenerator stores file identity" in error for error in errors
    ):
        raise AssertionError("shared writer generator exact identity was not rejected")

    shared_write_loop_identity = """
use lance_file::version::ConcreteFileVersion;
pub(super) async fn do_write_fragments_impl(
    version: ConcreteFileVersion,
) {}
"""
    errors = check_source(
        Path("rust/lance/src/dataset/write.rs"), shared_write_loop_identity
    )
    if not any("shared write loop accepts file identity" in error for error in errors):
        raise AssertionError("shared write loop exact identity was not rejected")

    sibling_import = """
use crate::versions::v2_2::reader::Reader;
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_3/writer.rs"), sibling_import
    )
    if not any("references sibling version module" in error for error in errors):
        raise AssertionError(
            "sibling exact-version implementation import was not rejected"
        )

    sibling_reexport = """
pub use super::v2_1::Writer;
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_2/mod.rs"), sibling_reexport
    )
    if not any("references sibling version module" in error for error in errors):
        raise AssertionError(
            "sibling exact-version implementation re-export was not rejected"
        )

    shared_writer_profile = """
struct FileWriterOptions {
    encoding_strategy: Arc<dyn FieldEncodingStrategy>,
}
"""
    errors = check_source(Path("rust/lance-file/src/writer.rs"), shared_writer_profile)
    if not any("carries format-profile field" in error for error in errors):
        raise AssertionError("shared writer format profile was not rejected")

    shared_writer_mechanism_profile = """
struct StructuralFileSink {
    minor_version: u16,
}
"""
    errors = check_source(
        Path("rust/lance-file/src/writer/structural.rs"),
        shared_writer_mechanism_profile,
    )
    if not any("carries format-profile field" in error for error in errors):
        raise AssertionError("shared writer mechanism format profile was not rejected")

    shared_writer_mechanism_input = """
pub fn finish(minor_version: u16) {}
"""
    errors = check_source(
        Path("rust/lance-file/src/writer/structural.rs"),
        shared_writer_mechanism_input,
    )
    if not any("accepts format-profile input" in error for error in errors):
        raise AssertionError("shared writer mechanism profile input was not rejected")

    shared_reader_profile = """
struct DecodeEngine {
    page_encoding: PageEncoding,
}
"""
    errors = check_source(Path("rust/lance-file/src/reader.rs"), shared_reader_profile)
    if not any("carries format-profile field" in error for error in errors):
        raise AssertionError("shared reader format profile was not rejected")

    shared_reader_mechanism_input = """
pub fn parse(accepted_grammar: AcceptedGrammar) {}
"""
    errors = check_source(
        Path("rust/lance-file/src/reader/structural.rs"),
        shared_reader_mechanism_input,
    )
    if not any("accepts format-profile input" in error for error in errors):
        raise AssertionError("shared reader mechanism profile input was not rejected")

    exact_reader_wrapper = """
pub struct Reader {
    state: FullReaderState,
}
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_2/reader.rs"),
        exact_reader_wrapper,
    )
    if not any("exact v2_2 reader wrapper" in error for error in errors):
        raise AssertionError("exact reader runtime wrapper was not rejected")

    shared_reader_dispatch = """
pub enum FileReader {
    V2_1(V21Reader),
    V2_2(V22Reader),
}
"""
    errors = check_source(
        Path("rust/lance-file/src/reader.rs"),
        shared_reader_dispatch,
    )
    if not any("must not dispatch every runtime method" in error for error in errors):
        raise AssertionError("shared per-version reader dispatch was not rejected")

    executable_reader_atom = """
struct DecodeEngine {
    read_projection: Arc<dyn ReadProjection>,
}
"""
    if check_source(Path("rust/lance-file/src/reader.rs"), executable_reader_atom):
        raise AssertionError("executable reader projection atom was rejected")

    complete_strategy_override = """
pub fn create_writer(
    strategy: Arc<dyn FieldEncodingStrategy>,
) -> Writer {
    todo!()
}
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_1/mod.rs"),
        complete_strategy_override,
    )
    if not any("complete FieldEncodingStrategy override" in error for error in errors):
        raise AssertionError(
            "complete exact-version strategy override was not rejected"
        )

    constructor_format_input = """
pub fn new_lazy(
    version: ConcreteFileVersion,
    options: FileWriterOptions,
) -> Writer {
    todo!()
}
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_1/writer.rs"),
        constructor_format_input,
    )
    if not any("constructor new_lazy accepts forbidden" in error for error in errors):
        raise AssertionError("exact-version constructor format input was not rejected")

    dataset_version_mirror = """
pub fn scan() {}
"""
    errors = check_source(
        Path("rust/lance/src/dataset/versions/v2_2.rs"), dataset_version_mirror
    )
    if not any("dataset per-version mirror modules" in error for error in errors):
        raise AssertionError("dataset per-version mirror module was not rejected")

    legacy_file_facade = """
pub mod previous;
"""
    errors = check_source(Path("rust/lance-file/src/lib.rs"), legacy_file_facade)
    if not any("legacy lance-file facade remains" in error for error in errors):
        raise AssertionError("legacy lance-file facade was not rejected")

    previous_file_identity = """
use lance_file::versions::v1::reader::FileReader as PreviousFileReader;
"""
    errors = check_source(Path("rust/example/src/leak.rs"), previous_file_identity)
    if not any("legacy file identity alias remains" in error for error in errors):
        raise AssertionError("legacy file identity alias was not rejected")

    lance_io_encoding_facade = """
use lance_io::encodings::plain::PlainEncoder;
"""
    errors = check_source(Path("rust/example/src/leak.rs"), lance_io_encoding_facade)
    if not any("v1 encoding grammar leaked into lance-io" in error for error in errors):
        raise AssertionError("lance-io v1 encoding facade was not rejected")

    previous_encoding_facade = """
pub mod previous;
"""
    errors = check_source(
        Path("rust/lance-encoding/src/lib.rs"), previous_encoding_facade
    )
    if not any("time-relative lance-encoding facade" in error for error in errors):
        raise AssertionError("lance-encoding previous facade was not rejected")

    array_strategy_leak = """
use lance_encoding::array_encoding::ArrayFieldEncodingStrategy;
fn strategy() {
    let _ = ArrayFieldEncodingStrategy::new();
}
"""
    errors = check_source(
        Path("rust/lance-file/src/versions/v2_1/writer.rs"), array_strategy_leak
    )
    if not any("field composition escaped the v2.0" in error for error in errors):
        raise AssertionError("v2.0 array field composition leak was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify that representative architecture violations are rejected",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        print("File-version boundary self-test passed.")
        return 0

    errors = run_checks()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("File-version boundaries are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
