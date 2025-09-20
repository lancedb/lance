#!/usr/bin/env python3
"""
Script to update line numbers in operation documentation files.
This script parses the OpenAPI spec YAML file and automatically updates
the line number references in each operation's markdown file.
"""

import os
import re
import yaml
from pathlib import Path

# Operations that have REST NAMESPACE ONLY sections (will be calculated dynamically)
REST_OPERATIONS_TO_FIND = [
    'list-namespaces',
    'list-tables',
    'list-table-tags',
    'insert-into-table',
    'merge-insert-into-table',
    'create-table'
]

# Error response schemas to find (will be calculated dynamically)
ERROR_RESPONSE_SCHEMAS = [
    'ErrorResponse',
    'BadRequestErrorResponse',
    'UnauthorizedErrorResponse',
    'ForbiddenErrorResponse',
    'NotFoundErrorResponse',
    'UnsupportedOperationErrorResponse',
    'ConflictErrorResponse',
    'ServiceUnavailableErrorResponse',
    'ServerErrorResponse'
]

# Mapping from operation IDs to their request/response schema names
OPERATION_SCHEMAS = {
    'create-namespace': {
        'request': 'CreateNamespaceRequest',
        'response': 'CreateNamespaceResponse'
    },
    'list-namespaces': {
        'request': 'ListNamespacesRequest',
        'response': 'ListNamespacesResponse',
        'additional_schemas': {
            'PageToken': 'Page Token',
            'PageLimit': 'Page Limit',
        }
    },
    'describe-namespace': {
        'request': 'DescribeNamespaceRequest',
        'response': 'DescribeNamespaceResponse'
    },
    'drop-namespace': {
        'request': 'DropNamespaceRequest',
        'response': 'DropNamespaceResponse'
    },
    'namespace-exists': {
        'request': 'NamespaceExistsRequest',
        'response': None  # No response schema
    },
    'list-tables': {
        'request': 'ListTablesRequest',
        'response': 'ListTablesResponse',
        'additional_schemas': {
            'PageToken': 'Page Token',
            'PageLimit': 'Page Limit',
        }
    },
    'register-table': {
        'request': 'RegisterTableRequest',
        'response': 'RegisterTableResponse'
    },
    'describe-table': {
        'request': 'DescribeTableRequest',
        'response': 'DescribeTableResponse',
        'additional_schemas': {
            'JsonArrowSchema': 'Json Arrow Schema',
            'JsonArrowField': 'Json Arrow Schema',
        }
    },
    'table-exists': {
        'request': 'TableExistsRequest',
        'response': None  # No response schema
    },
    'drop-table': {
        'request': 'DropTableRequest',
        'response': 'DropTableResponse'
    },
    'deregister-table': {
        'request': 'DeregisterTableRequest',
        'response': 'DeregisterTableResponse'
    },
    'insert-into-table': {
        'request': 'InsertIntoTableRequest',
        'response': 'InsertIntoTableResponse'
    },
    'merge-insert-into-table': {
        'request': 'MergeInsertIntoTableRequest',
        'response': 'MergeInsertIntoTableResponse'
    },
    'update-table': {
        'request': 'UpdateTableRequest',
        'response': 'UpdateTableResponse'
    },
    'delete-from-table': {
        'request': 'DeleteFromTableRequest',
        'response': 'DeleteFromTableResponse'
    },
    'query-table': {
        'request': 'QueryTableRequest',
        'response': None  # No response schema (returns Arrow data)
    },
    'count-table-rows': {
        'request': 'CountTableRowsRequest',
        'response': 'CountTableRowsResponse'
    },
    'create-table': {
        'request': 'CreateTableRequest',
        'response': 'CreateTableResponse'
    },
    'create-empty-table': {
        'request': 'CreateEmptyTableRequest',
        'response': 'CreateEmptyTableResponse'
    },
    'create-table-index': {
        'request': 'CreateTableIndexRequest',
        'response': 'CreateTableIndexResponse'
    },
    'list-table-indices': {
        'request': 'ListTableIndicesRequest',
        'response': 'ListTableIndicesResponse'
    },
    'describe-table-index-stats': {
        'request': 'DescribeTableIndexStatsRequest',
        'response': 'DescribeTableIndexStatsResponse'
    },
    'describe-transaction': {
        'request': 'DescribeTransactionRequest',
        'response': 'DescribeTransactionResponse'
    },
    'alter-transaction': {
        'request': 'AlterTransactionRequest',
        'response': 'AlterTransactionResponse',
        'additional_schemas': {
            'AlterTransactionSetStatus': 'Set Status Action',
            'AlterTransactionSetProperty': 'Set Property Action', 
            'AlterTransactionUnsetProperty': 'Unset Property Action'
        }
    },
    # Tag operations
    'list-table-tags': {
        'request': None,  # GET operation
        'response': 'ListTableTagsResponse'
    },
    'get-table-tag-version': {
        'request': 'GetTableTagVersionRequest',
        'response': 'GetTableTagVersionResponse'
    },
    'create-table-tag': {
        'request': 'CreateTableTagRequest',
        'response': None  # No response schema
    },
    'delete-table-tag': {
        'request': 'DeleteTableTagRequest',
        'response': None  # No response schema
    },
    'update-table-tag': {
        'request': 'UpdateTableTagRequest',
        'response': None  # No response schema
    },
    # Table operations
    'restore-table': {
        'request': 'RestoreTableRequest',
        'response': 'RestoreTableResponse'
    },
    'list-table-versions': {
        'request': 'ListTableVersionsRequest',
        'response': 'ListTableVersionsResponse',
        'additional_schemas': {
            'TableVersion': 'Table Version'
        }
    },
    'explain-table-query-plan': {
        'request': 'ExplainTableQueryPlanRequest',
        'response': 'ExplainTableQueryPlanResponse'
    },
    'analyze-table-query-plan': {
        'request': 'AnalyzeTableQueryPlanRequest',
        'response': 'AnalyzeTableQueryPlanResponse'
    },
    'alter-table-add-columns': {
        'request': 'AlterTableAddColumnsRequest',
        'response': 'AlterTableAddColumnsResponse',
        'additional_schemas': {
            'NewColumnTransform': 'New Column Transform'
        }
    },
    'alter-table-alter-columns': {
        'request': 'AlterTableAlterColumnsRequest',
        'response': 'AlterTableAlterColumnsResponse',
        'additional_schemas': {
            'ColumnAlteration': 'Column Alteration'
        }
    },
    'alter-table-drop-columns': {
        'request': 'AlterTableDropColumnsRequest',
        'response': 'AlterTableDropColumnsResponse'
    },
    'get-table-stats': {
        'request': 'GetTableStatsRequest',
        'response': 'GetTableStatsResponse'
    },
    # Index operations
    'drop-table-index': {
        'request': 'DropTableIndexRequest',
        'response': 'DropTableIndexResponse'
    }
}

def find_rest_operation_lines(yaml_content, operation_id):
    """Find the start and end line numbers for a REST operation that contains 'REST NAMESPACE ONLY'."""
    lines = yaml_content.split('\n')
    
    # Convert kebab-case to PascalCase for operationId lookup
    pascal_case_id = ''.join(word.capitalize() for word in operation_id.split('-'))
    
    # Find the operation start by finding the operationId first
    operation_start = None
    for i, line in enumerate(lines):
        if f"operationId: {pascal_case_id}" in line:
            # Go back to find the path definition line (e.g., "  /v1/namespace/{id}/list:")
            for j in range(i, -1, -1):
                if re.match(r'^  /.*:$', lines[j]):
                    operation_start = j + 1  # Convert to 1-based indexing
                    break
            break
    
    if operation_start is None:
        return None, None
    
    # Find the operation end by looking for the next path definition
    operation_end = None
    for i in range(operation_start - 1 + 1, len(lines)):  # Start after the path line we found
        line = lines[i]
        # Look for next path definition (starts with two spaces and a slash)
        if re.match(r'^  /.*:$', line):
            operation_end = i  # End before this line
            break
    
    # If no next path found, look for major sections like components
    if operation_end is None:
        for i in range(operation_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('components:') or line.startswith('info:') or line.startswith('openapi:'):
                operation_end = i  # End before this line
                break
    
    # If still no end found, use end of file
    if operation_end is None:
        operation_end = len(lines)
    
    # Now we need to trim the end to exclude empty lines and comments at the end
    # Go backwards from operation_end to find the last meaningful line
    for i in range(operation_end - 1, operation_start - 1, -1):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            operation_end = i + 1  # Convert to 1-based and include this line
            break
    
    return operation_start, operation_end

def find_schema_lines(yaml_content, schema_name):
    """Find the start and end line numbers for a schema in the YAML content."""
    lines = yaml_content.split('\n')
    start_line = None
    end_line = None
    indent_level = None
    
    # Find the start line
    for i, line in enumerate(lines):
        if f"    {schema_name}:" in line:  # Schema under components.schemas
            start_line = i + 1  # Convert to 1-based indexing
            # Get the indentation level of this schema
            indent_level = len(line) - len(line.lstrip())
            break
    
    if start_line is None:
        return None, None
    
    # Find the end line by looking for the next schema at the same indent level
    for i in range(start_line, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line at the same or lesser indent that's not part of this schema
        if current_indent <= indent_level and i > start_line:
            # Check if it's another schema or a different section
            if line.strip().endswith(':') and not line.strip().startswith('-'):
                end_line = i  # End before this line
                break
    
    # If we didn't find an end, use the end of the file
    if end_line is None:
        end_line = len(lines)
    
    return start_line, end_line

def find_operation_description_lines(yaml_content, operation_id):
    """Find the line numbers for description of a given operation ID from the REST paths."""
    lines = yaml_content.split('\n')
    
    # Find the line with operationId
    for i, line in enumerate(lines):
        if f"operationId: {operation_id}" in line:
            # Look for description in the following lines
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip().startswith('description:'):
                    start_line = j + 1  # Convert to 1-based indexing
                    
                    # All descriptions now use multiline format with |
                    # Find end of description by looking for the next field at same indentation
                    end_line = start_line
                    for k in range(j + 1, len(lines)):
                        desc_line = lines[k]
                        # Stop when we hit the next field (requestBody, responses, etc.)
                        if (desc_line.strip().startswith('requestBody:') or 
                            desc_line.strip().startswith('responses:') or
                            desc_line.strip().startswith('summary:')):
                            end_line = k  # End before this line
                            break
                        # Stop if we encounter "REST NAMESPACE ONLY" keyword
                        elif desc_line.strip() == 'REST NAMESPACE ONLY':
                            end_line = k  # End before this line
                            break
                        elif desc_line.startswith('        ') and desc_line.strip():  # Description content line
                            end_line = k + 1  # Include this line
                    
                    return start_line, end_line
                        
                elif next_line.strip().startswith('summary:'):
                    # Found summary before description, no description available
                    break
    
    return None, None

def update_operation_file(operation_file, yaml_content, schemas, operation_id):
    """Update line numbers and description in an operation markdown file."""
    print(f"Updating {operation_file}...")
    
    with open(operation_file, 'r') as f:
        content = f.read()
    
    # Update description from REST route
    desc_start, desc_end = find_operation_description_lines(yaml_content, operation_id)
    if desc_start and desc_end:
        # Create line number reference for description
        desc_reference = f'```yaml\n--8<-- "src/rest.yaml:{desc_start}:{desc_end}"\n```'
        # Find and replace the description section
        # Look for the pattern: ## Description\n\n<existing description>
        desc_pattern = r'(## Description\s*\n\s*\n)(.*?)(\n\s*##|\n\s*$)'
        if re.search(desc_pattern, content, re.DOTALL):
            content = re.sub(desc_pattern, f'\\1{desc_reference}\\3', content, flags=re.DOTALL)
    
    # Update request schema - look for pattern in Request Schema section
    if schemas['request']:
        start, end = find_schema_lines(yaml_content, schemas['request'])
        if start and end:
            # Find and replace request schema line numbers in the Request Schema section
            pattern = r'(## Request Schema\s*\n\s*```yaml\s*\n--8<-- "src/rest\.yaml:)\d+:\d+(")'
            new_ref = f'\\g<1>{start}:{end}\\g<2>'
            content = re.sub(pattern, new_ref, content, flags=re.MULTILINE | re.DOTALL)
    
    # Update response schema - look for pattern in Response Schema section
    if schemas['response']:
        start, end = find_schema_lines(yaml_content, schemas['response'])
        if start and end:
            # Find and replace response schema line numbers in the Response Schema section
            pattern = r'(## Response Schema\s*\n\s*```yaml\s*\n--8<-- "src/rest\.yaml:)\d+:\d+(")'
            new_ref = f'\\g<1>{start}:{end}\\g<2>'
            content = re.sub(pattern, new_ref, content, flags=re.MULTILINE | re.DOTALL)
    
    # Handle additional schemas - append as new sections under "Related Components Schema"
    if 'additional_schemas' in schemas:
        # First, remove any existing "Related Components Schema" section
        related_schema_pattern = r'\n\n## Related Components Schema.*$'
        content = re.sub(related_schema_pattern, '', content, flags=re.DOTALL)
        
        additional_sections = []
        for schema_name, description in schemas['additional_schemas'].items():
            start, end = find_schema_lines(yaml_content, schema_name)
            if start and end:
                section = f"\n### {description}\n\n```yaml\n--8<-- \"src/rest.yaml:{start}:{end}\"\n```"
                additional_sections.append(section)
        
        if additional_sections:
            # Add sections with the main header before the final newline
            content = content.rstrip() + '\n\n## Related Components Schema' + ''.join(additional_sections) + '\n'
    
    # Write the updated content back
    with open(operation_file, 'w') as f:
        f.write(content)

def update_rest_operation_file(rest_file, yaml_content, operation_id):
    """Update line numbers in a REST implementation markdown file."""
    print(f"Updating REST file {rest_file}...")
    
    with open(rest_file, 'r') as f:
        content = f.read()
    
    # Get the REST operation line numbers dynamically
    start, end = find_rest_operation_lines(yaml_content, operation_id)
    if start and end:
        # Update the REST route definition line numbers
        pattern = r'(--8<-- "src/rest\.yaml:)\d+:\d+(")'
        new_ref = f'\\g<1>{start}:{end}\\g<2>'
        content = re.sub(pattern, new_ref, content)
    else:
        print(f"Warning: Could not find line numbers for operation {operation_id}")
    
    # Write the updated content back
    with open(rest_file, 'w') as f:
        f.write(content)

def update_operations_index_errors(index_file, yaml_content):
    """Update line numbers in the operations index file for error responses."""
    print(f"Updating error sections in {index_file}...")
    
    with open(index_file, 'r') as f:
        content = f.read()
    
    # Error response mappings with their corresponding patterns
    error_mappings = [
        ('ErrorResponse', r'(## Error Response Model.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('BadRequestErrorResponse', r'(### 400 - Bad Request Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('UnauthorizedErrorResponse', r'(### 401 - Unauthorized Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('ForbiddenErrorResponse', r'(### 403 - Forbidden Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('NotFoundErrorResponse', r'(### 404 - Not Found Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('UnsupportedOperationErrorResponse', r'(### 406 - Unsupported Operation Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('ConflictErrorResponse', r'(### 409 - Conflict Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('ServiceUnavailableErrorResponse', r'(### 503 - Service Unavailable Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")'),
        ('ServerErrorResponse', r'(### 5XX - Server Error Response.*?--8<-- "src/rest\.yaml:)\d+:\d+(")')
    ]
    
    # Update each error response dynamically
    for schema_name, pattern in error_mappings:
        start, end = find_schema_lines(yaml_content, schema_name)
        if start and end:
            content = re.sub(pattern, f'\\g<1>{start}:{end}\\g<2>', content, flags=re.DOTALL)
        else:
            print(f"Warning: Could not find line numbers for error schema {schema_name}")
    
    # Write the updated content back
    with open(index_file, 'w') as f:
        f.write(content)

def main():
    """Main function to update all operation files."""
    # Get the script directory (docs/src)
    script_dir = Path(__file__).parent
    
    # Path to the YAML spec file (same directory)
    yaml_file = script_dir / 'rest.yaml'
    # Path to operations directory (subdirectory)
    operations_dir = script_dir / 'operations'
    # Path to REST implementation directory
    rest_dir = script_dir / 'impls' / 'rest'
    
    if not yaml_file.exists():
        print(f"Error: {yaml_file} not found")
        return 1
    
    if not operations_dir.exists():
        print(f"Error: {operations_dir} not found")
        return 1
    
    # Read the YAML content
    with open(yaml_file, 'r') as f:
        yaml_content = f.read()
    
    # Update each operation file
    for operation_id, schemas in OPERATION_SCHEMAS.items():
        operation_file = operations_dir / f'{operation_id}.md'
        if operation_file.exists():
            # Convert kebab-case to PascalCase for operationId lookup
            pascal_case_id = ''.join(word.capitalize() for word in operation_id.split('-'))
            update_operation_file(operation_file, yaml_content, schemas, pascal_case_id)
        else:
            print(f"Warning: {operation_file} not found")
    
    # Update REST implementation files
    if rest_dir.exists():
        for operation_id in REST_OPERATIONS_TO_FIND:
            rest_file = rest_dir / f'{operation_id}.md'
            if rest_file.exists():
                update_rest_operation_file(rest_file, yaml_content, operation_id)
            else:
                print(f"Warning: REST file {rest_file} not found")
        
    # Update error sections in operations index file
    operations_index_file = operations_dir / 'index.md'
    if operations_index_file.exists():
        update_operations_index_errors(operations_index_file, yaml_content)
    else:
        print(f"Warning: {operations_index_file} not found")
    
    print("Line number update complete!")
    return 0

if __name__ == '__main__':
    exit(main())