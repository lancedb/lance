#!/usr/bin/env python3
"""
Interactive SQL Interface for Lance Datasets

Provides a command-line SQL REPL for querying Lance datasets with DuckDB.
Designed for local filesystem Lance datasets.
"""

import sys
from pathlib import Path
import readline


import duckdb
import lance



class InteractiveSQLInterface:
    """Interactive SQL interface for Lance datasets"""

    def __init__(self, dataset_path: str = "./lance_data"):
        self.dataset_path = Path(dataset_path)
        self.conn = duckdb.connect(":memory:")
        self.loaded_tables = set()
        self._load_datasets()
        self._setup_history()

    def _load_datasets(self):
        """Automatically discover and load Lance datasets from the specified directory"""
        if not self.dataset_path.exists():
            print(f"⚠️ Dataset directory '{self.dataset_path}' does not exist")
            print(f"Create it and add Lance datasets, or specify a different path")
            return

        print(f"🔄 Discovering Lance datasets in '{self.dataset_path}'...")

        # Look for Lance datasets (directories with .lance files or _versions directory)
        for item in self.dataset_path.iterdir():
            if item.is_dir():
                # Check if it's a Lance dataset
                if self._is_lance_dataset(item):
                    self._load_lance_dataset(item)

        if not self.loaded_tables:
            print(f"❌ No Lance datasets found in '{self.dataset_path}'")
            print("Ensure your Lance datasets are in the specified directory")

    def _is_lance_dataset(self, path: Path) -> bool:
        """Check if a directory contains a Lance dataset"""
        # Lance datasets have a _versions directory
        return (path / "_versions").exists()

    def _load_lance_dataset(self, dataset_path: Path):
        """Load a Lance dataset and register it with DuckDB"""
        table_name = dataset_path.name

        try:
            dataset = lance.dataset(str(dataset_path))
            self.conn.register(table_name, dataset.to_table())
            self.loaded_tables.add(table_name)
            print(f"✅ Loaded '{table_name}' table: {dataset.count_rows():,} rows")

        except Exception as e:
            print(f"❌ Could not load dataset '{table_name}': {e}")

    def _setup_history(self):
        """Setup command history file"""
        try:
            history_file = Path.home() / ".lance_sql_history"
            if history_file.exists():
                readline.read_history_file(str(history_file))
            self.history_file = history_file
        except Exception:
            self.history_file = None

    def load_dataset(self, dataset_path: str, table_name: str = None):
        """Load a specific Lance dataset"""
        path = Path(dataset_path)
        if not path.exists():
            print(f"❌ Dataset path '{dataset_path}' does not exist")
            return

        if table_name is None:
            table_name = path.name

        try:
            dataset = lance.dataset(str(path))
            self.conn.register(table_name, dataset.to_table())
            self.loaded_tables.add(table_name)
            print(f"✅ Loaded '{table_name}' table: {dataset.count_rows():,} rows")

        except Exception as e:
            print(f"❌ Could not load dataset '{table_name}': {e}")

    def show_tables(self):
        """Show available tables"""
        if not self.loaded_tables:
            print("📋 No tables loaded")
            return

        print("\n📋 Available Tables:")
        for table in sorted(self.loaded_tables):
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                count = result[0] if result else "Unknown"
                print(f"  • {table:<20} {count:>10,} rows")
            except Exception as e:
                print(f"  • {table:<20} {'Error':>10} - {e}")

    def describe_table(self, table_name: str):
        """Describe table schema"""
        if table_name not in self.loaded_tables:
            print(
                f"❌ Table '{table_name}' not loaded. Available: {', '.join(sorted(self.loaded_tables))}"
            )
            return

        try:
            # Get schema information
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            print(f"\n📋 Schema for '{table_name}':")
            print(result.to_string(index=False))

            # Get sample data
            print(f"\n🔍 Sample data from '{table_name}':")
            sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()

            # Truncate long columns for display
            for col in sample.columns:
                if sample[col].dtype == "object":
                    sample[col] = sample[col].astype(str).str[:50] + "..."

            print(sample.to_string(index=False))

        except Exception as e:
            print(f"❌ Error describing table: {e}")

    def execute_query(self, sql: str) -> bool:
        """Execute SQL query and display results"""
        if not sql.strip():
            return True

        try:
            # Execute query
            result = self.conn.execute(sql).fetchdf()

            # Display results
            if len(result) == 0:
                print("📊 Query returned no rows")
            else:
                print(f"📊 Query returned {len(result):,} rows")

                # Truncate display for large results
                if len(result) > 20:
                    print("Showing first 20 rows (use LIMIT for more control):")
                    display_df = result.head(20)
                else:
                    display_df = result

                # Truncate long string columns
                for col in display_df.columns:
                    if display_df[col].dtype == "object":
                        display_df[col] = display_df[col].astype(str).str[:80]

                print(display_df.to_string(index=False))

            return True

        except Exception as e:
            print(f"❌ SQL Error: {e}")
            return True

    def show_help(self):
        """Show help information"""
        print(
            """
🚀 Interactive SQL Interface for Lance Datasets

📋 Available Commands:
  .help                     - Show this help
  .tables                   - List available tables
  .describe <table>         - Show table schema and sample data
  .load <path> [table_name] - Load a Lance dataset from path
  .quit or .exit            - Exit the interface

🔍 Example Queries:
  SELECT COUNT(*) FROM my_dataset;
  SELECT * FROM my_dataset LIMIT 10;
  SELECT column1, COUNT(*) FROM my_dataset GROUP BY column1;

  -- Filter data
  SELECT * FROM my_dataset WHERE column1 > 100;

  -- Join tables (if multiple datasets loaded)
  SELECT a.*, b.column2
  FROM dataset_a a
  JOIN dataset_b b ON a.id = b.id;

💡 Tips:
  • Use LIMIT to control result size
  • Press Tab for SQL keyword completion
  • Use arrow keys to navigate command history
  • Long strings are truncated for display
  • Place Lance datasets in the data directory or use .load command
        """
        )

    def save_history(self):
        """Save command history"""
        if self.history_file:
            try:
                readline.write_history_file(str(self.history_file))
            except Exception:
                pass

    def run(self):
        """Run the interactive SQL interface"""
        print(
            f"""
🚀 Lance Dataset - Interactive SQL Interface
==========================================
Dataset directory: {self.dataset_path}
Type .help for commands, .quit to exit
        """
        )

        self.show_tables()

        try:
            while True:
                try:
                    # Get user input
                    query = input("\nlance_sql> ").strip()

                    if not query:
                        continue

                    # Handle special commands
                    if query.lower() in [".quit", ".exit"]:
                        break
                    elif query.lower() == ".help":
                        self.show_help()
                    elif query.lower() == ".tables":
                        self.show_tables()
                    elif query.lower().startswith(".describe "):
                        table_name = query.split(" ", 1)[1].strip()
                        self.describe_table(table_name)
                    elif query.lower().startswith(".load "):
                        parts = query.split(" ", 2)[1:]
                        if len(parts) == 1:
                            self.load_dataset(parts[0])
                        elif len(parts) == 2:
                            self.load_dataset(parts[0], parts[1])
                        else:
                            print("Usage: .load <path> [table_name]")
                    else:
                        # Execute SQL query
                        self.execute_query(query)

                except KeyboardInterrupt:
                    print("\n\nUse .quit to exit")
                    continue
                except EOFError:
                    break

        finally:
            self.save_history()
            print("\n👋 Goodbye!")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive SQL interface for Lance datasets"
    )
    parser.add_argument(
        "--data-dir",
        default="./lance_data",
        help="Directory containing Lance datasets (default: ./lance_data)"
    )

    args = parser.parse_args()

    interface = InteractiveSQLInterface(args.data_dir)
    interface.run()


if __name__ == "__main__":
    main()