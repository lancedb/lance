"""CLI for lance-memtest."""

import sys
import memtest


def main():
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] == "path":
        lib_path = memtest.get_library_path()
        if lib_path is None:
            print(
                "lance-memtest is not supported on this platform",
                file=sys.stderr,
            )
            return 1
        print(lib_path)
        return 0
    if args[0] == "stats":
        memtest.print_stats()
        return 0
    if args[0] == "reset":
        memtest.reset_stats()
        return 0
    else:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        print("Usage: lance-memtest [path|stats|reset]", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
