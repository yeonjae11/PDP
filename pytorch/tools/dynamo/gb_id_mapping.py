# mypy: ignore-errors

import argparse
import ast
import json
import re
import sys
from pathlib import Path


def get_source_segment(source, node):
    return ast.get_source_segment(source, node)


def load_registry(path):
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def save_registry(reg, path):
    with path.open("w") as f:
        json.dump(reg, f, indent=2)


def next_gb_id(reg):
    ids = [int(x[2:]) for x in reg if x.startswith("GB") and x[2:].isdigit()]
    return f"GB{(max(ids, default=0) + 1):04d}"


def clean_string(s):
    """
    Normalizes string literals by removing formatting artifacts and escape sequences.
    Handles f-strings, quotes, newlines, and other syntax elements for cleaner output.
    """
    if isinstance(s, str):
        # Convert f-string prefix to regular string prefix (e.g., f"hello" -> "hello")
        s = re.sub(r'^f["\']', r'"', s)
        # Replace quoted strings with f-prefix in the middle with a space (e.g., " f"" -> " ")
        s = re.sub(r'["\'] f["\']', " ", s)
        # Remove surrounding quotes, keeping only the content (e.g., "hello" -> hello)
        s = re.sub(r'^["\'](.*)["\']$', r"\1", s)
        # Replace any whitespace
        s = " ".join(s.splitlines())
        # Replace escaped quotes with their unescaped versions
        s = s.encode().decode("unicode_escape")
        # Replace adjacent quoted strings with a space (e.g., " "" -> " ")
        s = re.sub(r'" "', " ", s)
    return s


def expand_hints(hints):
    # Expands hint references to their actual values from graph_break_hints.
    from torch._dynamo import graph_break_hints

    hint_constants = {
        name: value
        for name, value in graph_break_hints.__dict__.items()
        if isinstance(value, list) and name.isupper()
    }

    expanded_hints = []
    for hint in hints:
        for name, value in hint_constants.items():
            if f"*graph_break_hints.{name}" in hint:
                expanded_hints.extend(value)
                break
    return expanded_hints


def extract_info_from_keyword(source, kw):
    """
    Extracts and returns the value of a keyword argument from an AST node.

    This function handles different types of AST nodes:
    - If the node is a constant, it returns the constant value.
    - If the node is an f-string, it reconstructs the string by
      evaluating formatted values and concatenating them with string literals.
    - For other types, it cleans the source segment to remove formatting artifacts.

    """
    param_source = get_source_segment(source, kw.value)
    if isinstance(kw.value, ast.Constant):
        return kw.value.value
    elif isinstance(kw.value, ast.JoinedStr):
        evaluated_context = []
        for value in kw.value.values:
            if isinstance(value, ast.FormattedValue):
                evaluated_context.append(f"{{{ast.unparse(value.value)}}}")
            elif isinstance(value, ast.Constant):
                evaluated_context.append(value.value)
        return "".join(evaluated_context)
    else:
        return clean_string(param_source)


def find_unimplemented_v2_calls(path):
    results = []
    path = Path(path)

    if path.is_dir():
        file_paths = path.glob("**/*.py")
    else:
        file_paths = [path]

    for file_path in file_paths:
        with open(file_path) as f:
            source = f.read()
            try:
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name == "unimplemented_v2":
                            continue
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "unimplemented_v2"
                    ):
                        info = {
                            "gb_type": None,
                            "context": None,
                            "explanation": None,
                            "hints": [],
                        }

                        for kw in node.keywords:
                            if kw.arg in info:
                                info[kw.arg] = extract_info_from_keyword(source, kw)

                        if info["gb_type"] is None:
                            continue

                        if info["hints"]:
                            hints = info["hints"]
                            expanded_hints = []
                            items = re.findall(r'"([^"]*)"', hints)
                            if items:
                                expanded_hints.extend(items)

                            if "*graph_break_hints." in hints:
                                expanded_hints.extend(expand_hints([hints]))

                            info["hints"] = expanded_hints

                        results.append(info)
            except SyntaxError:
                print(f"Syntax error in {file_path}")

    return results


def cmd_add_new_gb_type(gb_type, file_path, registry_path):
    """
    Add a new graph break type to the registry.

    Args:
        gb_type: The graph break type to add
        file_path: Path to the file containing the unimplemented_v2 call
        registry_path: Path to the registry JSON file
    """
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)

    existing_gb_types = {entry[0]["Gb_type"] for entry in reg.values()}
    if gb_type in existing_gb_types:
        print(
            f"Error: gb_type '{gb_type}' already exists in registry. Please rename the gb_type so it can be unique."
        )
        return False

    calls = find_unimplemented_v2_calls(Path(file_path))
    matching_call = next((call for call in calls if call["gb_type"] == gb_type), None)

    if not matching_call:
        print(
            f"Error: Could not find unimplemented_v2 call with gb_type '{gb_type}' in {file_path}"
        )
        return False

    gb_id = next_gb_id(reg)
    reg[gb_id] = [
        {
            "Gb_type": gb_type,
            "Context": matching_call["context"],
            "Explanation": matching_call["explanation"],
            "Hints": matching_call["hints"] or [],
        }
    ]

    save_registry(reg, registry_path)
    print(f"Added {gb_type} to registry with ID {gb_id}")
    return True


def cmd_update_gb_type(old_gb_type, file_path, registry_path, new_gb_type=None):
    """
    Update an existing graph break type in the registry by adding a new version
    to the version history list.

    Args:
        old_gb_type: The current graph break type to update
        file_path: Path to the file containing the updated unimplemented_v2 call
        registry_path: Path to the registry JSON file
        new_gb_type: Optional new gb_type name to replace the old one
    """
    registry_path = Path(registry_path)
    reg = load_registry(registry_path)

    gb_id_map = {entry[0]["Gb_type"]: id for id, entry in reg.items()}
    gb_id = gb_id_map.get(old_gb_type)

    if gb_id is None:
        print(f"Error: gb_type '{old_gb_type}' not found in registry.")
        return False

    search_gb_type = new_gb_type if new_gb_type else old_gb_type
    calls = find_unimplemented_v2_calls(Path(file_path))
    matching_call = next(
        (call for call in calls if call["gb_type"] == search_gb_type), None
    )

    if not matching_call:
        print(
            f"Error: Could not find unimplemented_v2 call with gb_type '{search_gb_type}' in {file_path}"
        )
        return False

    if (
        matching_call["gb_type"] != old_gb_type
        and matching_call["gb_type"] in gb_id_map
    ):
        print(
            f"Error: New gb_type '{matching_call['gb_type']}' already exists in registry. Please use a unique gb_type."
        )
        return False

    new_entry = {
        "Gb_type": matching_call["gb_type"],
        "Context": matching_call["context"],
        "Explanation": matching_call["explanation"],
        "Hints": matching_call["hints"] or [],
    }

    reg[gb_id].insert(0, new_entry)

    save_registry(reg, registry_path)
    print(
        f"Updated {old_gb_type} to {matching_call['gb_type']} in registry with ID {gb_id}"
    )
    return True


def create_registry(dynamo_dir, registry_path):
    calls = find_unimplemented_v2_calls(dynamo_dir)
    registry = {}

    gb_types = {}
    for info in calls:
        gb_types[info["gb_type"]] = info

    GB_ID_INDEX = 0000
    for i, (gb_type, info) in enumerate(sorted(gb_types.items()), GB_ID_INDEX):
        gb_id = f"GB{i:04d}"
        hints = info["hints"]

        registry[gb_id] = [
            {
                "Gb_type": gb_type,
                "Context": info["context"],
                "Explanation": info["explanation"],
                "Hints": hints if hints else [],
            }
        ]

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    registry_path = script_dir / "graph_break_registry.json"

    try:
        import torch._dynamo

        default_dynamo_dir = str(Path(torch._dynamo.__file__).parent)
    except ImportError:
        default_dynamo_dir = str(repo_root / "torch" / "_dynamo")

    parser = argparse.ArgumentParser(description="Manage graph break registry.")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    create_parser = subparsers.add_parser("create", help="Create registry from scratch")
    create_parser.add_argument(
        "--dynamo_dir",
        type=str,
        default=default_dynamo_dir,
        help="Directory to search for unimplemented_v2 calls.",
    )

    add_parser = subparsers.add_parser("add", help="Add a gb_type to registry")
    add_parser.add_argument("gb_type", help="The gb_type to add")
    add_parser.add_argument(
        "file_path", help="Path to the file containing the unimplemented_v2 call"
    )

    update_parser = subparsers.add_parser(
        "update", help="Update an existing gb_type in registry"
    )
    update_parser.add_argument("gb_type", help="The gb_type to update")
    update_parser.add_argument(
        "file_path",
        help="Path to the file containing the updated unimplemented_v2 call",
    )
    update_parser.add_argument(
        "--new_gb_type", help="New gb_type name if it has changed", default=None
    )

    parser.add_argument(
        "--registry-path",
        type=str,
        default=str(registry_path),
        help="Path to save the registry JSON file",
    )

    args = parser.parse_args()

    if args.command == "create":
        create_registry(args.dynamo_dir, args.registry_path)
    elif args.command == "add":
        success = cmd_add_new_gb_type(args.gb_type, args.file_path, args.registry_path)
        if not success:
            sys.exit(1)
    elif args.command == "update":
        success = cmd_update_gb_type(
            args.gb_type, args.file_path, args.registry_path, args.new_gb_type
        )
        if not success:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
