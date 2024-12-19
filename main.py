import os
import sys
import glob
import json
import csv
import yaml
from typing import Any, Dict, List
'''
What this code does:

Reads cson_config.yaml (the CSON file).
Validates that inputs, transformations, and outputs exist.
Loads CSV files from ./input_data based on patterns in inputs.
Applies a simple filter (e.g., = $price > 20) to the products transformation.
Maps CSV columns to JSON fields as defined in item_structure.
Writes the resulting filtered JSON array to ./output_data/filtered_products.json.
If the CSON file is missing sections or is malformed, it prints errors and does not produce output. If an error occurs during loading inputs or transformations, it also prints the error and exits without writing outputs.

This provides a starting point you can run as-is and adapt or enhance as needed.
'''


def process_cson(cson_file_path: str,
                 input_directory: str,
                 output_directory: str,
                 parameters: Dict[str, Any] = None):
    # Load CSON config
    cson_config = load_cson_config(cson_file_path)
    if not cson_config:
        # Error already printed in load_cson_config
        sys.exit(1)

    # Validate CSON structure
    errors = validate_cson(cson_config)
    if errors:
        print("Errors in CSON configuration:", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        sys.exit(1)

    # Load CSV inputs
    try:
        data_store = load_inputs(cson_config, input_directory)
    except Exception as e:
        print(f"Error loading inputs: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply transformations
    try:
        transformed_data = apply_transformations(cson_config, data_store)
    except Exception as e:
        print(f"Error during transformations: {e}", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    try:
        write_outputs(cson_config, transformed_data, output_directory)
    except Exception as e:
        print(f"Error writing outputs: {e}", file=sys.stderr)
        sys.exit(1)


def load_cson_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load CSON file: {e}", file=sys.stderr)
        return None


def validate_cson(cson_config: Dict[str, Any]) -> List[str]:
    errors = []
    if "inputs" not in cson_config or not isinstance(cson_config["inputs"],
                                                     list):
        errors.append("Missing or invalid 'inputs' section.")
    if "transformations" not in cson_config or not isinstance(
            cson_config["transformations"], dict):
        errors.append("Missing or invalid 'transformations' section.")
    if "outputs" not in cson_config or not isinstance(cson_config["outputs"],
                                                      list):
        errors.append("Missing or invalid 'outputs' section.")

    # Check at least one transformation is properly defined
    # For simplicity, just ensure there's a dictionary with 'type' and 'from'
    for k, v in cson_config.get("transformations", {}).items():
        if isinstance(v, dict):
            if v.get("type") == "array" and "from" in v:
                # Seems okay
                break
    else:
        errors.append(
            "No valid array transformation found. Must have at least one {type: array, from: ...} definition."
        )

    return errors


def load_inputs(cson_config: Dict[str, Any],
                input_directory: str) -> Dict[str, List[Dict[str, Any]]]:
    data_store = {}
    for inp in cson_config.get("inputs", []):
        name = inp.get("name")
        pattern = inp.get("file_pattern")
        if not name or not pattern:
            continue

        full_pattern = os.path.join(input_directory, pattern)
        matched_files = glob.glob(full_pattern)
        rows = []
        for fpath in matched_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric columns if needed (simple heuristic)
                    # In a real DSL, you'd have column_types in config
                    for k, v in row.items():
                        if v.isdigit():
                            row[k] = int(v)
                        else:
                            # Try float
                            try:
                                val_float = float(v)
                                row[k] = val_float
                            except:
                                pass
                    rows.append(row)
        data_store[name] = rows
    return data_store


def apply_transformations(
        cson_config: Dict[str, Any],
        data_store: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    transformations = cson_config.get("transformations", {})
    transformed_data = {}
    for t_key, t_val in transformations.items():
        if not isinstance(t_val, dict):
            continue

        # Handle only type: array transformations in this example
        if t_val.get("type") == "array":
            source_name = t_val.get("from")
            if source_name not in data_store:
                raise Exception(
                    f"Source {source_name} not found in data_store")
            rows = data_store[source_name]

            # Apply filter if present
            fil = t_val.get("filter")
            if fil:
                rows = filter_data(rows, fil)

            # Map item_structure
            item_struc = t_val.get("item_structure", {})
            out_array = []
            for row in rows:
                out_item = {}
                for field_key, field_val in item_struc.items():
                    val = resolve_field(field_val, row)
                    out_item[field_key] = val
                out_array.append(out_item)

            transformed_data[t_key] = out_array
        else:
            # For simplicity, ignore other transformation types in this demo
            pass
    return transformed_data


def filter_data(rows: List[Dict[str, Any]], fil: str) -> List[Dict[str, Any]]:
    # Expected format: "= $column_name > 0"
    # We will parse very simply: assume it's always in form "= $col > number"
    # This is a big simplification. Real DSL parsing is more complex.
    fil = fil.strip()
    if not fil.startswith("="):
        return rows  # no filter
    fil_expr = fil[1:].strip()  # remove leading '='
    # fil_expr should look like "$price > 20"
    parts = fil_expr.split()
    if len(parts) != 3:
        return rows
    col_ref, op, value_str = parts
    if col_ref.startswith("$"):
        col_name = col_ref[1:]
    else:
        return rows

    # Convert value_str to float or int if possible
    try:
        filter_val = float(value_str)
    except:
        filter_val = value_str

    # Only handle '>' operator for this demo
    def check(row):
        row_val = row.get(col_name)
        if isinstance(row_val,
                      (int, float)) and isinstance(filter_val, (int, float)):
            if op == ">":
                return row_val > filter_val
            # Could add more ops if needed
        return True  # if doesn't match conditions, don't filter

    return [r for r in rows if check(r)]


def resolve_field(field_val: str, row: Dict[str, Any]) -> Any:
    # If field_val starts with "$", return that column
    # Else return field_val as literal string
    # This is a simplification. Real DSL might handle expressions.
    field_val = str(field_val)
    if field_val.startswith("$"):
        col_name = field_val[1:]
        return row.get(col_name)
    return field_val


def write_outputs(cson_config: Dict[str, Any],
                  transformed_data: Dict[str, Any], output_directory: str):
    outputs = cson_config.get("outputs", [])
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    for out_conf in outputs:
        root = out_conf.get("root")
        format_ = out_conf.get("format", "json")
        file_ = out_conf.get("file")
        if not root or not file_:
            continue

        data_to_write = transformed_data.get(root, [])

        output_path = os.path.join(output_directory, file_)

        if format_ == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_write,
                          f,
                          indent=2 if out_conf.get("pretty") else None)
        else:
            # For simplicity, only handle JSON in this example
            raise Exception("Only JSON output supported in this demo")


if __name__ == "__main__":
    # Example usage:
    # Ensure you have 'cson_config.yaml' in the same dir,
    # 'input_data' folder with matching CSVs,
    # and an 'output_data' folder.
    process_cson(cson_file_path="cson_config.yaml",
                 input_directory="./input_data",
                 output_directory="./output_data",
                 parameters={
                     "ENVIRONMENT": "dev",
                     "INCLUDE_PRICING": True,
                     "DATE_CUTOFF": "2021-01-01"
                 })
