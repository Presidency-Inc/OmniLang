import os
import sys
import glob
import json
import csv
import yaml
import re
from typing import Any, Dict, List, Callable, Optional, Union
from datetime import datetime
import importlib.util
import math
import logging
import logging.handlers

#####################################
# Setup Logging
#####################################
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("CSON")
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "cson.log"),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)

# Console handler (stderr) for warnings and errors
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

#####################################
# Global function registry for plugins
#####################################
PLUGIN_FUNCTIONS: Dict[str, Callable] = {}

#####################################
# Error Handling Modes
#####################################
# on_missing_column: "error", "warn", "ignore"
# on_type_mismatch: "error", "warn", "ignore"
# on_pattern_violation: "error", "warn", "ignore"
# on_function_failure: "error", "warn", "ignore"

#####################################
# Expression Parser and Evaluator (as before)
#####################################

TOKEN_PATTERN = re.compile(
    r"[()]|==|!=|>=|<=|[=<>]|[A-Za-z0-9_.$]+|\"[^\"]*\"|'[^']*'|,|\+|\-|\*|/|\band\b|\bor\b|\bnot\b"
)


class ExpressionError(Exception):
    pass


def tokenize(expression: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(expression)
    return [t.strip() for t in tokens if t.strip()]


class Parser:

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        if self.pos >= len(self.tokens):
            raise ExpressionError("Unexpected end of expression.")
        t = self.tokens[self.pos]
        self.pos += 1
        if expected and t != expected:
            raise ExpressionError(f"Expected '{expected}' but got '{t}'.")
        return t

    def parse_expr(self) -> Any:
        return self.parse_or_expr()

    def parse_or_expr(self):
        node = self.parse_and_expr()
        while self.peek() == 'or':
            op = self.consume()
            right = self.parse_and_expr()
            node = ('op', op, node, right)
        return node

    def parse_and_expr(self):
        node = self.parse_not_expr()
        while self.peek() == 'and':
            op = self.consume()
            right = self.parse_not_expr()
            node = ('op', op, node, right)
        return node

    def parse_not_expr(self):
        ops = []
        while self.peek() == 'not':
            ops.append(self.consume())
        node = self.parse_cmp_expr()
        for op in reversed(ops):
            node = ('op', op, node, None)
        return node

    def parse_cmp_expr(self):
        node = self.parse_sum_expr()
        while True:
            p = self.peek()
            if p in ['==', '!=', '>', '<', '>=', '<=']:
                op = self.consume()
                right = self.parse_sum_expr()
                node = ('op', op, node, right)
            else:
                break
        return node

    def parse_sum_expr(self):
        node = self.parse_term()
        while True:
            p = self.peek()
            if p in ['+', '-']:
                op = self.consume()
                right = self.parse_term()
                node = ('op', op, node, right)
            else:
                break
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            p = self.peek()
            if p in ['*', '/']:
                op = self.consume()
                right = self.parse_factor()
                node = ('op', op, node, right)
            else:
                break
        return node

    def parse_factor(self):
        p = self.peek()
        if p is None:
            raise ExpressionError("Unexpected end of expression.")
        if p in ['(', ')', ',', 'and', 'or', 'not']:
            if p == '(':
                self.consume('(')
                node = self.parse_expr()
                self.consume(')')
                return node
            else:
                raise ExpressionError(f"Unexpected token {p} in factor.")
        elif p.startswith('$'):
            var = self.consume()
            return ('var', var[1:])
        elif p.startswith("'") or p.startswith('"'):
            s = self.consume()
            val = s.strip('"').strip("'")
            return ('str', val)
        elif p.replace('.', '', 1).isdigit() or (p.replace('.', '', 1).replace(
                '-', '', 1).isdigit()):
            num_t = self.consume()
            if '.' in num_t:
                return ('num', float(num_t))
            else:
                return ('num', int(num_t))
        elif re.match(r"[A-Za-z_][A-Za-z0-9_]*", p):
            ident = self.consume()
            if self.peek() == '(':
                self.consume('(')
                args = []
                if self.peek() != ')':
                    args.append(self.parse_expr())
                    while self.peek() == ',':
                        self.consume(',')
                        args.append(self.parse_expr())
                self.consume(')')
                return ('func', ident, args)
            else:
                if ident.lower() in ['true', 'false']:
                    return ('bool', ident.lower() == 'true')
                return ('param', ident)
        else:
            raise ExpressionError(f"Unexpected token {p} in factor.")


def parse_expression(expr: str):
    tokens = tokenize(expr)
    parser = Parser(tokens)
    node = parser.parse_expr()
    if parser.pos != len(tokens):
        raise ExpressionError("Extra tokens after valid expression.")
    return node


def evaluate_ast(node, row, parameters, macros, column_lookup: Callable[[str],
                                                                        Any]):
    ntype = node[0]

    if ntype == 'op':
        op = node[1]
        left = evaluate_ast(node[2], row, parameters, macros, column_lookup)
        right = node[3]
        if right is not None:
            right = evaluate_ast(right, row, parameters, macros, column_lookup)
        return apply_op(op, left, right)
    elif ntype == 'var':
        col = node[1]
        return column_lookup(col)
    elif ntype == 'param':
        pname = node[1]
        return parameters.get(pname, None)
    elif ntype == 'str':
        return node[1]
    elif ntype == 'num':
        return node[1]
    elif ntype == 'bool':
        return node[1]
    elif ntype == 'func':
        fname = node[1]
        args = [
            evaluate_ast(a, row, parameters, macros, column_lookup)
            for a in node[2]
        ]
        return apply_function(fname, args)
    return None


def try_float(val):
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return float(val)
        except:
            return None
    return None


def safe_compare(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (left, right)
    left_num = try_float(left)
    right_num = try_float(right)
    if left_num is not None and right_num is not None:
        return (left_num, right_num)
    if isinstance(left, str) and isinstance(right, str):
        return (left, right)
    return None


def apply_op(op, left, right):
    if op == 'not':
        return not bool(left)

    if op == '+':
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        return str(left) + str(right)
    elif op == '-':
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left - right
    elif op == '*':
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left * right
    elif op == '/':
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if right == 0:
                return None
            return left / right
    elif op in ['==', '!=', '>', '<', '>=', '<=']:
        pair = safe_compare(left, right)
        if pair is None:
            # Can't compare
            return False
        left, right = pair
        if op == '==':
            return left == right
        elif op == '!=':
            return left != right
        elif op == '>':
            return left > right
        elif op == '<':
            return left < right
        elif op == '>=':
            return left >= right
        elif op == '<=':
            return left <= right
    elif op == 'and':
        return bool(left) and bool(right)
    elif op == 'or':
        return bool(left) or bool(right)

    return None


def apply_function(fname: str, args: List[Any]):
    fn = fname.lower()
    if fn == 'if':
        if len(args) != 3:
            return None
        return args[1] if args[0] else args[2]
    elif fn == 'concat':
        return "".join(str(a) for a in args)
    elif fn == 'lowercase':
        return str(args[0]).lower() if args else None
    elif fn == 'uppercase':
        return str(args[0]).upper() if args else None
    elif fn == 'substring':
        if len(args) == 3 and isinstance(args[1], int) and isinstance(
                args[2], int):
            val = str(args[0])
            return val[args[1]:args[1] + args[2]]
        return None
    elif fn == 'match':
        if len(args) == 2 and isinstance(args[0], str):
            pattern = args[1]
            return re.match(pattern, args[0]) is not None
        return False
    elif fn == 'format_date':
        if len(args) == 2 and isinstance(args[0], str):
            try:
                dt = datetime.fromisoformat(args[0])
                return dt.strftime(args[1])
            except:
                return args[0]
        return args[0] if args else None
    elif fn == 'percent':
        if args and isinstance(args[0], (int, float)):
            return f"{args[0]*100:.2f}%"
        return None
    elif fn in PLUGIN_FUNCTIONS:
        return PLUGIN_FUNCTIONS[fn](*args)
    return None


#####################################
# Main Processing
#####################################


def process_cson(cson_file_path: str,
                 input_directory: str,
                 output_directory: str,
                 parameters: Dict[str, Any] = None):
    logger.info("Starting CSON processing")
    cson_config = load_cson_config(cson_file_path)
    if not cson_config:
        logger.error("CSON config not loaded. Exiting.")
        sys.exit(1)

    errors = validate_cson(cson_config)
    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)

    if parameters:
        cson_config["parameters"] = parameters
    else:
        cson_config["parameters"] = {}

    load_plugins(cson_config.get("plugins", []))

    data_store = load_inputs(cson_config, input_directory)
    transformed_data = apply_transformations(cson_config, data_store)
    write_outputs(cson_config, transformed_data, output_directory)
    logger.info("CSON processing completed successfully.")


def load_cson_config(path: str) -> Dict[str, Any]:
    logger.debug(f"Loading CSON config from {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load CSON file: {e}")
        return {}


def validate_cson(cson_config: Dict[str, Any]) -> List[str]:
    errors = []
    logger.debug("Validating CSON structure.")
    if "inputs" not in cson_config or not isinstance(cson_config["inputs"],
                                                     list):
        errors.append("Missing or invalid 'inputs' section.")
    if "transformations" not in cson_config or not isinstance(
            cson_config["transformations"], dict):
        errors.append("Missing or invalid 'transformations' section.")
    if "outputs" not in cson_config or not isinstance(cson_config["outputs"],
                                                      list):
        errors.append("Missing or invalid 'outputs' section.")
    return errors


def load_plugins(plugin_paths: List[str]):
    logger.debug("Loading plugins...")
    for p in plugin_paths:
        logger.debug(f"Loading plugin {p}")
        try:
            spec = importlib.util.spec_from_file_location("plugin_module", p)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)  # type: ignore
            if hasattr(plugin_module, "register_functions"):
                plugin_module.register_functions(PLUGIN_FUNCTIONS)
                logger.debug(f"Plugin {p} loaded successfully.")
        except Exception as e:
            logger.warning(f"Error loading plugin {p}: {e}")


def load_inputs(cson_config: Dict[str, Any],
                input_directory: str) -> Dict[str, List[Dict[str, Any]]]:
    logger.info("Loading input CSV files.")
    data_store = {}
    parameters = cson_config.get("parameters", {})
    error_handling = cson_config.get("error_handling", {})
    inputs = cson_config.get("inputs", [])
    for inp in inputs:
        name = inp.get("name")
        pattern = inp.get("file_pattern")
        if not name or not pattern:
            continue
        logger.debug(f"Processing input pattern {pattern} for {name}")

        col_types = inp.get("column_types", {})
        required_columns = inp.get("validate", {}).get("required_columns", [])
        category_pattern = inp.get("validate", {}).get("category_pattern")
        trim_values = inp.get("trim_values", False)
        date_formats = inp.get("date_formats", {})

        full_pattern = os.path.join(input_directory, pattern)
        matched_files = glob.glob(full_pattern)
        rows = []
        for fpath in matched_files:
            logger.debug(f"Reading {fpath}")
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for rc in required_columns:
                        if rc not in row:
                            handle_error(
                                "on_missing_column",
                                f"Missing required column {rc} in {fpath}",
                                error_handling)

                    if trim_values:
                        for k in row:
                            if isinstance(row[k], str):
                                row[k] = row[k].strip()

                    for c, t in col_types.items():
                        if c in row:
                            val = row[c]
                            converted = convert_type(val, t)
                            if converted is None and val != '' and val is not None:
                                handle_error(
                                    "on_type_mismatch",
                                    f"Column {c} value '{val}' cannot be converted to {t}",
                                    error_handling)
                            else:
                                row[c] = converted

                    if category_pattern and "category" in row:
                        cat_val = row["category"]
                        if isinstance(cat_val, str) and not re.match(
                                category_pattern, cat_val):
                            handle_error(
                                "on_pattern_violation",
                                f"category '{cat_val}' does not match pattern {category_pattern}",
                                error_handling)

                    for c, fmt in date_formats.items():
                        if c in row and isinstance(row[c], str):
                            try:
                                dt = datetime.strptime(row[c], fmt)
                                row[c] = dt.isoformat()
                            except:
                                # Not raising error here, just leave as string
                                pass

                    rows.append(row)
        data_store[name] = rows
        logger.debug(f"Loaded {len(rows)} rows for {name}.")
    return data_store


def convert_type(val: Any, t: str) -> Any:
    if val is None or val == '':
        return None
    if t == 'int':
        try:
            return int(val)
        except:
            return None
    elif t == 'float':
        try:
            return float(val)
        except:
            return None
    elif t == 'bool':
        v = str(val).lower()
        if v in ['true', '1', 'yes']:
            return True
        elif v in ['false', '0', 'no']:
            return False
        return None
    elif t == 'date':
        return val
    return val


def handle_error(mode_key: str, message: str, error_handling: Dict[str, str]):
    mode = error_handling.get(mode_key, "error")
    if mode == "error":
        logger.error(message)
        sys.exit(1)
    elif mode == "warn":
        logger.warning(message)
    elif mode == "ignore":
        logger.debug(f"Ignored issue: {message}")


def apply_transformations(
        cson_config: Dict[str, Any],
        data_store: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    logger.info("Applying transformations.")
    transformations = cson_config.get("transformations", {})
    parameters = cson_config.get("parameters", {})
    macros = cson_config.get("macros", {})
    result = {}
    for t_key, t_val in transformations.items():
        logger.debug(f"Processing transformation {t_key}")
        if not isinstance(t_val, dict):
            continue

        ttype = t_val.get("type")
        if ttype == "object":
            obj = build_object(t_val, data_store, parameters, macros, result)
            result[t_key] = obj
        elif ttype == "array":
            from_name = t_val.get("from")
            rows = data_store.get(from_name, [])

            fil_expr = t_val.get("filter")
            if fil_expr:
                logger.debug(f"Filtering rows for {t_key}")
                rows = [
                    r for r in rows
                    if evaluate_expression(fil_expr, r, parameters, macros)
                ]

            cond_expr = t_val.get("condition")
            if cond_expr and not evaluate_expression(cond_expr, {}, parameters,
                                                     macros):
                logger.debug(
                    f"Condition for {t_key} not met, producing empty array.")
                result[t_key] = []
                continue

            if "group_by" in t_val:
                logger.debug(f"Grouping rows for {t_key}")
                grouped = group_rows(rows, t_val["group_by"], parameters,
                                     macros)
                arr = build_grouped_array(grouped,
                                          t_val.get("item_structure", {}),
                                          parameters, macros)
                result[t_key] = arr
            else:
                item_struc = t_val.get("item_structure", {})
                arr = [
                    build_item(r, item_struc, parameters, macros) for r in rows
                ]
                result[t_key] = arr
        else:
            logger.debug(
                f"Unsupported transformation type: {ttype} for {t_key}")
    return result


def build_object(obj_def: Dict[str, Any], data_store, parameters, macros,
                 transformed) -> Dict[str, Any]:
    result = {}
    for k, v in obj_def.items():
        if k in [
                "type", "from", "filter", "group_by", "condition",
                "item_structure", "fields", "description"
        ]:
            continue
        val = evaluate_expression(v, {}, parameters, macros)
        result[k] = val
    fields = obj_def.get("fields", {})
    for fk, fv in fields.items():
        result[fk] = evaluate_expression(fv, {}, parameters, macros)
    return result


def evaluate_expression(expr: Any, row: Dict[str, Any],
                        parameters: Dict[str, Any], macros: Dict[str,
                                                                 str]) -> Any:
    if not isinstance(expr, str):
        return expr
    expr = expr.strip()
    if expr.startswith("="):
        expr_val = expand_macros(expr[1:].strip(), macros)
        node = parse_expression(expr_val)
        return evaluate_ast(node, row, parameters, macros,
                            lambda c: row.get(c))
    elif expr.startswith("$"):
        col = expr[1:]
        return row.get(col)
    else:
        return expr


def expand_macros(expr: str, macros: Dict[str, str]) -> str:
    for m_name, m_expr in macros.items():
        if expr.startswith(m_name + "("):
            if m_expr.startswith("="):
                m_expr_clean = m_expr[1:].strip()
                return expand_macros(m_expr_clean, macros)
            return m_expr
    return expr


def build_item(row: Dict[str, Any], item_structure: Dict[str, Any], parameters,
               macros) -> Dict[str, Any]:
    out_item = {}
    for field_key, field_val in item_structure.items():
        if isinstance(field_val, dict):
            out_item[field_key] = build_item(row, field_val, parameters,
                                             macros)
        else:
            val = evaluate_expression(field_val, row, parameters, macros)
            out_item[field_key] = val
    return out_item


def group_rows(rows: List[Dict[str, Any]], group_expr: str, parameters,
               macros) -> Dict[Any, List[Dict[str, Any]]]:
    grouped = {}
    for r in rows:
        gval = evaluate_expression(group_expr, r, parameters, macros)
        grouped.setdefault(gval, []).append(r)
    return grouped


def build_grouped_array(grouped: Dict[Any, List[Dict[str, Any]]],
                        item_structure: Dict[str, Any], parameters,
                        macros) -> List[Dict[str, Any]]:
    arr = []
    for k, group_rows in grouped.items():
        item = {}
        for fk, fv in item_structure.items():
            val = evaluate_aggregation(fv, group_rows, parameters, macros)
            item[fk] = val
        arr.append(item)
    return arr


def evaluate_aggregation(expr: Any, rows: List[Dict[str, Any]], parameters,
                         macros) -> Any:
    if not isinstance(expr, str):
        return expr
    ex = expr.strip()
    if not ex.startswith("="):
        return expr
    ex_content = ex[1:].strip()

    m = re.match(r"(sum|min|max|avg|count|count_unique)\(([^)]+)\)",
                 ex_content)
    if m:
        func = m.group(1)
        col = m.group(2).strip()
        values = []
        for r in rows:
            val = r.get(col)
            if val is not None:
                values.append(val)
        if func == "sum":
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            return sum(numeric_vals) if numeric_vals else 0
        elif func == "min":
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            return min(numeric_vals) if numeric_vals else None
        elif func == "max":
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            return max(numeric_vals) if numeric_vals else None
        elif func == "avg":
            numeric_vals = [v for v in values if isinstance(v, (int, float))]
            return (sum(numeric_vals) /
                    len(numeric_vals)) if numeric_vals else None
        elif func == "count":
            return len(values)
        elif func == "count_unique":
            return len(set(values))
    else:
        node = parse_expression(ex_content)
        return evaluate_ast(node, {}, parameters, macros, lambda c: None)


def write_outputs(cson_config: Dict[str, Any],
                  transformed_data: Dict[str, Any], output_directory: str):
    logger.info("Writing outputs.")
    outputs = cson_config.get("outputs", [])
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    for out_conf in outputs:
        root = out_conf.get("root")
        fmt = out_conf.get("format", "json")
        file_ = out_conf.get("file")
        if not root or not file_:
            continue
        data_to_write = transformed_data.get(root, [])

        split_by = out_conf.get("split_by")
        if split_by:
            if not isinstance(data_to_write, list):
                data_to_write = []
            grouped = {}
            for item in data_to_write:
                val = nested_get(item, split_by)
                grouped.setdefault(val, []).append(item)
            pattern = out_conf.get("split_file_pattern", "output_{value}.json")
            for gv, glist in grouped.items():
                out_path = os.path.join(output_directory,
                                        pattern.replace("{value}", str(gv)))
                logger.debug(f"Writing split output {out_path}")
                write_output_file(fmt, out_path, glist, out_conf)
                run_post_process(out_conf.get("post_process", []), out_path)
        else:
            out_path = os.path.join(output_directory, file_)
            logger.debug(f"Writing output {out_path}")
            write_output_file(fmt, out_path, data_to_write, out_conf)
            run_post_process(out_conf.get("post_process", []), out_path)


def write_output_file(fmt: str, path: str, data: Any, out_conf: Dict[str,
                                                                     Any]):
    if fmt == "json":
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2 if out_conf.get("pretty") else None)
    elif fmt == "csv":
        if not isinstance(data, list):
            data = []
        cols = out_conf.get("columns", [])
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in data:
                writer.writerow({c: flatten_value(row, c) for c in cols})
    else:
        logger.error(f"Unsupported output format: {fmt}")
        sys.exit(1)


def flatten_value(item: Dict[str, Any], key: str):
    return item.get(key, "")


def run_post_process(steps: List[Dict[str, Any]], file_path: str):
    for step in steps:
        stype = step.get("type")
        if stype == "compress":
            logger.info(f"Compressing {file_path} (stub)")
        elif stype == "upload":
            logger.info(f"Uploading {file_path} (stub)")
        elif stype == "notify":
            logger.info(f"Notifying about {file_path} (stub)")


def nested_get(obj: Any, path: str):
    parts = path.split('.')
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


if __name__ == "__main__":
    # Example usage
    process_cson(cson_file_path="cson_config.yaml",
                 input_directory="./input_data",
                 output_directory="./output_data",
                 parameters={
                     "ENVIRONMENT": "prod",
                     "INCLUDE_PRICING": True,
                     "DATE_CUTOFF": "2022-01-01"
                 })
