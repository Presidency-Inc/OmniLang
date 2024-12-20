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

file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "cson.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

PLUGIN_FUNCTIONS: Dict[str, Callable] = {}

class ExpressionError(Exception):
    pass

#####################################
# Parsing and Evaluation Utilities
#####################################

TOKEN_PATTERN = re.compile(
    r"[()]|==|!=|>=|<=|[=<>]|[A-Za-z0-9_.$]+|\"[^\"]*\"|'[^']*'|,|\+|\-|\*|/|\band\b|\bor\b|\bnot\b"
)

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
            if p in ['==','!=','>','<','>=','<=']:
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
            if p in ['+','-']:
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
            if p in ['*','/']:
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
        elif p.replace('.','',1).isdigit() or (p.replace('.','',1).replace('-','',1).isdigit()):
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
                if ident.lower() in ['true','false']:
                    return ('bool', ident.lower() == 'true')
                return ('param', ident)
        else:
            raise ExpressionError(f"Unexpected token {p} in factor.")

def parse_expression(expr: str):
    logger.debug(f"parse_expression: expr={expr}")
    tokens = tokenize(expr)
    parser = Parser(tokens)
    node = parser.parse_expr()
    if parser.pos != len(tokens):
        raise ExpressionError("Extra tokens after valid expression.")
    return node

def try_float(val):
    if isinstance(val, (int,float)):
        return val
    if isinstance(val, str):
        try:
            return float(val)
        except:
            return None
    return None

def try_parse_date(val):
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except:
            pass
    return None

def string_compare(op, left, right):
    return {
        '==': left == right,
        '!=': left != right,
        '>': left > right,
        '<': left < right,
        '>=': left >= right,
        '<=': left <= right
    }[op]

def numeric_compare(op, left, right):
    return {
        '==': left == right,
        '!=': left != right,
        '>': left > right,
        '<': left < right,
        '>=': left >= right,
        '<=': left <= right
    }[op]

def convert_for_comparison(left, right):
    left_dt = try_parse_date(left)
    right_dt = try_parse_date(right)
    if left_dt and right_dt:
        return left_dt, right_dt
    left_num = try_float(left)
    right_num = try_float(right)
    if left_num is not None and right_num is not None:
        return left_num, right_num
    return None, None

def apply_op(op, left, right):
    logger.debug(f"apply_op: op={op}, left={left}, right={right}")
    if op == 'not':
        return not bool(left)
    if op == '+':
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            return left+right
        return str(left)+str(right)
    elif op == '-':
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            return left - right
    elif op == '*':
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            return left*right
    elif op == '/':
        if isinstance(left,(int,float)) and isinstance(right,(int,float)):
            if right == 0:
                return None
            return left/right
    elif op in ['==','!=','>','<','>=','<=']:
        left_converted, right_converted = convert_for_comparison(left, right)
        if left_converted is None or right_converted is None:
            if isinstance(left, str) and isinstance(right, str):
                return string_compare(op, left, right)
            return False
        if isinstance(left_converted, datetime) and isinstance(right_converted, datetime):
            if op == '==':
                return left_converted == right_converted
            elif op == '!=':
                return left_converted != right_converted
            elif op == '>':
                return left_converted > right_converted
            elif op == '<':
                return left_converted < right_converted
            elif op == '>=':
                return left_converted >= right_converted
            elif op == '<=':
                return left_converted <= right_converted
        else:
            return numeric_compare(op, left_converted, right_converted)
    elif op == 'and':
        return bool(left) and bool(right)
    elif op == 'or':
        return bool(left) or bool(right)
    return None

def apply_function(fname: str, args: List[Any]):
    logger.debug(f"apply_function: fname={fname}, args={args}")
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
        if len(args)==3 and isinstance(args[1],int) and isinstance(args[2],int):
            val = str(args[0])
            return val[args[1]:args[1]+args[2]]
        return None
    elif fn == 'match':
        if len(args)==2 and isinstance(args[0],str):
            pattern = args[1]
            return re.match(pattern,args[0]) is not None
        return False
    elif fn == 'format_date':
        if len(args)==2 and isinstance(args[0],str):
            dt = try_parse_date(args[0])
            if dt:
                return dt.strftime(args[1])
            return args[0]
        return args[0] if args else None
    elif fn == 'percent':
        if args and isinstance(args[0],(int,float)):
            return f"{args[0]*100:.2f}%"
        return None
    elif fn in PLUGIN_FUNCTIONS:
        return PLUGIN_FUNCTIONS[fn](*args)
    return None

def evaluate_ast(node, row, parameters, macros, column_lookup: Callable[[str],Any]):
    logger.debug(f"evaluate_ast: node={node}, row={row}")
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
        args = [evaluate_ast(a, row, parameters, macros, column_lookup) for a in node[2]]
        return apply_function(fname, args)
    return None

def expand_macros(expr: str, macros: Dict[str,str]) -> str:
    logger.debug(f"expand_macros: before expansion expr={expr}, macros={macros}")
    # We'll handle macros that may appear as function calls like compute_area().
    # For each macro, if it's called as macro_name(), we replace that entire call.
    # If the macro definition does not need parentheses, remove them.

    for m_name, m_expr in macros.items():
        # Clean the macro expression by removing the leading '=' if present
        macro_expr = m_expr[1:].strip() if m_expr.startswith('=') else m_expr

        # First handle the case where macro is used as a function call, e.g. compute_area()
        func_pattern = m_name + r'\(\)'
        if re.search(func_pattern, expr):
            # Replace macro_name() with macro_expr
            expr = re.sub(func_pattern, macro_expr, expr)

        # If macro appears without parentheses, just a variable substitution
        # But only replace if the macro_name isn't followed by '(' to avoid partial replacements of function calls
        if m_name in expr:
            # We do a simple replace now that function calls are handled
            expr = expr.replace(m_name, macro_expr)

    logger.debug(f"expand_macros: after expansion expr={expr}")
    return expr

AGG_REGEX = re.compile(r"^(sum|min|max|avg|count|count_unique)\(([^)]+)\)$")

def evaluate_expression(expr: Any, row: Dict[str,Any], parameters: Dict[str,Any], macros: Dict[str,str]) -> Any:
    logger.debug(f"evaluate_expression: expr={expr}, row={row}")
    if not isinstance(expr, str):
        return expr
    expr = expr.strip()
    if expr.startswith("="):
        expr_val = expand_macros(expr[1:].strip(), macros)
        node = parse_expression(expr_val)
        return evaluate_ast(node, row, parameters, macros, lambda c: row.get(c))
    elif expr.startswith("$"):
        col = expr[1:]
        return row.get(col)
    else:
        return expr

def evaluate_expression_with_aggregators(expr:Any, row:Dict[str,Any], parameters:Dict[str,Any], macros:Dict[str,str], data_store:Dict[str,List[Dict[str,Any]]]) -> Any:
    logger.debug(f"evaluate_expression_with_aggregators: expr={expr}, row={row}")
    if not isinstance(expr,str):
        return expr
    expr = expr.strip()
    if expr.startswith("="):
        ex_content = expr[1:].strip()
        m = AGG_REGEX.match(ex_content)
        if m:
            logger.debug(f"Aggregator detected: {m.group(0)}")
            func = m.group(1)
            colref = m.group(2).strip()
            if '.' in colref:
                ds, col = colref.split('.',1)
                if ds in data_store:
                    values = [r.get(col) for r in data_store[ds] if r.get(col) is not None]
                else:
                    values = []
            else:
                values = []
            logger.debug(f"Aggregator: func={func}, values={values}")
            if func == "sum":
                numeric_vals = [v for v in values if isinstance(v,(int,float))]
                return sum(numeric_vals) if numeric_vals else 0
            elif func == "min":
                numeric_vals = [v for v in values if isinstance(v,(int,float))]
                return min(numeric_vals) if numeric_vals else None
            elif func == "max":
                numeric_vals = [v for v in values if isinstance(v,(int,float))]
                return max(numeric_vals) if numeric_vals else None
            elif func == "avg":
                numeric_vals = [v for v in values if isinstance(v,(int,float))]
                return (sum(numeric_vals)/len(numeric_vals)) if numeric_vals else None
            elif func == "count":
                return len(values)
            elif func == "count_unique":
                return len(set(values))
        else:
            node = parse_expression(ex_content)
            return evaluate_ast(node, row, parameters, macros, lambda c: row.get(c))
    elif expr.startswith("$"):
        col = expr[1:]
        return row.get(col)
    else:
        return expr

def group_rows(rows: List[Dict[str,Any]], group_expr: str, parameters, macros) -> Dict[Any,List[Dict[str,Any]]]:
    logger.debug(f"group_rows: group_expr={group_expr}, rows_count={len(rows)}")
    grouped = {}
    for r in rows:
        val = evaluate_expression(group_expr, r, parameters, macros)
        grouped.setdefault(val, []).append(r)
    logger.debug(f"group_rows result: {grouped.keys()}")
    return grouped

def nested_get(obj: Any, path: str):
    parts = path.split('.')
    cur = obj
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur

def run_post_process(steps: List[Dict[str,Any]], file_path: str):
    for step in steps:
        stype = step.get("type")
        logger.debug(f"post_process step={step} for file={file_path}")
        if stype == "compress":
            logger.info(f"Compressing {file_path} (stub)")
        elif stype == "upload":
            logger.info(f"Uploading {file_path} (stub)")
        elif stype == "notify":
            logger.info(f"Notifying about {file_path} (stub)")

def write_output_file(fmt: str, path: str, data: Any, out_conf: Dict[str,Any]):
    logger.debug(f"write_output_file: fmt={fmt}, path={path}, data_count={len(data) if isinstance(data, list) else 'N/A'}")
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
                writer.writerow({c: (row.get(c,"") if c in row else "") for c in cols})
    else:
        logger.error(f"Unsupported output format: {fmt}")
        sys.exit(1)

def convert_type(val: Any, t: str) -> Any:
    logger.debug(f"convert_type: val={val}, type={t}")
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
        if v in ['true','1','yes']:
            return True
        elif v in ['false','0','no']:
            return False
        return None
    elif t == 'date':
        return val
    return val

def handle_error(mode_key: str, message: str, error_handling: Dict[str,str]):
    mode = error_handling.get(mode_key, "error")
    logger.debug(f"handle_error: mode_key={mode_key}, message={message}, mode={mode}")
    if mode == "error":
        logger.error(message)
        sys.exit(1)
    elif mode == "warn":
        logger.warning(message)
    elif mode == "ignore":
        logger.debug(f"Ignored issue: {message}")

def load_cson_config(path: str) -> Dict[str, Any]:
    logger.debug(f"Loading CSON config from {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load CSON file: {e}")
        return {}

def validate_cson(cson_config: Dict[str,Any]) -> List[str]:
    logger.debug("Validating CSON structure.")
    errors = []
    if "inputs" not in cson_config or not isinstance(cson_config["inputs"], list):
        errors.append("Missing or invalid 'inputs' section.")
    if "transformations" not in cson_config or not isinstance(cson_config["transformations"], dict):
        errors.append("Missing or invalid 'transformations' section.")
    if "outputs" not in cson_config or not isinstance(cson_config["outputs"], list):
        errors.append("Missing or invalid 'outputs' section.")
    return errors

def load_plugins(plugin_paths: List[str], project_path: str):
    logger.debug("Loading plugins...")
    for p in plugin_paths:
        absolute_path = os.path.join(project_path, p.lstrip('./'))
        logger.debug(f"Loading plugin {absolute_path}")
        try:
            spec = importlib.util.spec_from_file_location("plugin_module", absolute_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)  # type: ignore
            if hasattr(plugin_module, "register_functions"):
                plugin_module.register_functions(PLUGIN_FUNCTIONS)
                logger.debug(f"Plugin {absolute_path} loaded successfully.")
        except Exception as e:
            logger.warning(f"Error loading plugin {absolute_path}: {str(e)}")

def load_inputs(cson_config: Dict[str,Any], input_directory: str) -> Dict[str, List[Dict[str,Any]]]:
    logger.info("Loading input CSV files.")
    data_store = {}
    error_handling = cson_config.get("error_handling", {})
    inputs = cson_config.get("inputs", [])
    for inp in inputs:
        name = inp.get("name")
        pattern = inp.get("file_pattern")
        logger.debug(f"Input def: name={name}, pattern={pattern}")
        if not name or not pattern:
            continue

        col_types = inp.get("column_types", {})
        required_columns = inp.get("validate", {}).get("required_columns", [])
        category_pattern = inp.get("validate", {}).get("category_pattern")
        trim_values = inp.get("trim_values", False)
        date_formats = inp.get("date_formats", {})

        full_pattern = os.path.join(input_directory, pattern)
        matched_files = glob.glob(full_pattern)
        rows = []
        for fpath in matched_files:
            logger.debug(f"Reading file {fpath} for input {name}")
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for rc in required_columns:
                        if rc not in row:
                            handle_error("on_missing_column", f"Missing required column {rc} in {fpath}", error_handling)

                    if trim_values:
                        for k in row:
                            if isinstance(row[k], str):
                                row[k] = row[k].strip()

                    for c, t in col_types.items():
                        if c in row:
                            val = row[c]
                            converted = convert_type(val, t)
                            if converted is None and val != '' and val is not None:
                                handle_error("on_type_mismatch", f"Column {c} value '{val}' cannot be converted to {t}", error_handling)
                            else:
                                row[c] = converted

                    if category_pattern and "category" in row:
                        cat_val = row["category"]
                        if isinstance(cat_val,str) and not re.match(category_pattern, cat_val):
                            handle_error("on_pattern_violation", f"category '{cat_val}' does not match pattern {category_pattern}", error_handling)

                    for c, fmt in date_formats.items():
                        if c in row and isinstance(row[c], str):
                            try:
                                dt = datetime.strptime(row[c], fmt)
                                row[c] = dt.isoformat()
                            except:
                                pass

                    rows.append(row)
        data_store[name] = rows
        logger.debug(f"Loaded {len(rows)} rows for {name}.")
    return data_store

def evaluate_aggregation(expr: Any, rows: List[Dict[str,Any]], parameters, macros, data_store:Dict[str,List[Dict[str,Any]]]) -> Any:
    logger.debug(f"evaluate_aggregation: expr={expr}, rows_count={len(rows)}")
    if not isinstance(expr, str):
        return expr
    ex = expr.strip()
    if not ex.startswith("="):
        if '$' in ex:
            if rows:
                first_row = rows[0]
                node = parse_expression(ex)
                return evaluate_ast(node, first_row, parameters, macros, lambda c: first_row.get(c))
            else:
                return None
        else:
            return expr

    ex_content = ex[1:].strip()
    m = re.match(r"(sum|min|max|avg|count|count_unique)\(([^)]+)\)", ex_content)
    if m:
        func = m.group(1)
        colref = m.group(2).strip()
        if '.' in colref:
            ds, col = colref.split('.',1)
            if ds in data_store:
                values = [r.get(col) for r in data_store[ds] if r.get(col) is not None]
            else:
                values = []
        else:
            values = [r.get(colref) for r in rows if r.get(colref) is not None]
        logger.debug(f"Aggregator in evaluate_aggregation: func={func}, values={values}")
        if func == "sum":
            numeric_vals = [v for v in values if isinstance(v,(int,float))]
            return sum(numeric_vals) if numeric_vals else 0
        elif func == "min":
            numeric_vals = [v for v in values if isinstance(v,(int,float))]
            return min(numeric_vals) if numeric_vals else None
        elif func == "max":
            numeric_vals = [v for v in values if isinstance(v,(int,float))]
            return max(numeric_vals) if numeric_vals else None
        elif func == "avg":
            numeric_vals = [v for v in values if isinstance(v,(int,float))]
            return (sum(numeric_vals)/len(numeric_vals)) if numeric_vals else None
        elif func == "count":
            return len(values)
        elif func == "count_unique":
            return len(set(values))
    else:
        # Non-aggregator expression
        if rows:
            first_row = rows[0]
            node = parse_expression(ex_content)
            return evaluate_ast(node, first_row, parameters, macros, lambda c: first_row.get(c))
        else:
            return None

def build_item(row: Dict[str,Any], item_structure: Dict[str,Any], parameters, macros, data_store:Dict[str,List[Dict[str,Any]]]={}) -> Dict[str,Any]:
    logger.debug(f"build_item: row={row}, item_structure_keys={list(item_structure.keys())}")
    out_item = {}
    for field_key, field_val in item_structure.items():
        logger.debug(f"build_item field: {field_key}")
        if isinstance(field_val, dict) and "type" in field_val:
            out_item[field_key] = execute_transformation(field_val, data_store, parameters, macros)
        elif isinstance(field_val, dict):
            mf = field_val.get("match_from")
            mb = field_val.get("match_by")
            cond_expr = field_val.get("condition")
            if mf and mb:
                ref_val = row.get(mb)
                logger.debug(f"build_item matching from={mf}, by={mb}, ref_val={ref_val}")
                if mf in data_store:
                    matched = [x for x in data_store[mf] if x.get(mb) == ref_val]
                    logger.debug(f"build_item matched rows: {len(matched)}")
                    if matched:
                        matched_row = matched[0]
                        sub_obj = {}
                        for sk, sv in field_val.items():
                            if sk in ["match_from","match_by","condition"]:
                                continue
                            if isinstance(sv, dict) and "type" in sv:
                                sub_obj[sk] = execute_transformation(sv, data_store, parameters, macros)
                            else:
                                sub_obj[sk] = evaluate_expression(sv, matched_row, parameters, macros)
                        if cond_expr:
                            logger.debug(f"build_item evaluating condition: {cond_expr}")
                            if evaluate_expression(cond_expr, row, parameters, macros):
                                out_item[field_key] = sub_obj
                        else:
                            out_item[field_key] = sub_obj
                    else:
                        logger.debug("No match found in joined dataset")
                else:
                    logger.debug(f"No dataset named {mf} found in data_store")
            else:
                out_item[field_key] = build_item(row, field_val, parameters, macros, data_store)
        else:
            val = evaluate_expression(field_val, row, parameters, macros)
            out_item[field_key] = val
    return out_item

def execute_object_transformation(obj_def: Dict[str,Any],
                                  data_store: Dict[str,List[Dict[str,Any]]],
                                  parameters: Dict[str,Any],
                                  macros: Dict[str,str]) -> Dict[str,Any]:
    logger.debug(f"execute_object_transformation: obj_def={obj_def}")
    result = {}
    fields = obj_def.get("fields", {})
    reserved_keys = ["type","from","filter","group_by","condition","item_structure","fields","description"]
    for k,v in obj_def.items():
        if k in reserved_keys:
            continue
        logger.debug(f"Processing object field {k} in object transformation")
        if isinstance(v, dict) and "type" in v:
            val = execute_transformation(v, data_store, parameters, macros)
        else:
            val = evaluate_expression_with_aggregators(v, {}, parameters, macros, data_store)
        result[k] = val

    for fk, fv in fields.items():
        logger.debug(f"Processing field {fk} in fields of object")
        if isinstance(fv, dict) and "type" in fv:
            val = execute_transformation(fv, data_store, parameters, macros)
        else:
            val = evaluate_expression_with_aggregators(fv, {}, parameters, macros, data_store)
        result[fk] = val

    return result

def execute_array_transformation(arr_def: Dict[str,Any],
                                 data_store: Dict[str,List[Dict[str,Any]]],
                                 parameters: Dict[str,Any],
                                 macros: Dict[str,str]) -> List[Any]:
    logger.debug(f"execute_array_transformation: arr_def={arr_def}")
    from_name = arr_def.get("from")
    if not from_name or from_name not in data_store:
        logger.debug("No 'from' dataset or dataset empty.")
        return []
    rows = data_store[from_name]

    fil_expr = arr_def.get("filter")
    if fil_expr:
        logger.debug(f"Applying filter on array: {fil_expr}")
        before_count = len(rows)
        rows = [r for r in rows if evaluate_expression(fil_expr, r, parameters, macros)]
        logger.debug(f"Filtered rows from {before_count} to {len(rows)}")

    cond_expr = arr_def.get("condition")
    if cond_expr:
        cond_res = evaluate_expression(cond_expr, {}, parameters, macros)
        logger.debug(f"Array condition: {cond_expr} => {cond_res}")
        if not cond_res:
            return []

    if "group_by" in arr_def:
        group_key = arr_def["group_by"]
        logger.debug(f"Grouping array by {group_key}")
        grouped = group_rows(rows, group_key, parameters, macros)
        item_struc = arr_def.get("item_structure", {})
        arr = []
        for k, group_list in grouped.items():
            logger.debug(f"Group: key={k}, count={len(group_list)}")
            item = {}
            for fk, fv in item_struc.items():
                val = evaluate_aggregation(fv, group_list, parameters, macros, data_store)
                item[fk] = val
            arr.append(item)
        return arr
    else:
        item_struc = arr_def.get("item_structure", {})
        logger.debug(f"Building items for array with {len(rows)} rows")
        arr = [build_item(r, item_struc, parameters, macros, data_store) for r in rows]
        return arr

def execute_transformation(trans_def: Dict[str, Any],
                           data_store: Dict[str, List[Dict[str,Any]]],
                           parameters: Dict[str,Any],
                           macros: Dict[str,str]) -> Any:
    logger.debug(f"execute_transformation: trans_def={trans_def}")
    ttype = trans_def.get("type")
    if ttype == "object":
        return execute_object_transformation(trans_def, data_store, parameters, macros)
    elif ttype == "array":
        return execute_array_transformation(trans_def, data_store, parameters, macros)
    else:
        return evaluate_expression(trans_def, {}, parameters, macros)

def write_outputs(cson_config: Dict[str, Any], transformed_data: Dict[str, Any], output_directory: str):
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
        logger.debug(f"Writing output for root={root}, count={(len(data_to_write) if isinstance(data_to_write,list) else 'N/A')}")
        split_by = out_conf.get("split_by")
        if split_by:
            if not isinstance(data_to_write, list):
                data_to_write = []
            grouped = {}
            for item in data_to_write:
                val = nested_get(item, split_by)
                grouped.setdefault(val, []).append(item)
            logger.debug(f"Splitting output by {split_by} into {len(grouped)} files.")
            pattern = out_conf.get("split_file_pattern", "output_{value}.json")
            for gv, glist in grouped.items():
                out_path = os.path.join(output_directory, pattern.replace("{value}", str(gv)))
                logger.debug(f"Split output {gv}: count={len(glist)} to {out_path}")
                write_output_file(fmt, out_path, glist, out_conf)
                run_post_process(out_conf.get("post_process", []), out_path)
        else:
            out_path = os.path.join(output_directory, file_)
            logger.debug(f"Single output to {out_path}")
            write_output_file(fmt, out_path, data_to_write, out_conf)
            run_post_process(out_conf.get("post_process", []), out_path)

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

    macros = cson_config.get("macros", {})
    cson_config["macros"] = macros

    project_path = os.path.dirname(os.path.abspath(cson_file_path))
    load_plugins(cson_config.get("plugins", []), project_path)

    data_store = load_inputs(cson_config, input_directory)

    transformations = cson_config.get("transformations", {})
    parameters = cson_config.get("parameters", {})
    macros = cson_config.get("macros", {})

    logger.debug(f"Final macros used: {macros}")

    transformed_data = {}
    for t_key, t_val in transformations.items():
        logger.debug(f"Top-level transformation {t_key}")
        val = execute_transformation(t_val, data_store, parameters, macros)
        transformed_data[t_key] = val
        logger.debug(f"Result for {t_key}: {val}")

    logger.debug(f"All transformations done. Transformed data keys: {transformed_data.keys()}")
    write_outputs(cson_config, transformed_data, output_directory)
    logger.info("CSON processing completed successfully.")

def setup_project_logging(project_path: str) -> logging.Logger:
    abs_project_path = os.path.abspath(project_path)
    log_dir = os.path.join(abs_project_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = os.path.basename(abs_project_path)
    log_filename = f"{project_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'w') as f:
        f.write('')
    logger = logging.getLogger(f"CSON_{project_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = logging.FileHandler(
        filename=log_path,
        encoding='utf-8',
        mode='w'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file created: {log_filename}")
    return logger

def process_project(project_path: str, parameters: Dict[str, Any] = None):
    abs_project_path = os.path.abspath(project_path)
    project_name = os.path.basename(abs_project_path)
    input_dir = os.path.join(abs_project_path, "input_data")
    output_dir = os.path.join(abs_project_path, "output_data")
    config_file = os.path.join(abs_project_path, "transformations.yaml")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    global logger
    logger = setup_project_logging(abs_project_path)  
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return False

    try:
        process_cson(
            cson_file_path=config_file,
            input_directory=input_dir,
            output_directory=output_dir,
            parameters=parameters
        )
        logger.info(f"Successfully processed project: {project_name}")
        return True
    except Exception as e:
        logger.error(f"Error processing project {project_name}: {str(e)}")
        return False

def list_projects(projects_root: str = "projects") -> List[str]:
    if not os.path.exists(projects_root):
        return []
    return [d for d in os.listdir(projects_root) if os.path.isdir(os.path.join(projects_root, d))]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CSON Project Processor')
    parser.add_argument('--project', help='Name of the project to process')
    parser.add_argument('--list-projects', action='store_true', help='List all available projects')
    parser.add_argument('--projects-root', default='projects', help='Root directory containing all projects')
    
    args = parser.parse_args()

    if args.list_projects:
        projects = list_projects(args.projects_root)
        if projects:
            print("Available projects:")
            for proj in projects:
                print(f"  - {proj}")
        else:
            print("No projects found.")
        sys.exit(0)
    
    if not args.project:
        print("Error: Please specify a project name with --project")
        sys.exit(1)
    
    project_path = os.path.join(args.projects_root, args.project)
    if not os.path.exists(project_path):
        print(f"Error: Project '{args.project}' not found")
        sys.exit(1)
    
    parameters = {
        "ENVIRONMENT": "prod",
        "INCLUDE_PRICING": True,
        "DATE_CUTOFF": "2022-01-01"
    }
    
    success = process_project(project_path, parameters)
    sys.exit(0 if success else 1)
