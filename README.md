# OmniLang CSON: CSV-to-JSON Transformation DSL

**CSON** (CSV-Structured Object Notation) is a declarative, YAML-based DSL (Domain-Specific Language) designed to transform one or more CSV files into structured JSON outputs. It gives you the power to:

- Define schemas for multiple, differently structured CSV inputs.
- Enforce data quality with validation rules (including regex patterns).
- Map CSV columns into nested JSON objects and arrays.
- Apply transformations: arithmetic, conditionals, string operations, date formatting, and aggregations.
- Filter rows and join data from multiple CSV sources on matching keys.
- Reference parameters and macros to avoid repetitive logic.
- Extend functionality through plugins.
- Produce multiple outputs in JSON, CSV, or other formats—split by fields if desired.
- Benefit from IntelliSense, autocomplete, and validation in your editor, backed by a JSON schema.

With CSON, you can maintain and evolve your ETL logic as data requirements change, all without modifying code—just update the DSL file.

---

## Key Concepts

### Inputs
**Inputs** define how CSON reads and validates your CSV files:

- **File Patterns**: Identify groups of CSVs with wildcards (e.g., `product_data_*.csv`).
- **Columns & Types**: Declare expected columns and their data types (`int`, `float`, `str`, `bool`, `date`).
- **Validation**: Enforce data integrity (required columns, min/max values, regex patterns for strings).
- **Pre-Processing**: Trim whitespace, parse dates, and apply basic validations before transformations.
  
For example, you might define:

```yaml
inputs:
  - name: "products"
    file_pattern: "product_data_*.csv"
    columns: ["product_id", "width", "height", "category"]
    column_types:
      product_id: int
      width: int
      height: int
      category: str
    validate:
      required_columns: ["product_id", "width", "height"]
      category_pattern: "^[A-Za-z0-9_-]+$"
```

### Transformations
**Transformations** describe the final JSON structure and how to derive it from inputs:

- **Column References**: `$column_name` maps CSV fields into the JSON output.
- **Expressions**: Start with `=` to perform arithmetic, conditionals, string manipulation, regex matching, and more.
- **Objects & Arrays**: Nested YAML maps produce nested JSON objects; `type: array` creates arrays with `item_structure`.
- **Filtering & Aggregation**: Use `filter` to exclude rows, and functions like `sum()`, `min()`, `max()`, `avg()`, `count()`, and `count_unique()` to aggregate data.
- **Conditional Fields**: Include fields or entire blocks only if certain conditions are met.
- **Joining Data**: `match_from` and `match_by` allow joining rows from another input by a key field.

For example:
```yaml
transformations:
  products:
    type: array
    from: "products"
    filter: "= $width > 0 and $height > 0 and match($category, '^[A-Za-z0-9_-]+$')"
    item_structure:
      id: "$product_id"
      area: "= $width * $height"
```

### Outputs
**Outputs** define how and where transformed data is written:

- **Single or Multiple Files**: Produce one big JSON or split by a field into many files.
- **Formats**: JSON, CSV, or other formats (if supported).
- **Formatting & Post-Processing**: Pretty-print JSON, run post-process steps (compress, upload, notify).

Example:
```yaml
outputs:
  - root: "products"
    format: "json"
    file: "products_enriched.json"
    pretty: true
```

### Parameters & Variables
**Parameters** allow dynamic behavior without editing the DSL. Reference parameters (injected from environment or CLI) in expressions, filters, and conditions:
```yaml
parameters:
  environment: "$ENVIRONMENT"
  date_cutoff: "$DATE_CUTOFF"
```

Then use `date_cutoff` in filters: `filter: "= $release_date >= date_cutoff"`

### Macros
**Macros** are reusable expressions defined once and referenced multiple times:
```yaml
macros:
  compute_area: "= $width * $height"
  large_item: "=if($width * $height > 50, true, false)"
```
Use them in transformations:
```yaml
area: "=compute_area()"
flag: "=large_item"
```

Macros can also incorporate parameters:
```yaml
macros:
  is_recent: "= $release_date >= date_cutoff"
```

### Plugins
**Plugins** extend CSON’s capabilities with custom functions. For example, add `format_date()`, `percent()`, or `replace()` to manipulate data in ways not supported natively.

```yaml
plugins:
  - "./plugins/custom_functions.py"
```

### Regular Expressions
**Regex** is used for both validation and expressions:

1. **Validation**: `category_pattern: "^[A-Za-z0-9_-]+$"`
2. **Expressions**: `filter: "= match($category, '^[A-Za-z0-9_-]+$')"`

With plugins, you might `replace()` substrings matching a regex pattern.

### Expression Syntax
Expressions begin with `=` and can reference columns, parameters, macros, and functions. Common features:

- **Column Reference**: `$column_name`
- **Arithmetic**: `= $width * $height`
- **Conditionals**: `=if($stock > 0, 'In Stock', 'Out of Stock')`
- **String Ops**: `concat()`, `lowercase()`, `substring()`
- **Regex Matching**: `match($category, 'pattern')`
- **Date/Time**: `format_date($date, '%Y-%m-%d')` (via plugin)
- **Multiple Conditions**: Combine with `and`, `or`:
  `= $price > 20 and $category == 'electronics'`
- **Nested Conditionals**:
  `=if($width > 0, if($height > 0, 'Valid', 'No Height'), 'No Width')`

### IntelliSense & Autocompletion
By integrating CSON’s JSON Schema with your editor, you get autocomplete, validation, and inline help. The schema file is located at `./core/cson-schema.json`.

In VSCode (`.vscode/settings.json`):
```json
"yaml.schemas": {
  "file:///full/path/to/core/cson-schema.json": "/**/*.cson.yaml"
}
```

This provides:
- **Autocomplete**: Suggestions for CSON keys and functions.
- **Syntax Validation**: Errors highlighted as you type.
- **Hover Documentation**: Inline help for fields and functions.

### Error Handling
CSON supports an `error_handling` section:
```yaml
error_handling:
  on_missing_column: "error"
  on_type_mismatch: "warn"
  on_pattern_violation: "error"
  on_function_failure: "error"
```
- **Behavior**: If a required column is missing or a pattern is violated, the tool may stop processing and report errors. For warnings, it might continue but log the issue. The exact behavior (stop, skip rows, or log) depends on the tool’s implementation. Check tool documentation for details.

### Performance & Scalability
CSON is declarative, and performance depends on the underlying implementation. For very large datasets:
- Consider streaming or chunked processing if supported by your tool.
- Profile transformations and optimize costly expressions.
- Integrate CSON into larger ETL frameworks or CI/CD pipelines.

### Security Considerations
- **Plugins**: Only load trusted plugins to avoid malicious code.
- **Expressions**: Ensure expression evaluation is sandboxed if untrusted data might appear.
- **Validation**: Strict validations and regex patterns help prevent unexpected data from propagating.

### Best Practices
- **Keep Config Modular**: Split large configs into includes and macros.
- **Version Control**: Store CSON files in Git for history and rollback.
- **Test on Small Subsets**: Use a dry-run mode (if supported) to validate logic before processing large datasets.
- **Document Transformations**: Add `description` fields for clarity.
- **Combine With CI/CD**: Validate and test your CSON files as part of a continuous integration pipeline.

---

## Example CSON Configuration

Below is a single, comprehensive example that demonstrates multiple inputs, filtering, regex usage, parameters, macros, plugins, transformations, outputs, and error handling:

```yaml
dsl_version: "2.0"
description: "A comprehensive CSON example with regex, parameters, macros, plugins, and outputs."

parameters:
  environment: "$ENVIRONMENT"
  include_pricing: "$INCLUDE_PRICING"
  date_cutoff: "$DATE_CUTOFF"

plugins:
  - "./plugins/custom_functions.py"

macros:
  compute_area: "= $width * $height"
  normalize_color: "=lowercase($color)"
  final_price: "= $base_price * (1 - $discount)"
  is_recent: "= $release_date >= date_cutoff"

inputs:
  - name: "products"
    file_pattern: "product_data_*.csv"
    columns: ["product_id", "width", "height", "color", "category", "release_date"]
    column_types:
      product_id: int
      width: int
      height: int
      color: str
      category: str
      release_date: date
    date_formats:
      release_date: "%Y-%m-%d"
    trim_values: true
    validate:
      required_columns: ["product_id", "width", "height", "category"]
      width_min: 1
      height_min: 1
      category_pattern: "^[A-Za-z0-9_-]+$"

  - name: "inventory"
    file_pattern: "inventory_data_*.csv"
    columns: ["product_id", "stock", "warehouse_location"]
    column_types:
      product_id: int
      stock: int
      warehouse_location: str
    trim_values: true
    validate:
      required_columns: ["product_id", "stock"]
      stock_min: 0

  - name: "pricing"
    file_pattern: "pricing_data_*.csv"
    columns: ["product_id", "base_price", "discount", "currency"]
    column_types:
      product_id: int
      base_price: float
      discount: float
      currency: str
    validate:
      required_columns: ["product_id", "base_price"]
      base_price_min: 0.0
      discount_range: [0.0, 1.0]

transformations:
  metadata:
    source_system: "Warehouse System"
    generated_at: "=now()"
    environment: "= $environment"
    formatted_date: "=format_date(now(), '%Y/%m/%d %H:%M:%S')"

  products:
    description: "Enriched product data joined with inventory and optionally pricing."
    type: array
    from: "products"
    filter: "= $width > 0 and $height > 0 and is_recent and match($category, '^[A-Za-z0-9_-]+$')"
    item_structure:
      productId: "$product_id"
      dimensions:
        width: "$width"
        height: "$height"
        area: "=compute_area()"
      attributes:
        color: "=normalize_color()"
        category: "$category"
        description: "=concat('Product ', $product_id, ': ', $width, 'x', $height, ' in ', $color)"
        largeItemFlag: "=if($width * $height > 50, true, false)"

      inventory:
        match_from: "inventory"
        match_by: "product_id"
        stock: "$stock"
        warehouse_location: "$warehouse_location"
        status: "=if($stock > 0, 'In Stock', 'Out of Stock')"

      pricing:
        condition: "= $include_pricing == true"
        match_from: "pricing"
        match_by: "product_id"
        basePrice: "$base_price"
        discount: "$discount"
        currency: "$currency"
        finalPrice: "=final_price()"
        discountPercent: "=percent($discount)"

  inventory_summary:
    description: "Summarized inventory data grouped by warehouse."
    type: object
    warehouses:
      type: array
      from: "inventory"
      group_by: "$warehouse_location"
      item_structure:
        warehouse: "$warehouse_location"
        totalStock: "=sum(stock)"
        minStock: "=min(stock)"
        maxStock: "=max(stock)"
        avgStock: "=avg(stock)"
    total_stock_all_warehouses: "=sum(inventory.stock)"
    warehouse_count: "=count_unique(inventory.warehouse_location)"

  partner_feed:
    description: "Feed filtered for partner, only 'electronics' category, recent products."
    type: array
    from: "products"
    filter: "=$category == 'electronics' and is_recent"
    item_structure:
      id: "$product_id"
      color: "=lowercase($color)"
      area: "=compute_area()"
      releaseDateFormatted: "=format_date($release_date, '%d-%m-%Y')"

outputs:
  - root: "products"
    format: "json"
    file: "enriched_products.json"
    pretty: true
    post_process:
      - type: "compress"
        format: "gzip"

  - root: "inventory_summary"
    format: "json"
    file: "inventory_summary.json"
    pretty: true

  - root: "partner_feed"
    format: "json"
    split_by: "partner_feed.color"
    split_file_pattern: "partner_feed_{value}.json"
    pretty: false

  - root: "products"
    format: "csv"
    file: "product_list.csv"
    columns: ["productId", "attributes.category", "dimensions.area"]
    pretty: false

error_handling:
  on_missing_column: "error"
  on_type_mismatch: "warn"
  on_pattern_violation: "error"
  on_function_failure: "error"
```

This example demonstrates how all these features fit together into one powerful and flexible configuration.

---
## List all projects
python main.py --list-projects

### Process specific project
python main.py --project project1

### Process project in custom projects directory
python main.py --project project1 --projects-root /path/to/projects


## Conclusion

CSON provides a rich, declarative environment for transforming CSV data into structured JSON, integrating validation, regex, parameters, macros, plugins, and robust expression handling. With a focus on usability, extensibility, and maintainability, CSON empowers developers and data engineers to adapt transformations quickly as data and requirements evolve.

By using parameters for dynamic behavior, macros for reusability, plugins for custom functions, and the JSON schema for IntelliSense, you can create a fully integrated, developer-friendly ETL workflow. The error handling options and best practices guide you toward stable and secure usage in production scenarios.

Welcome to CSON—your powerful companion for CSV-to-JSON transformations!
