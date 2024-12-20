# OmniLang CSON
## CSV-to-JSON Transformation Language & Framework with OmniLang/CSON

Below is a comprehensive README documentation for the project. It covers:

1. How to solve a CSV-to-JSON migration problem with this framework.
2. The OmniLang/CSON language guide (conceptual overview).
3. Instructions on managing and running projects, including configuring inputs, transformations, outputs, and using the provided `main.py`.

## Overview

This project provides a flexible and declarative way to transform CSV data into enriched JSON outputs. It leverages a custom domain-specific language called **OmniLang/CSON** to define transformations, filter criteria, aggregations, and output formats. By configuring transformations in YAML, you can easily adapt to changing business rules, quickly implement new data pipelines, and avoid hardcoding logic in code.

The project is organized around the concept of **projects**, each having its own input CSV files, a `transformations.yaml` configuration file, and an output directory. By following the guide below, you can solve migration problems (e.g., converting old CSV-based datasets into modern JSON APIs), customize transformations, and manage multiple projects in a consistent manner.

---

## 1. Solving a Migration Problem: From CSV to JSON

### Typical Migration Challenge

You have legacy data in CSV files that need to be transformed into structured JSON outputs. You may need to:

- Combine multiple CSV files (products, inventory, pricing) into a single enriched JSON dataset.
- Filter and enrich data according to certain conditions (e.g., only recent products, category-specific filters).
- Aggregate data across multiple dimensions (e.g., summary of inventory by warehouse).
- Produce multiple output formats (e.g., JSON arrays, aggregated JSON objects, CSV exports for simplified lists).

### Approach

This framework abstracts the process into three main steps:

1. **Input Specification**: Describe your CSV inputs, including columns, types, validation rules, and patterns.

2. **Transformations (OmniLang/CSON)**: Define transformations in a YAML file (`transformations.yaml`) that specify:
   - Which datasets to read from (`from` fields).
   - How to filter rows (`filter` expressions).
   - How to map rows to objects (`item_structure`).
   - How to aggregate and group data (`group_by` and aggregator functions).

3. **Outputs**: Describe how transformed data should be written out (JSON or CSV), optionally apply post-processing (compression, notifications), and specify splitting logic for multiple output files.

By externalizing all logic into `transformations.yaml` and macros, you can quickly adjust the logic without modifying Python code.

---

## 2. OmniLang/CSON Language Guide

OmniLang/CSON is a domain-specific, JSON/YAML-friendly language for data transformations. It’s declarative: you declare what you want done (filters, transformations) rather than how to do it imperatively. Key features:

### Expressions

- **Variables**: Access input fields with `$field_name`.
- **Parameters**: Access user-defined parameters (from `transformations.yaml` or passed in code) using `$parameter_name`.
- **Functions**: Invoke built-in or plugin-registered functions like `concat()`, `format_date()`, `if()`.
- **Operators**: Support arithmetic (`+`, `-`, `*`, `/`), comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`), and logical operators (`and`, `or`, `not`).

### Macros

- **Macros**: Define reusable pieces of logic, e.g.:
  ```yaml
  macros:
    compute_area: "= $width * $height"
    is_recent: "= $release_date >= $date_cutoff"
    large_item: "=if($width * $height > 50, true, false)"
  ```
  These can be referenced in filters or item structures (e.g., `=compute_area()`).

### Data Structures

- **Objects**: Transformations can produce objects with `fields`.
- **Arrays**: Use `type: array` to map rows of an input dataset or a grouped subset into arrays of items.
- **Grouping and Aggregations**: Use `group_by` to aggregate rows and aggregator functions like `=sum(column)`, `=min(column)`, etc.

### Conditions and Filters

- **Filters**: Apply filters on arrays, e.g., `filter: "= $width > 0 and $height > 0 and is_recent"`.
- **Conditional Fields**: Use `if()` function inside item structures to set fields conditionally.

### Example Snippet

```yaml
transformations:
  products:
    type: array
    from: products
    filter: "= $width > 0 and $height > 0 and is_recent"
    item_structure:
      productId: "$product_id"
      dimensions:
        width: "$width"
        height: "$height"
        area: "=compute_area()"  # expands to "= $width * $height"
      attributes:
        category: "$category"
        largeItemFlag: "=large_item"
```

This snippet reads from the `products` dataset, filters products by width/height and recency, and builds an array of product objects. The `area` and `largeItemFlag` fields use macros defined earlier.

---

## 3. Managing and Running Projects

### Project Structure

A project directory typically looks like this:

```
project_name/
    input_data/
        product_data_2021.csv
        product_data_2022.csv
        inventory_data.csv
        pricing_data.csv
    output_data/
    transformations.yaml
    plugins/
        custom_functions.py
```

**Key Files:**

- `transformations.yaml`: Defines inputs, transformations, outputs, macros, and parameters.
- `input_data/`: Contains all CSV files matching patterns in `transformations.yaml`.
- `output_data/`: Where the transformed JSON/CSV results will be written.
- `plugins/`: (Optional) Python files that register custom functions available in transformations.

### Configuring Inputs

In `transformations.yaml`, specify inputs:

```yaml
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
    validate:
      required_columns: ["product_id", "width", "height", "category"]
```

### Defining Transformations

In `transformations.yaml` under `transformations:` define objects and arrays. Examples:

- A top-level object containing metadata.
- An array transformation producing enriched product data.
- Another array producing a summarized inventory report.

### Specifying Outputs

Under `outputs:` define how and where to write data:

```yaml
outputs:
  - name: "enriched_products_output"
    root: "products"
    format: "json"
    file: "enriched_products.json"
    pretty: true
    post_process:
      - type: "compress"
        format: "gzip"
```

You can split outputs by a field in the transformed data, produce CSV outputs, etc.

### Running a Project

Use the `main.py` script to process a project. You can run:

```bash
python main.py --project sample_project2
```

The script:

- Loads `transformations.yaml`.
- Validates structure.
- Loads CSV inputs.
- Applies transformations.
- Writes outputs to `output_data/`.

If everything is configured correctly, you'll find JSON/CSV outputs generated according to your transformation rules.

### Error Handling

The `transformations.yaml` can define error handling rules for missing columns, type mismatches, etc. By default, critical issues cause the script to exit. Warnings may log and continue processing. Customize these behaviors as needed:

```yaml
error_handling:
  on_missing_column: "error"
  on_type_mismatch: "warn"
  on_pattern_violation: "error"
  on_function_failure: "warn"
```

---

## Conclusion

This framework provides a scalable, declarative approach to migrating CSV data to JSON. With OmniLang/CSON, transformations are flexible and maintainable. By carefully defining inputs, transformations, and outputs, you can solve migration challenges efficiently and adapt quickly to evolving requirements.