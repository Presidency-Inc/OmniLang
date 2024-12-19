def register_functions(func_dict):
    # Provide a custom function format_color that uppercase a color and prefix it
    def format_color(col):
        return f"C-{str(col).upper()}"
    func_dict['format_color'] = format_color
