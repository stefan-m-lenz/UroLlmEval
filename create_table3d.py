import os
from jinja2 import Environment, FileSystemLoader
import pandas as pd


def transform_data_for_jinja_template(df):
    if pd.api.types.is_categorical_dtype(df["category"]):
        all_categories = df["category"].cat.categories # Get unique categories in pre-defined sorted order
    else:
        all_categories = sorted(df['category'].unique())  # Get unique categories in sorted order

    processed_data = []

    # Group the DataFrame by 'group' and 'subgroup'
    grouped = df.groupby(['group', 'subgroup'], observed=True)

    # Collect all unique groups and subgroups
    for group_name, group_data in grouped:
        group, subgroup = group_name
        category_values = {category: '' for category in all_categories}
        for _, row in group_data.iterrows():
            formatted_value = "{:.2f}".format(row['value'])
            if formatted_value == "nan":
                formatted_value = "NaN"
            category_values[row['category']] = formatted_value

        # Check if the group already exists in processed_data
        group_entry = next((item for item in processed_data if item["name"] == group), None)
        if not group_entry:
            group_entry = {"name": group, "rowspan": 0, "subgroups": []}
            processed_data.append(group_entry)

        subgroup_entry = {"name": subgroup, "categories": category_values}
        group_entry["subgroups"].append(subgroup_entry)

    # Calculate rowspans for each group
    for group_entry in processed_data:
        group_entry["rowspan"] = len(group_entry["subgroups"])

    return processed_data, all_categories


def render_table3d(df, output_file):
    # Template file is on same path as the script:
    template_path = os.path.dirname(os.path.abspath(__file__))
    # Load the template
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template('table3d.html')

    # Transform input data and render it
    processed_data, all_categories = transform_data_for_jinja_template(df)
    html_content = template.render(data=processed_data, categories=all_categories)

    # Write to HTML file
    with open(output_file, "w") as file:
        file.write(html_content)


# Example
# data = {
#     "group": ["Group 1", "Group 1", "Group 1", "Group 1", "Group 2", "Group 2"],
#     "subgroup": ["Subgroup 1.1", "Subgroup 1.1", "Subgroup 1.2", "Subgroup 1.2", "Subgroup 2.1", "Subgroup 2.1"],
#     "category": ["Category A", "Category B", "Category A", "Category B", "Category A", "Category B"],
#     "value": [10, 15, 20, 25, 30, 35]
# }

# render_table3d(df=pd.DataFrame(data), output_file="UroLlmEval/output/table3dtest.html")