import os
import json
from jinja2 import Environment, FileSystemLoader
from data_processing_module import preprocess_data

def main():
    current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(current_dir, "data")
    template_dir = os.path.join(current_dir, "templates")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    analysis_file_path = os.path.join(data_dir, "analysis_step1.csv")

    analysis_df, unique_models = preprocess_data(
        analysis_file_path,
        models_to_display=["PAPER_MODELS"])

    data_dict = {
        "data": analysis_df.to_dict(orient="records"),
        "unique_models": unique_models
    }

    prepared_data_json = json.dumps(data_dict)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("template_step_1.html")

    rendered_html = template.render(prepared_data=prepared_data_json)

    output_html_path = os.path.join(output_dir, "step_1_interactive_plot_standalone.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"Standalone HTML file generated: {output_html_path}")

if __name__ == "__main__":
    main()
