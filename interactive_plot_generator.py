import os
import json
from jinja2 import Environment, FileSystemLoader
from create_summary_plot import preprocess_data, get_analysis_file_path, split_df, find_best_result, find_best_results_df
from results_helper import MODEL_LABELS, extract_model_name_from_filename

def get_best_per_prompt(df):
    idx = df.groupby(["model", "prompt_type"])["p_snippets_correct"].idxmax()
    return df.loc[idx]

def create_interactive_plot(step: int, models_to_display: list[str] = ["PAPER_MODELS"]) -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "interactive_plots"))
    data_dir = os.path.join(base_dir, "data")
    template_dir = os.path.join(base_dir, "templates")
    libs_dir = os.path.join(base_dir, "libs")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    if step == 1:
        csv_file = "analysis_step1.csv"
        template_name = "template_step_1.html"
        output_name = "Figure S1.html"
    elif step == 2:
        csv_file = "analysis_step2.csv"
        template_name = "template_step_2.html"
        output_name = "Figure S2.html"
    elif step == 3:
        csv_file = "analysis_step3.csv"
        template_name = "template_step_3.html"
        output_name = "Figure S3.html"
    else:
        raise ValueError(f"Unknown Step: {step}. Only 1, 2 or 3 valid.")

    analysis_path = os.path.join(data_dir, csv_file)
    df, unique_models = preprocess_data(analysis_path, models_to_display=models_to_display)

    if step == 2:
        idx = df.groupby(["model", "prompt_type"])["p_snippets_correct"].idxmax()
        df = df.loc[idx]
    elif step == 3:
        idx_best = df.groupby(["model", "prompt_type"])["p_correct"].idxmax()
        df_data = df.loc[idx_best].reset_index(drop=True)

        best_models = {}
        for step_i in range(1, 3):
            best_result = find_best_result(get_analysis_file_path(data_dir, step_i), models=best_models)
            best_models[step_i] = extract_model_name_from_filename(best_result, step_i)

        _, best_model_prev_step_df = split_df(df, step=3)
        best_df = find_best_results_df(best_model_prev_step_df, step=3, best_models=best_models)
        levenshtein_best_df = best_df[best_df["model"] == "levenshtein-regex"]

        df = df_data

    data_dict = {"data": df.to_dict(orient="records"), "unique_models": unique_models, **({"levenshtein_regex": levenshtein_best_df.to_dict(orient="records")} if step == 3 else {})}
    prepared_data = json.dumps(data_dict)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    with open(os.path.join(libs_dir, "chart.min.js"), encoding="utf-8") as f:
        chart_js_content = f.read()
    with open(os.path.join(libs_dir, "sortable.min.js"), encoding="utf-8") as f:
        sortable_js_content = f.read()
    with open(os.path.join(libs_dir, "fontawesome-all.min.js"), encoding="utf-8") as f:
        fontawesome_js_content = f.read()

    context = {
        "prepared_data": prepared_data,
        "chart_js_content": chart_js_content,
        "sortable_js_content": sortable_js_content,
        "fontawesome_js_content": fontawesome_js_content,
        "model_labels": MODEL_LABELS
    }

    rendered = template.render(**context)
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Interactive Plot for step {step} generated: {output_path}")


def main():
    create_interactive_plot(step=1)
    create_interactive_plot(step=2)
    create_interactive_plot(step=3)


if __name__ == "__main__":
    main()
