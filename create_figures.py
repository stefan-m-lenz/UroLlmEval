import os
from create_summary_plot import create_summary_plot
from create_few_shot_comparison_graph import create_few_shot_comparison_graph

# Define paths
results_folder = r"output"  # Folder with results from the experiments
output_dir = os.path.join(results_folder, "summary", "plots")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Summary plot for Step 1
create_summary_plot(
    results_folder=results_folder,
    step=1,
    bar_params=["recall", "specificity", "p_correct_total", "p_na"],
    output_folder=output_dir,
    x_start=1.15,
    y_start=2.35,
    spacing=0.4,
    fontsize_legend=21,
    legend_border_width=1.1,
    legend_border_height=3.4,
    legend_border_linewidth=1,
    legend_border_x=0.2,
    legend_border_y=-0.6,
    bar_height=0.35,
    fontsize=21.0,
    plot_width=8.5,
    plot_height=18.0,
    model_offset=0.0,
    models_to_display=["PAPER_MODELS"],
)


create_few_shot_comparison_graph(
    analysis_file=os.path.join(results_folder, "analysis_step1.csv"),
    output_dir=output_dir,
    models=[
        "meta-llama_Llama-3.2-1B-Instruct",
        "utter-project_EuroLLM-1.7B-Instruct",
        "meta-llama_Llama-3.2-3B-Instruct",
        "mistralai_Mistral-7B-Instruct-v0.3",
        "BioMistral_BioMistral-7B",
        "LeoLM_leo-hessianai-7b-chat",
        "meta-llama_Meta-Llama-3.1-8B-Instruct",
        "mistralai_Mistral-Nemo-Instruct-2407",
        "mistralai_Mixtral-8x7B-Instruct-v0.1",
        "meta-llama_Meta-Llama-3.1-70B-Instruct"
    ],
    grid=(2, 2),
    size_cm=(42, 45),
    font_size=20,
    legend_margin=0.23,
    wspace=0.2,
    hspace=0.25,
    margin_left=0.07,
    margin_right=0.01
)

# Summary plots for Step 2
create_summary_plot(
    results_folder=results_folder,
    step=2,
    bar_params=[
        "p_all_diagnoses_found_in_snippets_with_diagnoses",
        "p_no_other_diagnoses_found_for_snippet",
        "p_snippets_correct"
    ],
    output_folder=output_dir,
    x_start=-1.1,
    y_start=-1,
    spacing=0.35,
    fontsize_legend=20,
    legend_border_width=2.075,
    legend_border_height=2,
    legend_border_linewidth=1,
    legend_border_x=0.2,
    legend_border_y=-2.65,
    fontsize=20,
    plot_width=10.5,
    plot_height=16,
    model_offset=0.15,
    models_to_display=["PAPER_MODELS"],
)

# Summary plots for Step 3
create_summary_plot(
    results_folder=results_folder,
    step=3,
    bar_params=["p_correct", "p_correct_total", "p_na"],
    output_folder=output_dir,
    x_start=-1.4,
    y_start=-0.95,
    spacing=0.4,
    fontsize_legend=20,
    legend_border_width=2.66,
    legend_border_height=2.175,
    legend_border_linewidth=1,
    legend_border_x=0.2,
    legend_border_y=-2.775,
    fontsize=20,
    plot_width=9.5,
    plot_height=17,
    model_offset=0.12,
    models_to_display=["PAPER_MODELS"],
)
