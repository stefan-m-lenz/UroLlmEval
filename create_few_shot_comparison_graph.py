import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from results_helper import (
    extract_model_name_from_filename, extract_prompt_type_from_filename, extract_step_from_filename,
    ensure_directory, get_metric_label, get_model_label)


# Define functions to extract shot number and example type from prompt_type
# This currently works only for step 1.
# Probably this plot also does not make so much sense for the other steps
def get_example_number_from_prompt_type(prompt_type):
    return int(prompt_type[0])

def get_example_type_from_prompt_type(prompt_type):
    if prompt_type.endswith("uro"):
        return "Uro"
    elif prompt_type.endswith("gyn"):
        return "Gyn"
    else:
        return "None"


def create_few_shot_comparison_graph(analysis_file, output_dir, models, grid, size_cm, font_size, legend_margin, wspace, hspace, margin_left, margin_right):
    step = extract_step_from_filename(analysis_file)

    # Read the CSV file into a DataFrame
    analysis_df = pd.read_csv(analysis_file)

    analysis_df["model_id"] = analysis_df['results_file'].apply(lambda filename: extract_model_name_from_filename(filename, step))
    analysis_df["model"] = analysis_df["model_id"].apply(get_model_label)

    if models:
        # filter using the models argument
        models_not_in_analysis_df = set(models) - set(analysis_df["model_id"])
        if models_not_in_analysis_df:
            raise ValueError("There are models specified that do not occur in the analysis results: " + (", ").join(models_not_in_analysis_df))
        analysis_df = analysis_df[analysis_df["model_id"].isin(models)]
        analysis_df['model'] = pd.Categorical(analysis_df['model'], categories=[get_model_label(m) for m in models], ordered=True)
    else:
        models = analysis_df["model"].unique()
    model_labels = [get_model_label(model) for model in models]

    analysis_df["prompt_type"] = analysis_df['results_file'].apply(extract_prompt_type_from_filename)

    analysis_df["example_number"] = analysis_df["prompt_type"].apply(get_example_number_from_prompt_type)
    analysis_df["example_type"] = analysis_df["prompt_type"].apply(get_example_type_from_prompt_type)

    # Set global font size and font
    plt.rcParams.update({'font.size': font_size, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

    # Define color mapping for example types
    RED = '#C1002B'
    BLUE = '#003C76'
    color_map = {"Uro": BLUE, "Gyn": RED, "None": "black"}

    # Plotting
    metrics = ['p_correct_total', 'p_na', 'recall', 'specificity']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(*grid, figsize=[s * cm for s in size_cm])

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    first_plot = True
    for ax, metric in zip(axes, metrics):
        for model, marker in zip(model_labels, markers):
            model_data = analysis_df[analysis_df['model'] == model]

            # Plot grey lines between zero and two shot numbers for each model
            zero_shot_data = model_data[model_data['example_number'] == 0]
            two_shot_data = model_data[model_data['example_number'] == 2]
            if not zero_shot_data.empty and not two_shot_data.empty:
                for _, zero_row in zero_shot_data.iterrows():
                    for _, two_row in two_shot_data.iterrows():
                        ax.plot([zero_row['example_number'], two_row['example_number']],
                                [zero_row[metric], two_row[metric]], color='grey', linestyle='--', linewidth=1)

            for example_type, color in color_map.items():
                type_data = model_data[model_data['example_type'] == example_type]
                type_data = type_data.sort_values(by='example_number')  # Sort by example_number for correct line plotting
                ax.plot(type_data['example_number'], type_data[metric], color=color, linestyle='-', linewidth=1)  # Plot lines
                ax.scatter(type_data['example_number'], type_data[metric], color=color, marker=marker, s=80,
                           label=f'{model} - {example_type}')

        if metric == "p_na":
            metric_label = "Proportion of unusable values"
        else:
            metric_label = get_metric_label(metric)

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        ax.set_xlabel('Number of examples')
        if grid[0] == 1:
            ax.set_title(metric_label, pad=15)
            if first_plot:
                ax.set_ylabel("Value of metric")
        else:
            ax.set_ylabel(metric_label)

        ax.set_xticks([0, 2, 4, 6])

        first_plot = False

    # Create legends
    # Legend for models
    model_handles = [
        plt.Line2D([0], [0], marker=marker, color='w', label=model_label, markerfacecolor='none', markeredgecolor='black',
                   markersize=round(0.8*font_size))
        for model_label, marker in zip(model_labels, markers)
    ]
    # Legend for example types
    type_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=example_type, markerfacecolor=color, markersize=round(0.8*font_size))
        for example_type, color in color_map.items()
    ]

    # Add legends to figure
    # Legend for models
    model_legend = fig.legend(handles=model_handles, loc="lower center", ncol=5, title="Models", bbox_to_anchor=(0.5, 0.075))
    model_legend.get_title().set_fontweight('bold')
    frame = model_legend.get_frame()
    frame.set_edgecolor('#003C76')

    # Legend for example types
    example_type_legend=fig.legend(handles=type_handles, loc="lower center", ncol=3, title="Example Types", bbox_to_anchor=(0.5, 0.015))
    example_type_legend.get_title().set_fontweight('bold')
    frame = example_type_legend.get_frame()
    frame.set_edgecolor('#003C76')

    plt.subplots_adjust(left=margin_left, right=1-margin_right, bottom=legend_margin, wspace=wspace, hspace=hspace)  # Adding margin at the bottom for legends

    filename = "few_shot_comparison"
    filepath = os.path.join(output_dir, f"{filename}.png")
    print(f"Created plot: {filepath}")
    plt.savefig(filepath, format="png")
    filepath = os.path.join(output_dir, f"{filename}.svg")
    plt.savefig(filepath, format="svg")


#create_few_shot_comparison_graph(r"C:\Users\lenzstef\Desktop\output\analysis_step1.csv", "output")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create plot for comparison of effect of few-shot prompting")
    parser.add_argument("--input-file", type=str, help="Input file with analysis results", required=True)
    parser.add_argument("--output-dir", type=str, help="Output path for plot", required=True)
    parser.add_argument("--models", nargs='+', default=[], help="Names of models to display in the plot.")
    parser.add_argument("--grid", nargs=2, type=int, metavar=("ROWS", "COLS"),
                        help="The number of rows and columns in the subplot grid.", default=(2, 2))
    parser.add_argument("--size", nargs=2, type=float, metavar=("ROWS", "COLS"),
                        help="The plot size in cm.", default=(20,22))
    parser.add_argument("--font-size", type=int, help="Font size for plot text", default=12)
    parser.add_argument("--legend-margin", type=float, help="Margin for legend", default=0.2)
    parser.add_argument("--wspace", type=float, help="Horizontal distance between plots", default=0.2)
    parser.add_argument("--hspace", type=float, help="Vertical distance between plots", default=0.2)
    parser.add_argument("--margin-left", type=float, help="Left margin ", default = 0.05)
    parser.add_argument("--margin-right", type=float, help="Right margin ", default = 0.05)

    args = parser.parse_args()

    ensure_directory(args.output_dir)

    create_few_shot_comparison_graph(args.input_file, args.output_dir,
                                     models=args.models,
                                     grid=args.grid, size_cm=args.size, font_size=args.font_size,
                                     legend_margin=args.legend_margin, wspace=args.wspace,
                                     hspace=args.hspace,
                                     margin_left=args.margin_left,
                                     margin_right=args.margin_right)



# Command line example:
# python.exe .\UroLlmEval\create_few_shot_comparison_graph.py --input-file "C:\Users\lenzstef\Desktop\output\analysis_step1.csv" --output-dir output --grid 1 4 --size 51 17 --font-size 20 --legend-margin 0.33 --wspace 0.25 --models mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct  mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct

# Paper-Plot:
#python.exe .\UroLlmEval\create_few_shot_comparison_graph.py --input-file "C:\Users\lenzstef\Desktop\output\analysis_step1.csv" --output-dir output --grid 2 2 --font-size 20 --models mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct  mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct --size 39 35 --wspace 0.2 --legend-margin 0.18 --margin-left 0.07 --margin-right 0.01 --hspace 0.27