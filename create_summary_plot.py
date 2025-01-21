import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import re
import argparse
import glob

from find_best_result import find_best_result_in_df, find_best_result
from results_helper import (
    extract_model_name_from_filename, extract_step_from_filename, extract_prompt_type_from_filename, get_model_label, get_metric_label,
    ensure_directory, get_paper_models, order_models)
from create_tables import find_model_without_other_model_in_previous_step


#Corporate Design Farben
SCHWARZ = '#000000'
ROT = '#C1002B'  # R:193, G:0, B:43
DUNKELBLAU = '#003C76'  # R:0, G:60, B:118
HELLBLAU = '#80A1C9'  # R:128, G:161, B:201
HELLBLAU_BALKEN = '#A0C1E9'
LILA = '#8800ff'
BRAUN = '#bf8734'

#matcht prompt_type aus df passend zur Legende
def get_number_and_color(prompt_type, step):
    if step == 1:
        if '0' in prompt_type:
            return 0, SCHWARZ, 0
        elif '2-gyn' in prompt_type:
            return 2, ROT, 0
        elif '2-uro' in prompt_type:
            return 2, DUNKELBLAU, 0
        elif '4-gyn' in prompt_type:
            return 4, ROT, 0
        elif '4-uro' in prompt_type:
            return 4, DUNKELBLAU, 0
        elif '6-gyn' in prompt_type:
            return 6, ROT, 0
        elif '6-uro' in prompt_type:
            return 6, DUNKELBLAU, 0
        else:
            return None, SCHWARZ, 0

    elif step == 2:
        if '2-uro-ctx' in prompt_type:
            return 2, DUNKELBLAU, 1
        elif '2-uro' in prompt_type:
            return 2, DUNKELBLAU, 0
        elif '2-gyn-ctx' in prompt_type:
            return 2, ROT, 1
        elif '2-gyn' in prompt_type:
            return 2, ROT, 0
        elif '0-uro-ctx' in prompt_type:
            return 0, DUNKELBLAU, 1
        elif '0-uro' in prompt_type:
            return 0, DUNKELBLAU, 0
        elif '0-gyn-ctx' in prompt_type:
            return 0, ROT, 1
        elif '0-gyn' in prompt_type:
            return 0, ROT, 0
        else:
            return None, SCHWARZ, 0

    else: #step==3
        if '0-dates' in prompt_type:
            return 0, LILA, 1
        elif '0-verify' in prompt_type:
            return 0, BRAUN, 0
        elif '4-verify' in prompt_type:
            return 4, BRAUN, 0
        elif '0' in prompt_type:
            return 0, LILA, 0
        elif '3-dates' in prompt_type:
            return 3, LILA, 1
        elif '3' in prompt_type:
            return 3, LILA, 0
        else:
            return None, SCHWARZ, 0

def calculate_label_offsets(bar_params, unique_models):
    max_label_length = max(len(get_metric_label(param)) for param in bar_params)
    max_model_name_length = max(len(get_model_label(model)) for model in unique_models)
    label_offset = max(0.1, 0.009 * max_label_length)
    model_offset = max_model_name_length * 0.01 + 0.05
    return label_offset, model_offset

#erstellt datasplits für step 2 und 3
def split_df(analysis_df, step):
    # fügt Spalte 'model_pev_step' hinzu
    analysis_df["model_prev_step"] = analysis_df["results_file"].apply(lambda filename: extract_model_name_from_filename(filename, step - 1))

    #finde Model und seine Daten, dass in beide Splits muss
    best_model = find_model_without_other_model_in_previous_step(analysis_df)
    #print(f'best: {best_model}')
    best_model_df = analysis_df[analysis_df['model'] == best_model]

    # Modelle, die im vorherigen Schritt sich selbst verwendet haben
    same_model_prev_step = analysis_df["model_prev_step"] == analysis_df["model"]
    # Modelle, die im vorherigen Schritt nicht sich selbst verwendet haben
    different_model_prev_step = ~same_model_prev_step

    if step == 3:
        levenshtein_regex_df = analysis_df[analysis_df['model'] == 'levenshtein-regex']
        same_model_prev_step_df = pd.concat([analysis_df[same_model_prev_step], levenshtein_regex_df])
    else:
        same_model_prev_step_df = analysis_df[same_model_prev_step]
     #step==2
    best_model_prev_step_df = pd.concat([analysis_df[different_model_prev_step], best_model_df])

    return same_model_prev_step_df, best_model_prev_step_df

#Achsen für Plot
def add_axes(ax, label_offset, max_model_name_length):
    ax.set_yticks([]) #keine y-Achsen ticks
    ax.set_yticklabels([]) #keine y-Achsenbeschriftung
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) #x-Achse oben
    ax.set_xlim(-label_offset - max_model_name_length * 0.01 - 0.15, 1) # x-Achse dyn nach Platzbedarf
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels([0.0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0], fontname='Helvetica')
    ax.spines['top'].set_bounds(0, 1)
    ax.spines['left'].set_position(('data', 0-0.01)) # Verschiebt die y-Achse nach rechts
    ax.spines['left'].set_visible(False)  # Linie links entfernen
    ax.spines['bottom'].set_visible(False)  # Linie unten entfernen
    ax.spines['right'].set_position(('data', 1+0.1))
    ax.spines['right'].set_visible(False)  # Linie rechts entfernen

#Balken für Plot
def add_bars(ax, y, best_df, unique_models, bar_params, bar_height, bar_gap, barcolor, plot_height, step):
    model_to_y = {model: i for i, model in enumerate(unique_models)}

    #Berechnet Platzbedarf um Balken entsprechend zu positionieren
    total_height_per_model = bar_height * len(bar_params) + bar_gap * (len(bar_params) - 1)
    total_height_needed = total_height_per_model * len(unique_models)
    if total_height_needed > plot_height:
        print(f"Warning: The total bar height ({total_height_needed}) exceeds the available plot height ({plot_height}). "
              "This leads to compression of the plot. "
              "Consider resizing the plot or reducing the bar_height.")

    y = np.arange(len(unique_models)) * total_height_per_model

    for i, param in enumerate(bar_params):
        label = get_metric_label(param)
        if label == 'Accuracy (NA=wrong)' and step == 1:
            label = 'Accuracy'
        for model in unique_models:
            model_data = best_df[best_df['model'] == model]
            if not model_data.empty:
                y_position = y[model_to_y[model]] + bar_height * (len(bar_params) - 1) / 2 - bar_height * i
                x_value = model_data[param].iloc[0]
                ax.barh(y_position, x_value, bar_height - bar_gap, color=barcolor, label=label)

#Labels = Modelnamen + Balken Label für Plot
def add_labels(ax, y, best_df, unique_models, bar_params, bar_height, label_offset, model_offset, fontsize, bar_gap, step):
    total_height_per_model = bar_height * len(bar_params) + (bar_gap * (len(bar_params) - 1))
    y = np.arange(len(unique_models)) * total_height_per_model

    for i, model in enumerate(unique_models):
        cat = get_model_label(model)
        formatted_modelname = r'$\textbf{' + cat + '}$'

        #Modelnamen-Positionierung mittig seiner Balken Label positionieren
        if len(bar_params) % 2 == 1:
            middle_index = len(bar_params) // 2
            middle_label_y_pos = y[i] + bar_height * (len(bar_params) - 1) / 2 - bar_height * middle_index
        else:
            middle_index = len(bar_params) // 2
            middle_label_y_pos = (y[i] + bar_height * (len(bar_params) - 1) / 2 -
                                  (bar_height * (middle_index - 0.5)))

        for j, param in enumerate(bar_params):
            label = get_metric_label(param)
            if label == 'Accuracy (NA=wrong)' and step == 1:
                label = 'Accuracy'
            y_pos = y[i] + bar_height * (len(bar_params) - 1) / 2 - bar_height * j
            ax.text(-0.02, y_pos, label, ha='right', va='center')
        ax.text(-label_offset - model_offset, middle_label_y_pos + 0.01, formatted_modelname, ha='right', va='center', fontweight='bold', fontname='Helvetica', fontsize=fontsize)

def add_markers(ax, y, analysis_df, unique_models, bar_params, bar_height, bar_gap):
    total_height_per_model = bar_height * len(bar_params) + (bar_gap * (len(bar_params) - 1))
    y = np.arange(len(unique_models)) * total_height_per_model

    for _, row in analysis_df.iterrows():
        model_index = list(unique_models).index(row['model'])
        number, dot_color, line = get_number_and_color(row['prompt_type'], row['step'])
        if number is not None:
            for j, param in enumerate(bar_params):
                y_pos = y[model_index] + bar_height * (len(bar_params) - 1) / 2 - bar_height * j
                x_pos = row[param]
                formatted_number = str(number)
                if line == 1:
                    if row['step'] == 2:
                        formatted_number = r"$\overline{\mathsf{" + formatted_number + "}}$"
                        y_pos -= 0.05
                        ax.text(x_pos, y_pos, formatted_number, ha='center', va='center', color=dot_color)
                    elif row['step'] == 3:
                        formatted_number = r"$\underline{\mathsf{" + formatted_number + "}}$"
                        y_pos -= 0.02
                        ax.text(x_pos, y_pos, formatted_number, ha='center', va='center', color=dot_color)
                else:
                    formatted_number = r"$\mathsf{"  + formatted_number + "}$"
                    if row['step'] == 1 or row['step'] == 2:
                        ax.text(x_pos, y_pos - 0.05, formatted_number, ha='center', va='center', color=dot_color)
                    else:
                        ax.text(x_pos, y_pos - 0.02, formatted_number, ha='center', va='center', color=dot_color)

#fügt Legende hinzu
def add_legend(ax, step, x_start, y_start, spacing, fontsize_legend, legend_border_width, legend_border_height, legend_border_linewidth, legend_border_x, legend_border_y):
    if legend_border_x == 0.2:
        legend_border_x = x_start - 0.02

    rect = patches.Rectangle(
        (legend_border_x, legend_border_y),
        width=legend_border_width,
        height=legend_border_height,
        linewidth=legend_border_linewidth,
        edgecolor=DUNKELBLAU,
        facecolor='none',
        clip_on=False
    )

    ax.add_patch(rect)

    if step == 1:
        ax.text(x_start, y_start, r'$\textbf{' + 'Prompting variants:' + '}$', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing, r'$\mathsf{0}$ Zero-shot prompting without examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 2, r'$\mathsf{2}$ Few-shot prompting with two examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 3, r'$\mathsf{4}$ Few-shot prompting with four examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 4, r'$\mathsf{6}$ Few-shot prompting with six examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.plot(x_start + 0.015, y_start - spacing * 5 - 0.17, 'bs', markersize=8, clip_on=False)
        ax.text(x_start + 0.05, y_start - spacing * 5 - (0.01 * fontsize_legend) , 'Ficticious examples from urology', ha='left', va='center', fontsize=fontsize_legend)
        ax.plot(x_start + 0.015, y_start - spacing * 6 - 0.17, 'rs', markersize=8, clip_on=False)
        ax.text(x_start + 0.05, y_start - spacing * 6 - (0.01 * fontsize_legend), 'Ficticious examples from gynecology', ha='left', va='center', fontsize=fontsize_legend)

    elif step == 2:
        ax.text(x_start, y_start, r'$\textbf{' + 'Prompting variants:' + '}$', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing, r'$\mathsf{0}$ Zero-shot prompting with two ICD-10 codes as examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 2, r'$\mathsf{\overline{0}}$ Zero-shot prompting with two examples + text context', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 3, r'$\mathsf{2}$ Few-shot prompting with two ICD-10 codes as examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 4, r'$\mathsf{\overline{2}}$ Few-shot prompting with two examples + text context' , ha='left', va='center', fontsize=fontsize_legend)
        ax.plot(x_start + (1/16 * fontsize_legend) + 0.015, y_start - spacing, 'bs', markersize=8, clip_on=False)
        ax.text(x_start + (1/16 * fontsize_legend) + 0.05, y_start - spacing, 'Ficticious examples from urology', ha='left', va='center', fontsize=fontsize_legend)
        ax.plot(x_start + (1/16 * fontsize_legend) + 0.015, y_start - spacing * 2, 'rs', markersize=8, clip_on=False)
        ax.text(x_start + (1/16 * fontsize_legend) + 0.05, y_start - spacing * 2, 'Ficticious examples from gynecology', ha='left', va='center', fontsize=fontsize_legend)

    elif step == 3:
        ax.text(x_start, y_start, r'$\textbf{' + 'Prompting variants:' + '}$', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing, r'$\mathsf{0}$', ha='left', va='center', fontsize=fontsize_legend, color=LILA)
        ax.text(x_start + 0.03, y_start - spacing, 'Zero-shot prompting', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 2, r'$\underline{\mathsf{0}}$', ha='left', va='center', fontsize=fontsize_legend, color=LILA)
        ax.text(x_start + 0.03, y_start - spacing * 2, 'Zero-shot prompting using dates filtered with regular expressions', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 3, r'$\mathsf{3}$', ha='left', va='center', fontsize=fontsize_legend, color=LILA)
        ax.text(x_start + 0.03, y_start - spacing * 3, 'Few-shot prompting with three ficticious examples', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start, y_start - spacing * 4, r'$\underline{\mathsf{3}}$', ha='left', va='center', fontsize=fontsize_legend, color=LILA)
        ax.text(x_start + 0.03, y_start - spacing * 4, 'Few-shot prompting using dates filtered with regular expressions', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start +(1/13 * fontsize_legend) , y_start - spacing, r'$\mathsf{0}$', ha='left', va='center', fontsize=fontsize_legend, color=BRAUN)
        ax.text(x_start + 0.03 +(1/13 * fontsize_legend) , y_start - spacing, 'Filter dates and verify using zero-shot prompting', ha='left', va='center', fontsize=fontsize_legend)
        ax.text(x_start +(1/13 * fontsize_legend) , y_start - spacing * 2, r'$\mathsf{4}$', ha='left', va='center', fontsize=fontsize_legend, color=BRAUN)
        ax.text(x_start + 0.03 +(1/13 * fontsize_legend) , y_start - spacing * 2, 'Filter dates and verify using few-shot prompting', ha='left', va='center', fontsize=fontsize_legend)

#speichern als png
def save_plot_to_file(output_folder, filename):
    if output_folder and filename:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #filename = filename.replace(".png", "_just_best.png")
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Created plot: {filepath}")

        filepath_svg = re.sub(".png$", ".svg", filepath)
        plt.savefig(filepath_svg, format = "svg", bbox_inches='tight')


#Schriftart zu Helvetica und Text texen
def configure_fonts(fontsize):
    rcParams['font.family'] = 'sans-serif'
    rcParams["font.size"] = fontsize
    rcParams['font.sans-serif'] = ['Helvetica']
    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = "Helvetica"
    plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    plt.rc("text.latex", preamble=r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")

#kompletten Plot generieren
def generate_plot(analysis_df, unique_models, bar_params, bar_height, best_df, bar_gap, barcolor, fontsize, step, x_start, y_start, spacing,
                  fontsize_legend, legend_border_width, legend_border_height, legend_border_linewidth, legend_border_x, legend_border_y,
                  plot_width=16, plot_height=9, model_offset=None, output_folder=None, filename=None):
    configure_fonts(fontsize)
    y = np.arange(len(unique_models))
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    label_offset, model_offset_calculated = calculate_label_offsets(bar_params, unique_models)
    model_offset = model_offset + model_offset_calculated

    add_axes(ax, label_offset, model_offset)
    add_bars(ax, y, best_df, unique_models, bar_params, bar_height, bar_gap, barcolor, plot_height, step)
    add_labels(ax, y, best_df, unique_models, bar_params, bar_height, label_offset, model_offset, fontsize, bar_gap, step)
    add_markers(ax, y, analysis_df, unique_models, bar_params, bar_height, bar_gap)
    add_legend(ax, step, x_start=x_start, y_start=y_start, spacing=spacing, fontsize_legend=fontsize_legend,
               legend_border_width=legend_border_width, legend_border_height=legend_border_height, legend_border_linewidth=legend_border_linewidth,
               legend_border_x=legend_border_x, legend_border_y=legend_border_y,)

    save_plot_to_file(output_folder, filename)

#dataframes erstellen
def preprocess_data(analysis_file_path, models_to_display):
    analysis_df = pd.read_csv(analysis_file_path)

    # fügt Spalte 'step', 'model' mit Modelnamen und 'prompt_type' hinzu
    analysis_df['step'] = analysis_df['results_file'].apply(extract_step_from_filename)
    if models_to_display and models_to_display[0] == "PAPER_MODELS":
        models_to_display = get_paper_models(analysis_df["step"][0])[::-1]
    analysis_df['model'] = analysis_df.apply(lambda row: extract_model_name_from_filename(row['results_file'], row['step']), axis=1)
    analysis_df['prompt_type'] = analysis_df['results_file'].apply(extract_prompt_type_from_filename)

    #definiert Modelnamen
    unique_models = order_models(analysis_df['model'].unique())
    if models_to_display:
        for model in models_to_display:
            if model not in unique_models:
                raise ValueError(f'{model} is not in csv. Check correct model name.')
        unique_models = models_to_display
    analysis_df = analysis_df[analysis_df['model'].isin(unique_models)]
    analysis_df.sort_values(by=['model'], inplace=True)

    return analysis_df, unique_models

#für step1 werden beste Ergebnisse über komplette csv datei gefunden
def find_best_results_step1(analysis_file_path, df):
    best_results_files = [find_best_result(analysis_file_path, models={1: model}) for model in df['model'].unique()]
    best_results_files = [result for result in best_results_files if result is not None]
    best_df = df[df['results_file'].isin(best_results_files)]
    return best_df

#für step 2 und 3 werden beste Ergebnisse über die jeweiligen dataframes der splits gefunden
def find_best_results_df(df, step, best_models={}):
    if best_models:
        def set_models(model):
            return {**best_models, step: model}
    else:
        def set_models(model):
            if model == "levenshtein-regex":
                return {3: "levenshtein-regex"}
            else:
                return {step_i: model for step_i in range(1, step+1)}
    best_results_files = [find_best_result_in_df(df, models=set_models(model)) for model in df['model'].unique()]
    best_results_files = [result for result in best_results_files if result is not None]
    best_df = df[df['results_file'].isin(best_results_files)]
    return best_df


def get_analysis_file_path(results_folder, step):
    return os.path.join(results_folder, f"analysis_step{step}.csv")


# main function
def create_summary_plot(
    results_folder: str,
    step: int,
    bar_params: list[str],
    output_folder: str,
    x_start: float = 1.0,
    y_start: float = -0.05,
    spacing: float = -0.1,
    fontsize_legend: float = 20.0,
    legend_border_width: float = 2.0,
    legend_border_height: float = 2.0,
    legend_border_linewidth: float = 2,
    legend_border_x: float = 0.2,
    legend_border_y: float = 0.2,
    models_to_display: list[str] = [],
    bar_height: float = 0.3,
    bar_gap: float = 0.02,
    barcolor: str = HELLBLAU_BALKEN,
    fontsize: float = 12.0,
    plot_width: float = 16.0,
    plot_height: float = 9.0,
    model_offset: float = 0.0,
):
    analysis_file_path = get_analysis_file_path(results_folder, step)
    models_to_display.reverse()
    analysis_df, unique_models = preprocess_data(analysis_file_path, models_to_display)
    if step == 1:
        best_df = find_best_results_step1(analysis_file_path, analysis_df)
        generate_plot(analysis_df=analysis_df,
                      unique_models=unique_models,
                      bar_params=bar_params,
                      bar_height=bar_height,
                      best_df=best_df,
                      bar_gap=bar_gap,
                      barcolor=barcolor,
                      fontsize=fontsize,
                      step=step,
                      x_start=x_start,
                      y_start=y_start,
                      spacing=spacing,
                      fontsize_legend=fontsize_legend,
                      legend_border_width=legend_border_width,
                      legend_border_height=legend_border_height,
                      legend_border_linewidth=legend_border_linewidth,
                      legend_border_x=legend_border_x,
                      legend_border_y=legend_border_y,
                      plot_width=plot_width,
                      plot_height=plot_height,
                      output_folder=output_folder,
                      model_offset=model_offset,
                      filename=f"summary_plot_step{step}.png")

    else:
        best_models = {}
        for step_i in range(1, step):
            best_result = find_best_result(analysis_file=get_analysis_file_path(results_folder, step_i),
                                                   models=best_models)
            best_models[step_i] = extract_model_name_from_filename(best_result, step_i)

        same_model_prev_step_df, best_model_prev_step_df = split_df(analysis_df, step)

        best_same_model_prev_step_df = find_best_results_df(same_model_prev_step_df,
                                                            step=step)
        best_best_model_prev_step_df = find_best_results_df(best_model_prev_step_df,
                                                            step=step,
                                                            best_models=best_models)

        generate_plot(analysis_df=same_model_prev_step_df,
                      unique_models=unique_models,
                      bar_params=bar_params,
                      bar_height=bar_height,
                      best_df=best_same_model_prev_step_df,
                      bar_gap=bar_gap,
                      barcolor=barcolor,
                      fontsize=fontsize,
                      step=step,
                      x_start=x_start,
                      y_start=y_start,
                      spacing=spacing,
                      fontsize_legend=fontsize_legend,
                      legend_border_width=legend_border_width,
                      legend_border_height=legend_border_height,
                      legend_border_linewidth=legend_border_linewidth,
                      legend_border_x=legend_border_x,
                      legend_border_y=legend_border_y,
                      plot_width=plot_width,
                      plot_height=plot_height,
                      output_folder=output_folder,
                      model_offset=model_offset,
                      filename=f"summary_plot_step{step}_same_model_prev_step.png")
        generate_plot(analysis_df=best_model_prev_step_df,
                      unique_models=unique_models,
                      bar_params=bar_params,
                      bar_height=bar_height,
                      best_df=best_best_model_prev_step_df,
                      bar_gap=bar_gap,
                      barcolor=barcolor,
                      fontsize=fontsize,
                      step=step,
                      x_start=x_start,
                      y_start=y_start,
                      spacing=spacing,
                      fontsize_legend=fontsize_legend,
                      legend_border_width=legend_border_width,
                      legend_border_height=legend_border_height,
                      legend_border_linewidth=legend_border_linewidth,
                      legend_border_x=legend_border_x,
                      legend_border_y=legend_border_y,
                      plot_width=plot_width,
                      plot_height=plot_height,
                      output_folder=output_folder,
                      model_offset=model_offset,
                      filename=f"summary_plot_step{step}_best_model_prev_step.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create plots for analysis CSV files.")

    parser.add_argument("input", type=str, help="File or directory with CSV files used as input.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--bar-params", nargs='+', required=True, help="Bar parameters to plot")
    parser.add_argument("--models-to-display", nargs='+', default=[], help="Model names of those to dispplay in the plot.")
    parser.add_argument("--bar-height", type=float, default=0.3, help="Height of bars in plot")
    parser.add_argument("--bar-gap", type=float, default=0.02, help="Gap between bars in plot")
    parser.add_argument("--barcolor", type=str, default=HELLBLAU_BALKEN, help="HexCode for bar color.")
    parser.add_argument("--fontsize", type=float, default=12, help="fontsize of model names")
    parser.add_argument("--plot_width", type=float, default=16, help="size of the output plot")
    parser.add_argument("--plot_height", type=float, default=9, help="size of the output plot")
    parser.add_argument("--model-offset", type=float, default=0.0)
    parser.add_argument("--x_start_legend", type=float, default=1.0, help="x position of legend")
    parser.add_argument("--y_start_legend", type=float, default=-0.05, help="y position of legend")
    parser.add_argument("--spacing_legend", type=float, default=-0.1, help="spacing between legend items")
    parser.add_argument("--fontsize_legend", type=float, default=20, help="fontsize of the legend text")
    parser.add_argument("--legend_border_width", type=float, default=2, help="width of the border around the legend")
    parser.add_argument("--legend_border_height", type=float, default=2, help="height of the border around the legend")
    parser.add_argument("--legend_border_linewidth", type=float, default=2, help="line width of the border around the legend")
    parser.add_argument("--legend_border_x", type=float, default=0.2, help="x coordinate of the border around the legend")
    parser.add_argument("--legend_border_y", type=float, default=0.2, help="y coordinate of the border around the legend")

    args = parser.parse_args()

    ensure_directory(args.output_dir)

    if os.path.isfile(args.input):
        create_summary_plot(analysis_file_path=args.input, bar_params=args.bar_params, output_folder=args.output_dir,
                            models_to_display=args.models_to_display, bar_height=args.bar_height, bar_gap=args.bar_gap,
                            barcolor=args.barcolor, fontsize=args.fontsize, plot_width=args.plot_width, plot_height=args.plot_height,
                            model_offset=args.model_offset, x_start=args.x_start_legend, y_start=args.y_start_legend,
                            spacing=args.spacing_legend, fontsize_legend=args.fontsize_legend,
                            legend_border_width=args.legend_border_width, legend_border_height=args.legend_border_height,
                            legend_border_linewidth=args.legend_border_linewidth,
                            legend_border_x=args.legend_border_x, legend_border_y=args.legend_border_y)
    else:  # it must be a directory
        analysis_files = glob.glob(os.path.join(args.input, "analysis_step*.csv"))
        for analysis_file in analysis_files:
            create_summary_plot(analysis_file_path=args.input, bar_params=args.bar_params, output_folder=args.output_dir,
                                models_to_display=args.models_to_display, bar_height=args.bar_height, bar_gap=args.bar_gap,
                                barcolor=args.barcolor, plot_width=args.plot_width, plot_height=args.plot_height,
                                model_offset=args.model_offset, x_start=args.x_start_legend, y_start=args.y_start_legend,
                                spacing=args.spacing_legend, fontsize_legend=args.fontsize_legend,
                                legend_border_width=args.legend_border_width, legend_border_height=args.legend_border_height,
                                legend_border_linewidth=args.legend_border_linewidth,
                                legend_border_x=args.legend_border_x, legend_border_y=args.legend_border_y)

#python create_summary_plot.py "C:\Users\ortilaki\Documents\UroLlmEval-master\analysis_step1.csv" --output-dir "C:\Users\ortilaki\Documents\UroLlmEval-master\Plots" --bar-params recall specificity p_na --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct --fontsize 11.5 --plot_width 9 --plot_height 9
#python create_summary_plot.py "C:\Users\ortilaki\Documents\UroLlmEval-master\analysis_step2.csv" --output-dir "C:\Users\ortilaki\Documents\UroLlmEval-master\Plots" --bar-params p_all_diagnoses_found_in_snippets_with_diagnoses, p_no_other_diagnoses_found_for_snippet, p_snippets_correct --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct --fontsize 11.5 --plot_width 9 --plot_height 9
#python create_summary_plot.py "C:\Users\ortilaki\Documents\UroLlmEval-master\analysis_step3.csv" --output-dir "C:\Users\ortilaki\Documents\UroLlmEval-master\Plots" --bar-params p_correct, p_correct_total, p_na --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct levenshtein-regex --fontsize 11.5 --plot_width 10 --plot_height 9

#For the GMDS poster:
#python .\UroLlmEval\create_summary_plot.py "C:\Users\lenzstef\Desktop\output\analysis_step1.csv" --output-dir "output" --bar-params recall specificity p_na --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct --fontsize 20 --plot_width 8.5 --plot_height 9
#python .\UroLlmEval\create_summary_plot.py "C:\Users\lenzstef\Desktop\output\analysis_step2.csv" --output-dir "output" --bar-params p_all_diagnoses_found_in_snippets_with_diagnoses, p_no_other_diagnoses_found_for_snippet, p_snippets_correct  --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct --fontsize 20 --plot_width 10.5 --plot_height 9 --model-offset 0.15
#python .\UroLlmEval\create_summary_plot.py "C:\Users\lenzstef\Desktop\output\analysis_step3.csv" --output-dir "output" --bar-params p_correct, p_correct_total, p_na  --models-to-display mistralai_Mistral-7B-Instruct-v0.3 BioMistral_BioMistral-7B LeoLM_leo-hessianai-7b-chat meta-llama_Meta-Llama-3.1-8B-Instruct mistralai_Mixtral-8x7B-Instruct-v0.1 meta-llama_Meta-Llama-3.1-70B-Instruct levenshtein-regex --fontsize 20 --plot_width 9.5 --plot_height 9 --model-offset 0.15

#For the paper:
# step1 with 3 metrics:
#python create_summary_plot.py "C:\Users\ortilaki\UroLlmEval\analysis_step1.csv" --output-dir C:\Users\ortilaki\UroLlmEval\test2 --bar-height 0.3 --models-to-display PAPER_MODELS --bar-params recall specificity p_na --fontsize 20 --plot_width 8.5 --plot_height 10 --x_start_legend 1.05 --y_start_legend 1.8 --spacing_legend 0.3 --fontsize_legend 16 --legend_border_width 0.8 --legend_border_height 2.3 --legend_border_linewidth 1 --legend_border_y -0.3
# step1 with 4 metrics:
#python create_summary_plot.py "C:\Users\ortilaki\UroLlmEval\analysis_step1.csv" --output-dir C:\Users\ortilaki\UroLlmEval\test2\mit_accuracy --models-to-display PAPER_MODELS --bar-params recall specificity p_correct_total p_na --model-offset 0.18 --fontsize 20 --bar-height 0.3 --plot_width 8.5 --plot_height 13.5 --x_start_legend 1.05 --y_start_legend 1.8 --spacing_legend 0.3 --fontsize_legend 16 --legend_border_width 0.85 --legend_border_height 2.45 --legend_border_linewidth 1 --legend_border_y -0.35  --bar-height 0.4
# step2:
#python create_summary_plot.py "C:\Users\ortilaki\UroLlmEval\analysis_step2.csv" --output-dir C:\Users\ortilaki\UroLlmEval\test2 --bar-params p_all_diagnoses_found_in_snippets_with_diagnoses, p_no_other_diagnoses_found_for_snippet, p_snippets_correct  --models-to-display PAPER_MODELS --fontsize 20 --plot_width 10.5 --plot_height 13 --model-offset 0.15 --x_start_legend -1 --y_start_legend -1 --spacing_legend 0.3 --fontsize_legend 16 --legend_border_width 1.68 --legend_border_height 1.68 --legend_border_linewidth 1 --legend_border_y -2.4
# step3:
#python create_summary_plot.py "C:\Users\ortilaki\UroLlmEval\analysis_step3.csv" --output-dir C:\Users\ortilaki\UroLlmEval\test2 --bar-params p_correct, p_correct_total, p_na --models-to-display PAPER_MODELS --fontsize 20 --plot_width 9.5 --plot_height 14 --model-offset 0.12 --x_start_legend -1.2 --y_start_legend -0.95 --spacing_legend 0.3 --fontsize_legend 16 --legend_border_width 2.15 --legend_border_height 1.68 --legend_border_linewidth 1 --legend_border_y -2.35