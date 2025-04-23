import pandas as pd
import os
from data_processing_module import find_best_result_in_df, find_model_without_other_model_in_previous_step, extract_model_name_from_filename

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
    if "levenshtein-regex" in best_df["model"].values:
        print("\nBeste gefundene Ergebnisse für 'levenshtein-regex':")
        print(best_df[best_df["model"] == "levenshtein-regex"][["results_file", "model", "prompt_type"] + [c for c in df.columns if c in ["p_correct", "p_correct_total", "p_na"]]])
    return best_df

def find_best_result(analysis_file, models=None):
    analysis_df = pd.read_csv(analysis_file)
    return find_best_result_in_df(analysis_df=analysis_df, models=models)

def get_analysis_file_path(results_folder, step):
    return os.path.join(results_folder, f"analysis_step{step}.csv")
