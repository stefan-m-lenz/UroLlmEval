import pandas as pd
from datetime import datetime

from analyze_step3_results import normalize_date_answer

def parse_date(date_str):
    """
    Versucht, einen normalisierten Datumsstring in ein datetime-Objekt zu parsen.
    Unterstützt Formate: YYYY-MM-DD, YYYY-MM und YYYY.
    Gibt None zurück, falls das Parsing fehlschlägt oder date_str ungültig ist.
    """
    if not date_str or date_str == "0":
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def compare_csv_dates(csv_file, output_csv=None):
    df = pd.read_csv(csv_file)

    df["model_norm"] = df["model_answer"].apply(normalize_date_answer)
    df["expected_norm"] = df["expected_answer"].apply(normalize_date_answer)

    def compute_diff(row):
        if row["expected_norm"] == "0":
            return pd.Series({"days_diff": None, "months_diff": None})
        model_date = parse_date(row["model_norm"])
        expected_date = parse_date(row["expected_norm"])
        if model_date is None or expected_date is None:
            return pd.Series({"days_diff": None, "months_diff": None})
        delta = model_date - expected_date
        days_diff = delta.days
        months_diff = days_diff // 30
        return pd.Series({"days_diff": days_diff, "months_diff": months_diff})

    df = df.join(df.apply(compute_diff, axis=1))

    if output_csv:
        df[["expected_answer", "model_answer", "model_norm", "expected_norm", "days_diff", "months_diff"]].to_csv(output_csv, index=False)

    return df

result_df = compare_csv_dates(
    "./output/step3_VAGOsolutions_Llama-3.1-SauerkrautLM-8b-Instruct__3-dates_step2_meta-llama_Meta-Llama-3.1-8B-Instruct__0-uro-ctx_step1__6-uro.csv",
    "./out.csv"
)