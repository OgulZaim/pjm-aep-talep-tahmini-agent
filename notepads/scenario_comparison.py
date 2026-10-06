import glob
import json
import pandas as pd


def load_all_scenarios(metrics_dir):
    """
    metrics_dir icindeki TUM run_summary*.json dosyalarini tarar,
    her senaryoyu tek bir pandas DataFrame'e toplar.
    Yeni bir JSON dosyasi eklendiginde otomatik olarak dahil olur -
    kalici, tekrar calistirilabilir bir ozellik.
    """
    rows = []

    # Ana run_summary.json (tekil ornek + varsa coklu senaryolar)
    main_path = f"{metrics_dir}\\run_summary.json"
    try:
        with open(main_path, encoding="utf-8") as f:
            main_data = json.load(f)
        if "end_to_end_example" in main_data:
            rows.append(("gece_(ana_ornek)", main_data["end_to_end_example"]))
        for name, ex in main_data.get("end_to_end_examples", {}).items():
            rows.append((name, ex))
    except FileNotFoundError:
        pass

    # Tum senaryo/stres testi dosyalari
    for path in glob.glob(f"{metrics_dir}\\run_summary_scenarios_*.json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for name, ex in data.get("end_to_end_examples", {}).items():
            rows.append((name, ex))

    records = []
    for name, ex in rows:
        forecast = ex["demand_forecast"]["forecast_mw"]
        capacity = ex["grid_status"]["available_capacity_mw"]
        margin_pct = round((capacity - forecast) / capacity * 100, 1)
        records.append({
            "senaryo": name,
            "tarih_saat": ex.get("actual_datetime", "-"),
            "tahmin_mw": forecast,
            "kapasite_mw": capacity,
            "marj_yuzde": margin_pct,
            "zaman_dilimi": ex["grid_status"].get("time_period", "-"),
            "durum": ex["agent_decision"]["status"],
            "karar": ex["agent_decision"]["decision"],
            "belirsizlik": ex["agent_decision"].get("uncertainty", "-"),
        })

    df_comparison = pd.DataFrame(records).sort_values("marj_yuzde").reset_index(drop=True)
    return df_comparison