# Bu dosya, ayni klasordeki PNG grafiklerini ureten koddur.
# 1.ipynb notebook'undan otomatik olarak kaydedilmistir.
import glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metrics_dir = r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\metrics"
graphs_dir = r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\graphs"

all_examples = {}
try:
    with open(f"{metrics_dir}\\run_summary.json", encoding="utf-8") as f:
        main_data = json.load(f)
    if "end_to_end_example" in main_data:
        all_examples["gece_(ana)"] = main_data["end_to_end_example"]
    if "end_to_end_examples" in main_data:
        all_examples.update(main_data["end_to_end_examples"])
except FileNotFoundError:
    pass
for path in glob.glob(f"{metrics_dir}\\run_summary_scenarios_*.json"):
    with open(path, encoding="utf-8") as f:
        all_examples.update(json.load(f).get("end_to_end_examples", {}))

labels = list(all_examples.keys())
forecasts = [ex["demand_forecast"]["forecast_mw"] for ex in all_examples.values()]
capacities = [ex["grid_status"]["available_capacity_mw"] for ex in all_examples.values()]
decisions = [ex["agent_decision"]["decision"] for ex in all_examples.values()]

decision_colors = {"normal_operation": "#1B98A4", "demand_response_alert": "#F2A541", "activate_reserve": "#C0392B"}
colors = [decision_colors.get(d, "#999999") for d in decisions]
margin_pct = [(c - f) / c * 100 for f, c in zip(forecasts, capacities)]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(labels))
ax.barh(y_pos, margin_pct, color=colors)
ax.set_yticks(y_pos); ax.set_yticklabels(labels)
ax.set_xlabel("Rezerv Marji (%)")
ax.set_title("Senaryo Bazinda Rezerv Marji ve Agent Karari")
for i, (m, d) in enumerate(zip(margin_pct, decisions)):
    ax.text(m + 0.5, i, d, va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{graphs_dir}\\viz_decision_map.png", dpi=150)
plt.close()

monthly = pd.read_csv(f"{metrics_dir}\\monthly_summary_full_period.csv")
monthly["year_month"] = pd.to_datetime(monthly["year_month"])
monthly["total_degree_days"] = monthly["total_hdd"] + monthly["total_cdd"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax1 = axes[0]
ax1.plot(monthly["year_month"], monthly["avg_mw"], color="#0F3057")
ax1.set_ylabel("Ortalama Talep (MW)", color="#0F3057")
ax2 = ax1.twinx()
ax2.plot(monthly["year_month"], monthly["avg_temp_f_mean"], color="#F2A541", alpha=0.7)
ax2.set_ylabel("Ortalama Sicaklik (F)", color="#F2A541")
ax1.set_title("14 Yillik Talep ve Sicaklik Trendi (2004-2018)")

ax3 = axes[1]
ax3.scatter(monthly["total_degree_days"], monthly["avg_mw"], color="#1B98A4", alpha=0.6)
z = np.polyfit(monthly["total_degree_days"], monthly["avg_mw"], 1)
trend_x = np.linspace(monthly["total_degree_days"].min(), monthly["total_degree_days"].max(), 50)
ax3.plot(trend_x, np.polyval(z, trend_x), color="#C0392B", linestyle="--", label="Regresyon")
r2 = np.corrcoef(monthly["total_degree_days"], monthly["avg_mw"])[0, 1] ** 2
ax3.set_xlabel("Toplam Derece-Gun (HDD+CDD)")
ax3.set_ylabel("Ortalama Talep (MW)")
ax3.set_title(f"Derece-Gun Bazli Talep Iliskisi (R^2={r2:.2f})")
ax3.legend()
plt.tight_layout()
plt.savefig(f"{graphs_dir}\\viz_monthly_degree_days.png", dpi=150)
plt.close()
