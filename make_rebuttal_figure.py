
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

# ========= paths =========
BASE = Path(__file__).resolve().parent
CELLFORGE_JSON = BASE / "BBBC021_CellFlux_folds.json"
CELLSCIENTIST_JSON = BASE / "BBBC021_Ours_folds.json"

# ========= typography =========
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ========= panel data =========
caption_A = (
    "CellForge home-turf transfer checks. CellScientist is evaluated on four "
    "CellForge/scPerturb tasks (Norman, Schiebinger, Papalexi RNA, Papalexi Protein) "
    "under the same six-metric family, showing strong transfer evidence beyond the shared BBBC021 setting."
)

panel_A_groups = [
    {
        "subtitle": "Gene Knock Out Perturbation - scRNAseq Dataset",
        "rows": [
            ["Norman", "CellForge",      "0.0034±0.0023", "0.9846±0.0418", "0.9609±0.0081", "0.1736±0.0677", "0.8109±0.0133", "0.5975±0.0539"],
            ["",       "CellScientist",  "0.0033±0.0019", "0.9871±0.0360", "0.9675±0.0035", "0.1846±0.0293", "0.9060±0.0512", "0.6328±0.0479"],
        ]
    },
    {
        "subtitle": "Cytokine Perturbation - scRNA-seq Dataset",
        "rows": [
            ["Schiebinger", "CellForge",      "0.0428±0.0205", "0.5697±0.0943", "0.5043±0.0541", "0.0144±0.0349", "0.3396±0.0403", "0.2832±0.1154"],
            ["",            "CellScientist",  "0.0427±0.0512", "0.8818±0.0181", "0.6026±0.0094", "0.0481±0.0151", "0.8935±0.0118", "0.5793±0.0134"],
        ]
    },
    {
        "subtitle": "Gene Knock Out Perturbation - scCITEseq (RNA) Dataset",
        "rows": [
            ["Papalexi RNA", "CellForge",      "0.0417±0.0051", "0.6935±0.1995", "0.3687±0.0651", "0.0535±0.1566", "0.6406±0.1940", "0.2354±0.0224"],
            ["",             "CellScientist",  "0.0283±0.0058", "0.8193±0.0035", "0.5134±0.0176", "0.0231±0.0056", "0.8311±0.0029", "0.5583±0.0203"],
        ]
    },
    {
        "subtitle": "Gene Knock Out Perturbation - scCITEseq (Protein) Dataset",
        "rows": [
            ["Papalexi Protein", "CellForge",      "0.0070±0.0387", "0.7495±0.0653", "0.6872±0.0956", "0.2921±0.0045", "0.7409±0.0970", "0.5489±0.0749"],
            ["",                 "CellScientist",  "0.00651±0.0264", "0.7941±0.0358", "0.6758±0.0184", "0.1914±0.0026", "0.8275±0.0108", "0.7441±0.0030"],
        ]
    },
]

caption_B = (
    "Matched-budget fairness pilot on BBBC021. Strong baselines are tuned under a shared 6-hour protocol "
    "to test whether budget-matched search qualitatively reverses the shared-setting conclusion."
)

panel_B_cols = ["Method", "Wall-clock (h)", "MSE↓", "PCC↑", "R²↑", "RMSE_DEG-20↓", "RMSE_DEG-50↓", "PCC_DEG-20↑", "PCC_DEG-50↑"]
panel_B_rows = [
    ["FLAML", "4.45", "0.1457±0.07", "0.9631±0.02", "0.1101±0.17", "0.7002±0.18", "0.5094±0.12", "0.9715±0.01", "0.9672±0.01"],
    ["Random Search", "6.27", "0.3421±0.13", "0.9091±0.03", "0.1499±0.10", "1.2338±0.31", "0.8335±0.18", "0.9149±0.04", "0.9134±0.03"],
    ["CellScientist", "—", "0.1057±0.03", "0.9735±0.01", "0.1053±0.09", "0.5844±0.11", "0.4325±0.08", "0.9818±0.01", "0.9781±0.01"],
]

caption_C = (
    "System-level trade-off analysis reorganized from Table 5. Transition-wise deltas show how HRT, LCA, "
    "and PDR improve feasibility, compliance, and peak quality relative to naive retrying."
)

panel_C_cols = [
    "Transition",
    "BBBC036 (ΔSR / ΔR-SR / ΔAvg PCC / ΔBest PCC / ΔTime)",
    "BBBC047 (ΔSR / ΔR-SR / ΔAvg PCC / ΔBest PCC / ΔTime)"
]
panel_C_rows = [
    ["M0 → M1+ Naive re-prompting", "-0.263 / -0.263 / +0.0799 / +0.0803 / +7.71h", "-0.158 / -0.158 / +0.0845 / +0.0829 / +11.71h"],
    ["M1 → M2+ HRT", "+0.684 / +0.421 / +0.0615 / +0.0565 / -7.62h", "+0.731 / +0.378 / +0.0205 / +0.0194 / -11.33h"],
    ["M2 → M3+ LCA", "+0.053 / +0.053 / +0.0412 / +0.0164 / -1.88h", "+0.059 / +0.096 / +0.0843 / +0.1187 / -2.58h"],
    ["M3 → M4† PDR (CellScientist)", "0.000 / +0.052 / +0.0230 / +0.0474 / -0.53h", "0.000 / +0.030 / +0.0224 / +0.0253 / -0.80h"],
]

caption_D = (
    "Cross-task system-level ablation on LINCS2020. The same M0–M4 closed-loop pattern is tested beyond morphology "
    "using success/compliance and average/best PCC metrics."
)

panel_D_cols = ["Variant", "HRT", "LCA", "PDR", "SR↑", "R-SR↑", "Best\nPCC↑"]
panel_D_rows = [
    ["M0", "✗", "✗", "✗", "0.210", "0.526", "0.5934"],
    ["M1", "✗", "✗", "✗", "0.526", "0.588", "0.6043"],
    ["M2", "✓", "✗", "✗", "0.941", "0.684", "0.6123"],
    ["M3", "✓", "✓", "✗", "0.947", "0.714", "0.6164"],
    ["M4", "✓", "✓", "✓", "1.000", "0.789", "0.6368"],
]

caption_E = (
    "Prompt robustness and repeated-run stability on BBBC021. Small prompt paraphrases preserve the shared-setting "
    "conclusion, and repeated runs show stable behavior with explicit rollback of no-improvement updates."
)

variants = ["V0", "V1", "V2"]
mean_pcc = np.array([0.9719, 0.9703, 0.9724])
std_pcc = np.array([0.0017, 0.0020, 0.0015])

runs = ["Run 1", "Run 2", "Run 3"]
best_pcc = np.array([0.9721, 0.9736, 0.9708])
status = ["Improved", "Improved", "Rollback"]

caption_F = (
    "Exact significance on the shared BBBC021 setting. Exact p-values are reported for the flagship "
    "CellScientist vs. CellForge comparison to statistically ground the main shared-setting result."
)

METRICS = ['MSE', 'PCC', 'R2', 'DEG_RMSE_20', 'DEG_RMSE_50', 'DEG_PCC_20', 'DEG_PCC_50']
DISPLAY_METRICS_MAP = {
    'MSE': 'MSE', 'PCC': 'PCC', 'R2': 'R²',
    'DEG_RMSE_20': 'RMSE@20', 'DEG_RMSE_50': 'RMSE@50',
    'DEG_PCC_20': 'PCC@20', 'DEG_PCC_50': 'PCC@50'
}
PALETTE = {'CellScientist': '#E64B35', 'CellForge': '#4DBBD5'}

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'per_fold' in data:
        data = data['per_fold']
        if isinstance(data, dict):
            keys = sorted(data, key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))
            data = [data[k] for k in keys]
    return data

def extract_metric_values(folds, metric):
    return np.array([row.get(metric, np.nan) for row in folds], dtype=float)

def safe_paired_ttest(a, b):
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if mask.sum() < 2:
        return np.nan, np.nan
    return stats.ttest_rel(a[mask], b[mask])

def draw_caption(ax, letter, text, y_title=0.98, y_body=0.78):
    ax.axis('off')
    ax.text(0.5, y_title, f"Caption {letter}:", ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(0.5, y_body, text, ha='center', va='top', fontsize=10.5, wrap=True)

def style_panel_box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.0)
        s.set_color('black')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')

def draw_structured_table(ax, columns, rows, col_widths=None, title=None,
                          subtitle_rows=None, highlight_rows=None, blue_text_rows=None):
    ax.axis('off')
    tbl_rows = []
    row_types = []
    if subtitle_rows is None:
        subtitle_rows = []
    if highlight_rows is None:
        highlight_rows = []
    if blue_text_rows is None:
        blue_text_rows = []

    for item in rows:
        if isinstance(item, dict) and item.get("type") == "subtitle":
            tbl_rows.append([item["text"]] + [""] * (len(columns) - 1))
            row_types.append("subtitle")
        else:
            tbl_rows.append(item)
            row_types.append("data")

    table = ax.table(
        cellText=tbl_rows,
        colLabels=columns,
        cellLoc='center',
        colLoc='center',
        colWidths=col_widths,
        bbox=[0.01, 0.02, 0.98, 0.96]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.38)

    for (r, c), cell in table.get_celld().items():
        cell.visible_edges = 'TB'
        cell.set_edgecolor('#666666')
        if r == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f2f2f2')
            cell.set_linewidth(1.0)
        else:
            cell.set_linewidth(0.6)

    data_row_counter = 0
    for i, kind in enumerate(row_types, start=1):
        if kind == "subtitle":
            for c in range(len(columns)):
                cell = table[(i, c)]
                if c == len(columns)//2:
                    cell.get_text().set_text(tbl_rows[i-1][0])
                else:
                    cell.get_text().set_text("")
                cell.set_facecolor("white")
                cell.set_text_props(color='royalblue', fontstyle='italic', fontweight='bold')
                cell.set_linewidth(0.0)
                cell.visible_edges = ''
        else:
            if data_row_counter in highlight_rows:
                for c in range(len(columns)):
                    table[(i, c)].set_facecolor('#eaf4ff')
                    table[(i, c)].set_text_props(fontweight='bold')
            if data_row_counter in blue_text_rows:
                for c in range(len(columns)):
                    table[(i, c)].set_text_props(color='royalblue', fontweight='bold')
            data_row_counter += 1
    return table

def build_panel_A(ax):
    columns = ["Method", "Dataset", "MSE ↓", "PCC ↑", "R² ↑", "MSE_DE ↓", "PCC_DE ↑", "R²_DE ↑"]
    rows = []
    highlight_rows = []
    data_row_ix = 0
    for g in panel_A_groups:
        rows.append({"type": "subtitle", "text": g["subtitle"]})
        for row in g["rows"]:
            rows.append(row)
            if row[1] == "CellScientist":
                highlight_rows.append(data_row_ix)
            data_row_ix += 1
    draw_structured_table(
        ax, columns, rows,
        col_widths=[0.13, 0.13, 0.12, 0.12, 0.11, 0.12, 0.12, 0.11],
        highlight_rows=highlight_rows
    )
    style_panel_box(ax)

def build_panel_B(ax):
    draw_structured_table(
        ax, panel_B_cols, panel_B_rows,
        col_widths=[0.15, 0.12, 0.11, 0.11, 0.09, 0.13, 0.13, 0.13, 0.13],
        highlight_rows=[2]
    )
    style_panel_box(ax)

def build_panel_C(ax):
    draw_structured_table(
        ax, panel_C_cols, panel_C_rows,
        col_widths=[0.22, 0.39, 0.39],
        highlight_rows=[3],
        blue_text_rows=[1,2,3]
    )
    style_panel_box(ax)

def build_panel_D(ax):
    draw_structured_table(
        ax, panel_D_cols, panel_D_rows,
        col_widths=[0.16, 0.10, 0.10, 0.10, 0.12, 0.12, 0.14],
        highlight_rows=[4]
    )
    style_panel_box(ax)

def build_panel_E(ax_parent):
    ax_parent.axis('off')
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_parent.get_subplotspec(), wspace=0.18)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    x = np.arange(len(variants))
    ax1.errorbar(x, mean_pcc, yerr=std_pcc, fmt='o-', linewidth=1.8, markersize=5, capsize=3, color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)
    ax1.set_ylabel("Mean PCC")
    ax1.set_title("Prompt sensitivity")
    ax1.set_ylim(0.95, 0.976)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.3)

    x = np.arange(len(runs))
    colors = {"Improved": "#2ca02c", "Rollback": "#ff7f0e"}
    for i in range(len(runs)):
        ax2.scatter(x[i], best_pcc[i], s=38, color=colors[status[i]], zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(runs)
    ax2.set_ylabel("Best PCC")
    ax2.set_title("Repeated-run stability")
    ax2.set_ylim(0.95, 0.976)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.3)

    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_linewidth(0.8)

def build_panel_F(ax_parent):
    ax_parent.axis('off')
    gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=ax_parent.get_subplotspec(), wspace=0.45, hspace=0.48)

    ours = load_json(CELLSCIENTIST_JSON)
    base = load_json(CELLFORGE_JSON)

    metric_order = ['MSE', 'PCC', 'R2', 'DEG_RMSE_20', 'DEG_RMSE_50', 'DEG_PCC_20', 'DEG_PCC_50']
    pvals = []

    for idx, metric in enumerate(metric_order):
        r, c = divmod(idx, 4)
        ax = plt.subplot(gs[r, c])

        ours_vals = extract_metric_values(ours, metric)
        base_vals = extract_metric_values(base, metric)

        bp = ax.boxplot(
            [ours_vals, base_vals],
            labels=['CellScientist', 'CellForge'],
            patch_artist=True,
            showfliers=False,
            widths=0.55
        )
        for patch, color in zip(bp['boxes'], [PALETTE['CellScientist'], PALETTE['CellForge']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_linewidth(0.8)
        for obj in bp['whiskers'] + bp['caps'] + bp['medians']:
            obj.set_linewidth(0.8)

        # scatter raw points
        jitter = [-0.05, -0.02, 0.0, 0.02, 0.05]
        for i, vals in enumerate([ours_vals, base_vals], start=1):
            for j, v in enumerate(vals):
                ax.scatter(i + jitter[j % len(jitter)], v, s=8, color='black', alpha=0.7, zorder=3)

        _, p = safe_paired_ttest(ours_vals, base_vals)
        pvals.append((DISPLAY_METRICS_MAP[metric], p))

        y_max = max(np.nanmax(ours_vals), np.nanmax(base_vals))
        y_min = min(np.nanmin(ours_vals), np.nanmin(base_vals))
        y_range = max(y_max - y_min, 1e-6)
        bar_y = y_max + y_range * 0.22
        ax.plot([1, 1, 2, 2], [bar_y, bar_y + 0.04*y_range, bar_y + 0.04*y_range, bar_y], lw=0.8, c='black')
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(1.5, bar_y + 0.045*y_range, sig, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax.set_title(DISPLAY_METRICS_MAP[metric], fontsize=9.5, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelsize=6.8)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.42*y_range)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_tbl = plt.subplot(gs[1, 3])
    ax_tbl.axis('off')
    table_rows = [[name, f"{p:.2e}"] for name, p in pvals]
    table = ax_tbl.table(
        cellText=table_rows,
        colLabels=["Metric", "CellForge"],
        loc='center',
        cellLoc='center',
        bbox=[0.05, 0.0, 0.9, 0.96]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(fontweight='bold')
    ax_tbl.set_title("Exact P-values", fontsize=9.2, fontweight='bold', pad=2)

def main():
    fig = plt.figure(figsize=(16, 22), dpi=300)
    outer = gridspec.GridSpec(
        12, 1, figure=fig,
        height_ratios=[0.55, 2.15, 0.45, 0.75, 0.48, 1.25, 0.45, 0.9, 0.45, 1.55, 0.45, 1.55],
        hspace=0.06
    )

    draw_caption(fig.add_subplot(outer[0]), "A", caption_A)
    build_panel_A(fig.add_subplot(outer[1]))

    draw_caption(fig.add_subplot(outer[2]), "B", caption_B)
    build_panel_B(fig.add_subplot(outer[3]))

    draw_caption(fig.add_subplot(outer[4]), "C", caption_C)
    build_panel_C(fig.add_subplot(outer[5]))

    draw_caption(fig.add_subplot(outer[6]), "D", caption_D)
    build_panel_D(fig.add_subplot(outer[7]))

    draw_caption(fig.add_subplot(outer[8]), "E", caption_E)
    build_panel_E(fig.add_subplot(outer[9]))

    draw_caption(fig.add_subplot(outer[10]), "F", caption_F)
    build_panel_F(fig.add_subplot(outer[11]))

    fig.patch.set_facecolor('white')
    png = BASE / "icml2026_rebuttal_supplement.png"
    pdf = BASE / "icml2026_rebuttal_supplement.pdf"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"saved: {png}")
    print(f"saved: {pdf}")

if __name__ == "__main__":
    main()
