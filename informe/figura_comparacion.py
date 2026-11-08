"""Genera la figura comparativa del informe a partir de los resultados de test (etapa 5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"Clásico": "#2a78d6", "Encoder": "#eb6834", "LLM": "#1baf7a"}
# (nombre, enfoque, macro_f1, ic_inf, ic_sup, inferencia s/1000 comentarios)
M = [
    ("Baseline TF-IDF + LogReg", "Clásico", 0.309, 0.272, 0.351, 0.079),
    ("N-gramas de caracteres + LogReg", "Clásico", 0.435, 0.384, 0.487, 0.654),
    ("BETO congelado + LogReg", "Encoder", 0.458, 0.408, 0.507, 3.5),
    ("BETO fine-tuning", "Encoder", 0.495, 0.453, 0.541, 1.2),
    ("Qwen2.5-1.5B few-shot (12)", "LLM", 0.530, 0.485, 0.575, 372.0),
    ("Qwen2.5-1.5B zero-shot", "LLM", 0.537, 0.484, 0.588, 37.2),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#888",
                     "axes.linewidth": 0.6, "xtick.color": "#444", "ytick.color": "#444"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})

for i, (n, e, f, lo, hi, _) in enumerate(M):
    a.plot([lo, hi], [i, i], color=COL[e], lw=2, solid_capstyle="round")
    a.plot(f, i, "o", ms=6, color=COL[e], mec="white", mew=1.2)
    a.text(hi + 0.008, i, f"{f:.3f}", va="center", fontsize=7, color="#333")
a.axvline(0.254, ls=":", c="#999", lw=1)
a.text(0.258, 5.45, "siempre «favor»", fontsize=6.5, color="#777")
a.set_yticks(range(len(M)), [m[0] for m in M], fontsize=7)
a.set_xlim(0.24, 0.64); a.set_ylim(-0.6, 5.8)
a.set_xlabel("Macro-F1 en test (IC 95% bootstrap)")
a.grid(axis="x", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
a.set_title("(a) Desempeño", loc="left", fontsize=8.5, fontweight="bold")

pos = {0: (0.11, 0.300, "left"), 1: (0.75, 0.418, "left"), 2: (4.6, 0.458, "left"), 3: (1.6, 0.495, "left"), 4: (372, 0.514, "center"), 5: (27, 0.537, "right")}
for i, (n, e, f, lo, hi, t) in enumerate(M):
    b.plot(t, f, "o", ms=7, color=COL[e], mec="white", mew=1.2, label=e if e not in b.get_legend_handles_labels()[1] else None)
    x, y, ha = pos[i]
    b.text(x, y, "Baseline" if i == 0 else n.replace("Qwen2.5-1.5B ", "Qwen "), ha=ha, va="center", fontsize=6.5, color="#333")
b.set_xscale("log"); b.set_xlim(0.03, 2000); b.set_ylim(0.28, 0.58)
b.set_xlabel("Segundos por 1.000 comentarios (log)")
b.set_ylabel("Macro-F1 en test")
b.grid(color="#e6e6e6", lw=0.6); b.set_axisbelow(True)
b.legend(fontsize=7, frameon=False, loc="lower right")
b.set_title("(b) Desempeño vs. inferencia", loc="left", fontsize=8.5, fontweight="bold")
for ax in (a, b):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout(w_pad=1.5)
plt.savefig("informe/figura_comparacion.svg")
plt.savefig("informe/figura_comparacion.png", dpi=200)
plt.savefig("informe/figura_comparacion.pdf")
