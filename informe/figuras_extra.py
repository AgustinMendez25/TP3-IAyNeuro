"""Genera las figuras 2 y 3 del informe a partir de los CSV de resultados de las etapas 2 a 5.
Uso (desde la raíz del repo): python informe/figuras_extra.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CLASES = ["favor", "against", "neutral"]
COL_ENF = {"Clásico": "#2a78d6", "Encoder": "#eb6834", "LLM": "#1baf7a"}
COL_CAL = {"ninguna": "#2a78d6", "contextual": "#eb6834", "lote": "#1baf7a", "sesgo_train": "#eda100"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#888",
                     "axes.linewidth": 0.6, "xtick.color": "#444", "ytick.color": "#444"})


def limpiar(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figura 2: LLM zero-shot, prompts x calibración (validación)
# ---------------------------------------------------------------------------
val = pd.read_csv("resultados_parte4_val.csv")
zs = val[val["grupo"] == "zero_shot"]
prompts = ["basico_sin_propuesta", "basico", "definiciones", "definiciones_en"]
nombres_p = ["Sin\npropuesta", "Con\npropuesta", "+ defini-\nciones", "+ defini-\nciones (en)"]
cals = ["ninguna", "contextual", "lote", "sesgo_train"]
nombres_c = {"ninguna": "Sin calibrar", "contextual": "Contextual", "lote": "Por lote",
             "sesgo_train": "Sesgo (200 de train)"}

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.35, 1]})
w = 0.19
x = np.arange(len(prompts))
for j, c in enumerate(cals):
    v = [zs[(zs.experimento == p) & (zs.calibracion == c)]["macro_f1"].iloc[0] for p in prompts]
    a.bar(x + (j - 1.5) * w, v, w * 0.9, color=COL_CAL[c], label=nombres_c[c])
mejor = zs.loc[zs["macro_f1"].idxmax()]
xb = prompts.index(mejor["experimento"]) + (cals.index(mejor["calibracion"]) - 1.5) * w
a.text(xb, mejor["macro_f1"] + 0.008, f"{mejor['macro_f1']:.3f}", ha="center", fontsize=6.5, color="#333")
a.set_xticks(x, nombres_p, fontsize=7)
a.set_ylim(0.3, 0.58)
a.set_ylabel("Macro-F1 (validación)")
a.legend(fontsize=6.5, ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 0.98), handlelength=1, columnspacing=0.8)
a.set_title("(a) Instrucción y calibración", loc="left", fontsize=8.5, fontweight="bold", pad=14)
limpiar(a)

# Distribución de las predicciones del prompt elegido según la calibración
pb = zs[zs.experimento == "basico"].set_index("calibracion").loc[cals]
filas = [("Real (validación)", [63.2, 21.1, 15.7])] + \
        [(nombres_c[c], pb.loc[c, ["%pred_favor", "%pred_against", "%pred_neutral"]].tolist()) for c in cals]
grises = ["#3d3d3a", "#8f8e88", "#d6d5cf"]
for i, (n, v) in enumerate(filas[::-1]):
    izq = 0
    for k, (val_k, g) in enumerate(zip(v, grises)):
        b.barh(i, val_k, left=izq, color=g, height=0.7, edgecolor="white", linewidth=1)
        if val_k >= 9:
            b.text(izq + val_k / 2, i, f"{val_k:.0f}", ha="center", va="center", fontsize=6.5,
                   color="white" if k < 2 else "#333")
        izq += val_k
b.set_yticks(range(len(filas)), [f[0] for f in filas[::-1]], fontsize=7)
b.set_xlim(0, 100)
b.set_xlabel("% de predicciones (instr. con propuesta)", fontsize=7)
for k, (cl, g) in enumerate(zip(CLASES, grises)):
    b.bar(0, 0, color=g, label=cl)
b.legend(fontsize=6.5, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1)
b.set_title("(b) Predicciones por clase", loc="left", fontsize=8.5, fontweight="bold", pad=14)
limpiar(b, grid_axis=None)
plt.tight_layout(w_pad=1.2)
plt.savefig("informe/figura_llm.pdf")
plt.savefig("informe/figura_llm.png", dpi=200)
plt.close()

# ---------------------------------------------------------------------------
# Figura 3: matrices de confusión en test + Macro-F1 según acuerdo entre anotadores
# ---------------------------------------------------------------------------
p2 = pd.read_csv("predicciones_parte2_test.csv")
p3 = pd.read_csv("predicciones_parte3_test.csv")
p4 = pd.read_csv("predicciones_parte4_test.csv")


def matriz(y, pred):
    m = pd.crosstab(pd.Categorical(y, CLASES), pd.Categorical(pred, CLASES), dropna=False).to_numpy()
    return m / m.sum(1, keepdims=True)


mats = [
    ("Clásico", "Clásico", matriz(p2.label, p2.pred_final)),
    ("BETO FT", "Encoder",
     np.mean([matriz(p3.label, p3[f"pred_finetuning_seed{s}"]) for s in (42, 43, 44)], axis=0)),
    ("Qwen zero-shot", "LLM", matriz(p4.label, p4.pred_zero_shot)),
]
cmap = LinearSegmentedColormap.from_list("azul", ["#f3f7fd", "#9ec5f4", "#3987e5", "#104281"])

fig = plt.figure(figsize=(7.2, 2.45))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.75, 1.3], wspace=0.12)
letras = "abc"
for i, (titulo, enf, m) in enumerate(mats):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m[r, c] > 0.5 else "#222")
    ax.set_xticks(range(3), ["fav.", "ag.", "neu."], fontsize=7)
    ax.set_yticks(range(3), ["fav.", "ag.", "neu."] if i == 0 else [""] * 3, fontsize=7)
    ax.set_xlabel("Predicción", fontsize=7)
    if i == 0:
        ax.set_ylabel("Etiqueta real", fontsize=7)
    ax.set_title(f"({letras[i]}) {titulo}", fontsize=7.2, fontweight="bold", loc="left", color="#222")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(COL_ENF[enf]); s.set_linewidth(1.6)

# Macro-F1 según acuerdo entre anotadores (valores de la etapa 5, sección 5.7)
acuerdo = [("Clásico", "Clásico", 0.468, 0.377), ("BETO FT", "Encoder", 0.563, 0.423),
           ("Qwen zero-shot", "LLM", 0.515, 0.508), ("Qwen few-shot", "LLM", 0.536, 0.486)]
ax = fig.add_subplot(gs[0, 4])
for i, (n, enf, tot, par) in enumerate(acuerdo[::-1]):
    ax.plot([par, tot], [i, i], color=COL_ENF[enf], lw=1.6, alpha=0.6)
    ax.plot(tot, i, "o", ms=6, color=COL_ENF[enf], mec="white", mew=1)
    ax.plot(par, i, "o", ms=6, mfc="white", mec=COL_ENF[enf], mew=1.5)
ax.set_yticks(range(len(acuerdo)), [a_[0] for a_ in acuerdo[::-1]], fontsize=7)
ax.set_xlim(0.35, 0.6); ax.set_ylim(-0.6, len(acuerdo) - 0.4)
ax.set_xlabel("Macro-F1 en test", fontsize=7)
ax.plot([], [], "o", color="#555", ms=5, label="acuerdo 3/3")
ax.plot([], [], "o", mfc="white", mec="#555", ms=5, label="acuerdo 2/3")
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 0.98), ncol=2,
          handletextpad=0.2, columnspacing=0.8)
ax.set_title("(d) Según acuerdo", fontsize=7.2, fontweight="bold", loc="left", pad=13)
limpiar(ax, grid_axis="x")
plt.savefig("informe/figura_errores.pdf", bbox_inches="tight")
plt.savefig("informe/figura_errores.png", dpi=200, bbox_inches="tight")
