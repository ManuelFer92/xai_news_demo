import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pickle
import io
import warnings
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="XAI Noticias — NNInv Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card {
    background: #1e293b; border-radius: 8px; padding: 12px 16px;
    border-left: 3px solid #2563eb; margin-bottom: 8px;
  }
  .metric-val { font-size: 1.4rem; font-weight: 700; color: #60a5fa; }
  .metric-lbl { font-size: 0.75rem; color: #94a3b8; }
  .topic-chip {
    display: inline-block; background: #1e3a5f; color: #93c5fd;
    border-radius: 4px; padding: 2px 8px; margin: 2px;
    font-size: 0.78rem; font-family: monospace;
  }
  .faith-ok  { color: #4ade80; font-weight: 600; }
  .faith-mid { color: #facc15; font-weight: 600; }
  .faith-low { color: #f87171; font-weight: 600; }
  .prompt-box {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 6px; padding: 12px; font-family: monospace;
    font-size: 0.78rem; color: #cbd5e1; white-space: pre-wrap;
    max-height: 320px; overflow-y: auto;
  }
  .stButton > button {
    background: #2563eb; color: white; border: none;
    border-radius: 6px; font-weight: 600; width: 100%;
  }
  .stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ── Arquitectura NNInv ────────────────────────────────────────────────────
class NNInv(nn.Module):
    def __init__(self, n_topic, n_cat=None, use_category=False, hidden_layers=None):
        super().__init__()
        self.use_category = use_category
        if hidden_layers is None:
            hidden_layers = [64, 256, 512, 512]
        layers = []
        in_dim = 2
        for i, out_dim in enumerate(hidden_layers):
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            if i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        self.backbone   = nn.Sequential(*layers)
        self.head_sent  = nn.Linear(in_dim, 3)
        self.head_topic = nn.Linear(in_dim, n_topic)
        self.head_fecha = nn.Linear(in_dim, 2)
        if use_category and n_cat:
            self.head_cat = nn.Linear(in_dim, n_cat)

    def forward(self, x):
        h = self.backbone(x)
        out = {
            "sent":  F.softmax(self.head_sent(h),  dim=-1),
            "topic": F.softmax(self.head_topic(h), dim=-1),
            "fecha": self.head_fecha(h),
        }
        if self.use_category:
            out["cat"] = self.head_cat(h)
        return out


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def cargar_modelo(metadata):
    import os
    n_topic = metadata["n_topic"]
    n_cat   = metadata["n_cat"]
    use_cat = metadata.get("use_category", False)
    arch    = metadata.get("best_arch", [64, 256, 512, 512])

    model = NNInv(n_topic=n_topic, n_cat=n_cat,
                  use_category=use_cat, hidden_layers=arch)

    if os.path.exists("nninv_best_weights.pt"):
        raw = torch.load("nninv_best_weights.pt",
                         map_location="cpu", weights_only=False)
    elif os.path.exists("nninv_best.pkl"):
        with open("nninv_best.pkl", "rb") as f:
            loaded = CPUUnpickler(f).load()
        raw = loaded.state_dict()
    else:
        raise FileNotFoundError(
            "No se encontró nninv_best_weights.pt ni nninv_best.pkl."
        )

    # Remapear claves cortas → claves largas
    mapping = {
        "bb":  "backbone",
        "hs":  "head_sent",
        "ht":  "head_topic",
        "hf":  "head_fecha",
        "hc":  "head_cat",
    }
    fixed = {}
    for k, v in raw.items():
        prefix = k.split(".")[0]
        rest   = ".".join(k.split(".")[1:])
        new_k  = mapping.get(prefix, prefix) + "." + rest
        fixed[new_k] = v

    model.load_state_dict(fixed)
    model.eval()
    return model, "nninv_best.pkl"

@st.cache_resource(show_spinner="Cargando datos y modelo NNInv...")
def cargar_todo():
    df       = pd.read_csv("df_clusters.csv", encoding="utf-8-sig")
    coords   = np.load("umap_coords.npy")
    metadata = joblib.load("nninv_metadata.pkl")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    COLS_SENT  = metadata["cols_sent"]
    COLS_DATE  = metadata["cols_date"]
    COLS_TOPIC = metadata["cols_topic"]
    tl_map     = metadata.get("topic_labels_map", {})

    topic_labels = {}
    for col in COLS_TOPIC:
        if col in tl_map and tl_map[col]:
            topic_labels[col] = tl_map[col]
        else:
            tid  = int(col.split("_")[1])
            mask = df["topic_id"] == tid
            topic_labels[col] = (
                df[mask]["topic_label"].iloc[0] if mask.sum() > 0 else "sin etiqueta"
            )

    model, fuente = cargar_modelo(metadata)
    return df, coords, metadata, COLS_SENT, COLS_DATE, COLS_TOPIC, topic_labels, model, fuente


def dec_fecha(s, c):
    a = np.arctan2(s, c)
    if a < 0:
        a += 2 * np.pi
    return ["Ene","Feb","Mar","Abr","May","Jun",
            "Jul","Ago","Sep","Oct","Nov","Dic"][min(int(a / (2*np.pi) * 12), 11)]


def faith_color(v):
    if v >= 0.6:  return "faith-ok"
    if v >= 0.35: return "faith-mid"
    return "faith-low"


@st.cache_data(show_spinner=False)
def build_base_scatter(_df, _coords):
    cats = sorted(_df["category"].unique())
    import plotly.express as px
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    traces = []
    for i, cat in enumerate(cats):
        mask = (_df["category"] == cat).values
        traces.append(go.Scattergl(
            x=_coords[mask, 0], y=_coords[mask, 1],
            mode="markers",
            marker=dict(size=3, color=colors[i % len(colors)], opacity=0.5),
            name=cat,
            text=_df.loc[mask, "title"].str[:60].values,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    return traces


# ── Carga inicial ─────────────────────────────────────────────────────────
try:
    df, coords, metadata, COLS_SENT, COLS_DATE, COLS_TOPIC, \
        topic_labels, model, fuente_modelo = cargar_todo()
except FileNotFoundError as e:
    st.error(f"❌ Archivo no encontrado: {e}")
    st.info(
        "Coloca en la misma carpeta que app.py los siguientes archivos:\n\n"
        "- `df_clusters.csv`\n"
        "- `umap_coords.npy`\n"
        "- `nninv_metadata.pkl`\n"
        "- `nninv_best_weights.pt`  ← preferido\n"
        "- `nninv_best.pkl`         ← alternativa si no tienes el .pt"
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar: {e}")
    st.stop()

X_MIN, X_MAX = float(coords[:,0].min()), float(coords[:,0].max())
Y_MIN, Y_MAX = float(coords[:,1].min()), float(coords[:,1].max())

# ── Session state ─────────────────────────────────────────────────────────
if "historial"  not in st.session_state: st.session_state.historial  = []
if "resultado"  not in st.session_state: st.session_state.resultado  = None
if "x_query"    not in st.session_state: st.session_state.x_query    = round((X_MIN+X_MAX)/2, 2)
if "y_query"    not in st.session_state: st.session_state.y_query    = round((Y_MIN+Y_MAX)/2, 2)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ XAI Noticias")
    st.markdown("**NNInv Explorer**")
    st.caption(f"Modelo: `{fuente_modelo}`")
    st.markdown("---")
    st.markdown(f"**Rango X:** `[{X_MIN:.1f}, {X_MAX:.1f}]`")
    st.markdown(f"**Rango Y:** `[{Y_MIN:.1f}, {Y_MAX:.1f}]`")
    st.markdown("---")
    st.markdown("### Coordenadas del punto")

    x_q = st.number_input(
        "X", min_value=float(X_MIN-2), max_value=float(X_MAX+2),
        value=st.session_state.x_query, step=0.1, format="%.2f"
    )
    y_q = st.number_input(
        "Y", min_value=float(Y_MIN-2), max_value=float(Y_MAX+2),
        value=st.session_state.y_query, step=0.1, format="%.2f"
    )
    k_q          = st.select_slider("Vecinos K", options=[5, 10, 20], value=10)
    zoom_margin  = st.slider("Margen de zoom", 1.0, 6.0, 3.0, 0.5)

    st.markdown("---")
    ejecutar = st.button("🔍 Ejecutar consulta", type="primary")

    st.markdown("---")
    st.markdown("### Faithfulness global del modelo")
    faith_opt = metadata.get("faith_opt", {})
    st.metric("Topic",     f"{faith_opt.get('faith_topic', 0):.4f}")
    st.metric("Sentiment", f"{faith_opt.get('faith_sent',  0):.4f}")
    st.metric("Media",     f"{faith_opt.get('faith_mean',  0):.4f}")

    if st.session_state.historial:
        st.markdown("---")
        st.markdown(f"### Historial ({len(st.session_state.historial)} consultas)")
        df_h = pd.DataFrame(st.session_state.historial)
        st.download_button(
            "📥 Descargar historial CSV",
            data=df_h.to_csv(index=False).encode("utf-8"),
            file_name="historial_consultas.csv",
            mime="text/csv",
        )

# ── Lógica de consulta ────────────────────────────────────────────────────
if ejecutar:
    st.session_state.x_query = x_q
    st.session_state.y_query = y_q

    with torch.no_grad():
        p   = torch.tensor([[x_q, y_q]], dtype=torch.float32)
        out = model(p)

    sent  = out["sent"].numpy()[0]
    topic = out["topic"].numpy()[0]
    fecha = out["fecha"].numpy()[0]

    sent_dom = ["Negativo","Neutro","Positivo"][sent.argmax()]
    mes_pred = dec_fecha(fecha[0], fecha[1])
    top5_idx = topic.argsort()[::-1][:5]

    dists   = euclidean_distances([[x_q, y_q]], coords)[0]
    idx_v   = dists.argsort()[:k_q]
    vecinos = df.iloc[idx_v].copy()
    vecinos["distancia"] = dists[idx_v]
    cat_dom  = vecinos["category"].value_counts().index[0]

    anio_mean = vecinos["date"].dt.year.mean()
    anio_min  = int(vecinos["date"].dt.year.min())
    anio_max  = int(vecinos["date"].dt.year.max())

    r_s, _ = pearsonr(sent,  vecinos[COLS_SENT].values.mean(0))
    r_t, _ = pearsonr(topic, vecinos[COLS_TOPIC].values.mean(0))
    r_s = float(r_s) if not np.isnan(r_s) else 0.0
    r_t = float(r_t) if not np.isnan(r_t) else 0.0

    en_rango = (X_MIN <= x_q <= X_MAX) and (Y_MIN <= y_q <= Y_MAX)

    headlines  = "\n".join([
        f'{i+1}. [{row["category"]}] ({int(row["date"].year) if pd.notna(row["date"]) else "N/A"}) {row["title"]}'
        for i, (_, row) in enumerate(vecinos.iterrows())
    ])
    topics_str = "\n".join([
        f'  - {topic_labels.get(COLS_TOPIC[i], COLS_TOPIC[i])}: {topic[i]:.3f}'
        for i in top5_idx
    ])
    sent_real_mean = vecinos[COLS_SENT].mean()
    sent_real_dom  = ["Negativo","Neutro","Positivo"][sent_real_mean.values.argmax()]

    PROMPT = (
        f"Eres un analista de medios especializado en noticias en ingles.\n"
        f"Se te proporciona informacion sobre una region de un mapa semantico de noticias\n"
        f"HuffPost (2012-2022), generado con UMAP y embeddings multimodales.\n\n"
        f"== PREDICCION DEL MODELO (NNInv) ==\n"
        f"Sentimiento predicho: {sent_dom}\n"
        f"  Negativo={sent[0]:.3f}  Neutro={sent[1]:.3f}  Positivo={sent[2]:.3f}\n\n"
        f"Topicos mas probables:\n{topics_str}\n\n"
        f"Epoca del anio: {mes_pred}\n"
        f"Anio real promedio: {anio_mean:.0f} (rango: {anio_min}-{anio_max})\n"
        f"Categoria dominante: {cat_dom}\n"
        f"Faithfulness local: sentiment={r_s:.4f}  topic={r_t:.4f}\n\n"
        f"== {k_q} NOTICIAS REALES MAS CERCANAS ==\n{headlines}\n\n"
        f"Sentiment real promedio: {sent_real_dom}\n\n"
        f"== TAREA ==\n"
        f"Genera una explicacion en espanol de 3-5 oraciones sobre:\n"
        f"1. Que tipo de noticias predominan en esta region\n"
        f"2. Cual es el tono predominante y por que\n"
        f"3. Que temas recurrentes se identifican\n"
        f"4. A que epoca corresponden (usa rango: {anio_min}-{anio_max})\n"
        f"Escribe claro y sin terminos tecnicos."
    )

    st.session_state.resultado = {
        "x": x_q, "y": y_q, "k": k_q,
        "sent": sent, "sent_dom": sent_dom,
        "topic": topic, "top5_idx": top5_idx,
        "fecha": fecha, "mes_pred": mes_pred,
        "vecinos": vecinos, "idx_v": idx_v,
        "cat_dom": cat_dom,
        "anio_mean": anio_mean, "anio_min": anio_min, "anio_max": anio_max,
        "r_s": r_s, "r_t": r_t,
        "en_rango": en_rango,
        "PROMPT": PROMPT,
    }

    st.session_state.historial.append({
        "consulta":    len(st.session_state.historial) + 1,
        "x": x_q, "y": y_q, "k": k_q,
        "sent_dom":    sent_dom,
        "topic_dom":   topic_labels.get(COLS_TOPIC[top5_idx[0]], ""),
        "anio_mean":   round(anio_mean, 1),
        "anio_rango":  f"{anio_min}-{anio_max}",
        "cat_dom":     cat_dom,
        "faith_sent":  round(r_s, 4),
        "faith_topic": round(r_t, 4),
        "faith_media": round((r_s+r_t)/2, 4),
        "en_rango":    en_rango,
    })

# ── Layout principal ──────────────────────────────────────────────────────
st.markdown("## 🗺️ XAI Noticias — NNInv Explorer")
st.caption("Exploración semántica explicable · HuffPost · UMAP + NNInv")

col_map, col_res = st.columns([3, 2], gap="medium")

# ── Scatterplot ───────────────────────────────────────────────────────────
with col_map:
    base_traces = build_base_scatter(df, coords)
    fig = go.Figure(data=base_traces)
    r   = st.session_state.resultado

    if r is not None:
        fig.add_trace(go.Scattergl(
            x=coords[r["idx_v"], 0], y=coords[r["idx_v"], 1],
            mode="markers",
            marker=dict(size=10, color="#facc15", symbol="circle",
                        line=dict(color="#d97706", width=1.5)),
            name=f"K={r['k']} vecinos",
            text=r["vecinos"]["title"].str[:60].values,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r["x"]], y=[r["y"]],
            mode="markers+text",
            marker=dict(size=18, color="#ef4444", symbol="x",
                        line=dict(color="#991b1b", width=3)),
            text=["  ← consulta"],
            textposition="middle right",
            textfont=dict(color="#ef4444", size=11),
            name="Punto consultado",
            hovertemplate=f"X={r['x']:.2f}, Y={r['y']:.2f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_range=[r["x"] - zoom_margin, r["x"] + zoom_margin],
            yaxis_range=[r["y"] - zoom_margin, r["y"] + zoom_margin],
        )
    else:
        fig.update_layout(
            xaxis_range=[X_MIN-0.5, X_MAX+0.5],
            yaxis_range=[Y_MIN-0.5, Y_MAX+0.5],
        )

    fig.update_layout(
        height=540,
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)", bordercolor="#334155",
            font=dict(size=8), itemsizing="constant",
            x=1.01, y=1,
        ),
        xaxis=dict(gridcolor="#1e293b", title="UMAP-X"),
        yaxis=dict(gridcolor="#1e293b", title="UMAP-Y"),
        title=dict(
            text="Mapa semántico UMAP 2D — HuffPost 10,080 artículos",
            font=dict(size=12), x=0.5,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    if r is not None and not r["en_rango"]:
        st.warning("⚠️ Punto fuera del rango del dataset. Los vecinos serán muy lejanos.")

# ── Resultados ────────────────────────────────────────────────────────────
with col_res:
    if r is None:
        st.info("👈 Ingresa coordenadas en el panel lateral y presiona **Ejecutar consulta**")
        st.markdown(f"**Rango X:** `[{X_MIN:.2f}, {X_MAX:.2f}]`")
        st.markdown(f"**Rango Y:** `[{Y_MIN:.2f}, {Y_MAX:.2f}]`")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Predicción", "📰 Vecinos", "💬 Prompt LLM"])

        with tab1:
            st.markdown(f"#### Consulta ({r['x']:.2f}, {r['y']:.2f}) · K={r['k']}")
            st.markdown("**Sentiment predicho**")
            sent_cols  = st.columns(3)
            labels_s   = ["Negativo", "Neutro", "Positivo"]
            colors_s   = ["#ef4444", "#94a3b8", "#4ade80"]
            for i, (col, lbl, cc) in enumerate(zip(sent_cols, labels_s, colors_s)):
                with col:
                    dom = " ★" if r["sent"].argmax() == i else ""
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-val" style="color:{cc}">{r["sent"][i]:.3f}</div>'
                        f'<div class="metric-lbl">{lbl}{dom}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Top 5 tópicos predichos**")
            for i in r["top5_idx"]:
                col  = COLS_TOPIC[i]
                lbl  = topic_labels.get(col, col)
                prob = r["topic"][i]
                pct  = int(prob * 100)
                st.markdown(
                    f'<div style="margin:4px 0">'
                    f'<span class="topic-chip">{col}</span> <b>{prob:.4f}</b> — {lbl}'
                    f'<div style="height:4px;background:#1e3a5f;border-radius:2px;margin-top:3px">'
                    f'<div style="width:{pct}%;height:100%;background:#2563eb;border-radius:2px">'
                    f'</div></div></div>',
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f"**Época del año:** `{r['mes_pred']}`")
            st.markdown(f"**Año real vecinos:** `{r['anio_mean']:.0f}` (rango {r['anio_min']}–{r['anio_max']})")
            st.markdown(f"**Categoría dominante:** `{r['cat_dom']}`")
            st.markdown("---")
            st.markdown("**Faithfulness local**")
            fc_s = faith_color(r["r_s"])
            fc_t = faith_color(r["r_t"])
            fc_m = faith_color((r["r_s"]+r["r_t"])/2)
            st.markdown(
                f'Sentiment: <span class="{fc_s}">{r["r_s"]:.4f}</span> &nbsp;|&nbsp; '
                f'Topic: <span class="{fc_t}">{r["r_t"]:.4f}</span> &nbsp;|&nbsp; '
                f'Media: <span class="{fc_m}">{(r["r_s"]+r["r_t"])/2:.4f}</span>',
                unsafe_allow_html=True)

        with tab2:
            st.markdown(f"#### {r['k']} vecinos reales más cercanos")
            for i, (_, row) in enumerate(r["vecinos"].iterrows()):
                anio = int(row["date"].year) if pd.notna(row["date"]) else "N/A"
                with st.expander(
                    f"{i+1}. [{row['category']}] {row['title'][:55]}...",
                    expanded=(i == 0)
                ):
                    st.markdown(f"**Categoría:** {row['category']}")
                    st.markdown(f"**Año:** {anio}  |  **Distancia:** {row['distancia']:.3f}")
                    st.markdown(f"**Título:** {row['title']}")
                    url = row.get("url", "")
                    if pd.notna(url) and url:
                        st.markdown(f"**URL:** [{str(url)[:70]}]({url})")

            st.markdown("---")
            st.markdown("**Distribución de categorías:**")
            for cat, cnt in r["vecinos"]["category"].value_counts().items():
                pct = cnt / r["k"] * 100
                st.markdown(f"`{cat}` — {cnt}/{r['k']} ({pct:.0f}%)")

        with tab3:
            st.markdown("#### Prompt listo para LLM")
            st.caption("Copia y pega en ChatGPT, Claude o Gemini")
            st.markdown(
                f'<div class="prompt-box">{r["PROMPT"]}</div>',
                unsafe_allow_html=True)
            st.download_button(
                "📋 Descargar prompt como .txt",
                data=r["PROMPT"].encode("utf-8"),
                file_name=f"prompt_xai_{r['x']}_{r['y']}.txt",
                mime="text/plain",
            )

st.markdown("---")
st.caption("Pipeline XAI · UMAP + HDBSCAN + NNInv · Dataset HuffPost 2012–2022 · Tesis de Posgrado")