import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import shap


# =====================================================
# 1) Définition du modèle LSTM (comme dans ton notebook)
# =====================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x : (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]           # (batch, hidden_dim)
        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits


# =====================================================
# 2) Chargement des data / modèles (avec cache)
# =====================================================

@st.cache_data
def load_data_and_features():
    df = pd.read_csv("df_ready_for_app.csv")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return df, feature_cols


@st.cache_resource
def load_rf_model():
    return joblib.load("random_forest.pkl")


@st.cache_resource
def load_lstm_model(input_dim, num_classes, hidden_dim=64, num_layers=1):
    device = torch.device("cpu")
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)
    state_dict = torch.load("lstm_rmsprop.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource
def load_scaler_lstm():
    return joblib.load("scaler_lstm.pkl")


@st.cache_data
def prepare_sequential_data(df, feature_cols, seq_len=10):
    df_seq = df.sort_values("timestamp").reset_index(drop=True)
    X_all = df_seq[feature_cols].values.astype("float32")

    scaler = load_scaler_lstm()
    X_all_scaled = scaler.transform(X_all)

    return df_seq, X_all, X_all_scaled


@st.cache_resource
def get_shap_explainer():
    rf = load_rf_model()
    return shap.TreeExplainer(rf)


# =====================================================
# 3) Fonctions utilitaires
# =====================================================

def create_seq_from_index(X_all_scaled, idx_last, seq_len=10):
    start = idx_last - seq_len + 1
    X_seq = X_all_scaled[start:idx_last + 1, :]
    return X_seq


def describe_time_window(series_vals):
    val_min = float(np.min(series_vals))
    val_max = float(np.max(series_vals))
    val_mean = float(np.mean(series_vals))
    start_val = float(series_vals[0])
    end_val = float(series_vals[-1])
    amplitude = max(val_max - val_min, 1e-8)
    delta = end_val - start_val

    if delta > 0.15 * amplitude:
        trend = "globalement **croissante**"
    elif delta < -0.15 * amplitude:
        trend = "globalement **décroissante**"
    else:
        trend = "globalement **stable**"

    vol = float(np.std(series_vals))
    if vol < 0.1 * amplitude:
        vol_text = "peu variable (faible volatilité)"
    elif vol < 0.3 * amplitude:
        vol_text = "modérément variable"
    else:
        vol_text = "très variable (forte volatilité)"

    txt = (
        f"- Valeur minimale : **{val_min:.3f}**  \n"
        f"- Valeur maximale : **{val_max:.3f}**  \n"
        f"- Valeur moyenne : **{val_mean:.3f}**  \n"
        f"- Tendance sur la fenêtre : {trend}  \n"
        f"- Comportement général : {vol_text}."
    )
    return txt


def describe_proba_distribution(proba_df):
    top = proba_df.iloc[0]
    cls = int(top["id_class"])
    p_max = float(top["probabilité"])

    if len(proba_df) > 1:
        p_second = float(proba_df.iloc[1]["probabilité"])
        ratio = p_max / max(p_second, 1e-8)
    else:
        ratio = np.inf

    if ratio > 3:
        conf_text = "Le modèle est **très confiant** (la première classe est largement dominante)."
    elif ratio > 1.5:
        conf_text = "Le modèle est **modérément confiant** (la première classe domine mais les suivantes restent non négligeables)."
    else:
        conf_text = "Le modèle est **peu confiant** (plusieurs classes ont des probabilités proches)."

    txt = (
        f"- Classe la plus probable : **{cls}** avec p = **{p_max:.3f}**.  \n"
        f"- {conf_text}"
    )
    return txt


# =====================================================
# 4) Config Streamlit + état
# =====================================================

st.set_page_config(
    page_title="Time Series – ML & Deep Learning",
    layout="wide"
)

SEQ_LEN = 10

if "idx_last" not in st.session_state:
    st.session_state.idx_last = SEQ_LEN - 1
if "is_replaying" not in st.session_state:
    st.session_state.is_replaying = False


# =====================================================
# 5) Contexte + description dataset
# =====================================================

st.title("🕒 Classification de signaux capteurs – ML vs Deep Learning")

df_meta, feature_cols_meta = load_data_and_features()
n_samples, n_cols = df_meta.shape
n_features = len(feature_cols_meta)
n_classes = df_meta["id_class"].nunique()

with st.expander("🧬 Contexte du projet et description des données", expanded=True):
    st.markdown(f"""
    Ce projet porte sur des **données de capteurs d'une montre connectée**.

    - Nombre d'échantillons après préparation : **{n_samples}**  
    - Nombre de features : **{n_features}**  
    - Nombre de classes (`id_class`) après regroupement : **{n_classes}**

    Chaque ligne correspond à un **instant de mesure** (timestamp + valeurs des capteurs).
    Les IDs d'intervalle ont été regroupés en 11 classes plus équilibrées.

    **Phase 1 – Machine Learning classique :**
    - vérification série temporelle, nettoyage, lags
    - modèles : Régression Logistique, RandomForest, Gradient Boosting…

    **Phase 2 – Deep Learning :**
    - MLP avec différentes variantes de descente de gradient
    - LSTM + RMSprop sur une fenêtre de 10 instants.
    """)


# =====================================================
# 6) Chargement des données pour l'app
# =====================================================

df, feature_cols = load_data_and_features()
df_seq, X_all, X_all_scaled = prepare_sequential_data(df, feature_cols, seq_len=SEQ_LEN)

num_features = len(feature_cols)
num_classes = len(np.unique(df["id_class"]))

if "id_group" in df_seq.columns:
    map_df = df_seq[["id_class", "id_group"]].drop_duplicates().sort_values("id_class")
    idclass_to_group = dict(zip(map_df["id_class"], map_df["id_group"]))
else:
    idclass_to_group = {}


# =====================================================
# 7) Sidebar – navigation + replay
# =====================================================

st.sidebar.header("⚙️ Navigation temporelle")

max_idx = len(df_seq) - 1
min_idx = SEQ_LEN - 1

c_prev, c_next, c_rand = st.sidebar.columns(3)
with c_prev:
    if st.button("⬅️ Prev"):
        st.session_state.idx_last = max(min_idx, st.session_state.idx_last - 1)
with c_next:
    if st.button("➡️ Next"):
        st.session_state.idx_last = min(max_idx, st.session_state.idx_last + 1)
with c_rand:
    if st.button("🎲 Random"):
        st.session_state.idx_last = int(np.random.randint(min_idx, max_idx + 1))

st.session_state.idx_last = st.sidebar.slider(
    "Index temporel (position dans la série)",
    min_value=min_idx,
    max_value=max_idx,
    value=st.session_state.idx_last
)

idx_last = st.session_state.idx_last
st.sidebar.write(f"Index sélectionné : `{idx_last}`")

# Replay temporel
st.sidebar.markdown("---")
st.sidebar.subheader("▶️ Replay temporel")

replay_speed = st.sidebar.slider(
    "Vitesse (secondes entre deux pas)",
    min_value=0.05,
    max_value=1.0,
    value=0.2,
    step=0.05,
)

if st.sidebar.button("▶️ Lancer / arrêter le replay"):
    st.session_state.is_replaying = not st.session_state.is_replaying

if st.session_state.is_replaying:
    if st.session_state.idx_last < max_idx:
        st.session_state.idx_last += 1
        time.sleep(replay_speed)
        st.rerun()          # <-- nouveau au lieu de st.experimental_rerun()
    else:
        st.session_state.is_replaying = False

# exploration par classe
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Explorer par classe")

available_classes = sorted(df_seq["id_class"].unique())
selected_class = st.sidebar.selectbox(
    "Choisir une classe (id_class) :",
    available_classes
)

if st.sidebar.button("Aller à un exemple de cette classe"):
    indices = df_seq.index[df_seq["id_class"] == selected_class].tolist()
    if indices:
        st.session_state.idx_last = int(np.random.choice(indices))
        idx_last = st.session_state.idx_last
    else:
        st.sidebar.warning("Pas d'échantillon pour cette classe.")

feature_to_plot = st.sidebar.selectbox(
    "Feature à afficher sur la fenêtre temporelle :",
    feature_cols,
    index=0
)


# =====================================================
# 8) Info observation + graphe temporel + résumé automatique
# =====================================================

row = df_seq.iloc[idx_last]
true_class = int(row["id_class"])
true_group = idclass_to_group.get(true_class, "N/A")

st.subheader("🧾 Observation sélectionnée")

c1, c2 = st.columns(2)
with c1:
    st.write("**Timestamp :**", row["timestamp"])
    st.write("**Classe réelle (id_class) :**", true_class)
    st.write("**id_group associé :**", true_group)
with c2:
    st.write("**Features au dernier instant (t)**")
    st.dataframe(row[feature_cols].to_frame().T)

st.markdown("---")

st.subheader("📈 Contexte temporel utilisé par le LSTM")

start_idx = idx_last - SEQ_LEN + 1
window_df = df_seq.iloc[start_idx:idx_last + 1].copy()
st.line_chart(
    window_df.set_index("timestamp")[feature_to_plot],
    height=250
)
st.caption(
    f"Évolution de `{feature_to_plot}` sur les {SEQ_LEN} instants "
    f"qui précèdent t (fenêtre d'entrée du LSTM)."
)

series_vals = window_df[feature_to_plot].values
summary_text = describe_time_window(series_vals)
st.markdown("**Résumé automatique de la fenêtre temporelle :**")
st.info(summary_text)

st.markdown("---")


# =====================================================
# 9) Tabs : RF vs LSTM
# =====================================================

tab_rf, tab_lstm = st.tabs(["🌲 RandomForest (ML) + SHAP", "🧠 LSTM + RMSprop (DL)"])


# ---------- TAB RandomForest ----------
with tab_rf:
    st.subheader("🌲 RandomForest – Machine Learning classique")

    rf = load_rf_model()
    X_rf = row[feature_cols].values.reshape(1, -1)

    pred_class_rf = int(rf.predict(X_rf)[0])
    proba_rf = rf.predict_proba(X_rf)[0]

    st.write(f"**Classe prédite (id_class) :** `{pred_class_rf}`")
    st.write(f"**Classe réelle (id_class) :** `{true_class}`")

    proba_df_rf = pd.DataFrame({
        "id_class": rf.classes_,
        "probabilité": proba_rf,
    }).sort_values("probabilité", ascending=False)

    top_k = 5
    top_df_rf = proba_df_rf.head(top_k)

    c_rf1, c_rf2 = st.columns(2)
    with c_rf1:
        st.write(f"**Top {top_k} classes (RandomForest) :**")
        st.dataframe(top_df_rf)
    with c_rf2:
        st.bar_chart(
            top_df_rf.set_index("id_class")["probabilité"],
            height=250,
        )

    st.markdown("**Analyse automatique des probabilités (RandomForest) :**")
    st.info(describe_proba_distribution(proba_df_rf))

    # ===== Explainability locale avec SHAP (corrigé) =====
    st.markdown("---")
    st.subheader("🧐 Explication locale de la prédiction (SHAP)")

    try:
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(X_rf)

        # multi-classes : shap_values = list d'arrays
        if isinstance(shap_values, list):
            classes_list = list(rf.classes_)
            class_index = classes_list.index(pred_class_rf)
            shap_vec = shap_values[class_index][0]  # (n_features,)
        else:
            shap_vec = shap_values[0]

        shap_vec = np.array(shap_vec).reshape(-1)
        n_feat = len(feature_cols)

        if len(shap_vec) != n_feat:
            min_len = min(len(shap_vec), n_feat)
            shap_vec = shap_vec[:min_len]
            feat_for_shap = feature_cols[:min_len]
        else:
            feat_for_shap = feature_cols

        shap_df = pd.DataFrame({
            "feature": feat_for_shap,
            "shap_value": shap_vec,
            "importance_abs": np.abs(shap_vec),
        }).sort_values("importance_abs", ascending=False)

        top_k_shap = 10
        top_shap_df = shap_df.head(top_k_shap)

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.write(f"**Top {top_k_shap} features les plus influentes (SHAP) :**")
            st.dataframe(
                top_shap_df[["feature", "shap_value"]],
                use_container_width=True,
            )
        with c_s2:
            st.bar_chart(
                top_shap_df.set_index("feature")["importance_abs"],
                height=300,
            )

        st.caption(
            "Les valeurs positives poussent la prédiction vers la classe actuelle, "
            "les valeurs négatives la poussent vers une autre classe."
        )

    except Exception as e:
        st.error(f"Erreur lors du calcul SHAP : {e}")


# ---------- TAB LSTM ----------
with tab_lstm:
    st.subheader("🧠 LSTM + RMSprop – Modèle séquentiel")

    model_lstm, device = load_lstm_model(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=1,
    )

    X_seq = create_seq_from_index(X_all_scaled, idx_last, seq_len=SEQ_LEN)
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_lstm(X_seq_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class_lstm = int(np.argmax(probs))

    st.write(f"**Classe prédite (id_class) :** `{pred_class_lstm}`")
    st.write(f"**Classe réelle (id_class) :** `{true_class}`")

    classes_lstm = sorted(df["id_class"].unique())
    proba_df_lstm = pd.DataFrame({
        "id_class": classes_lstm,
        "probabilité": probs,
    }).sort_values("probabilité", ascending=False)

    top_k = 5
    top_df_lstm = proba_df_lstm.head(top_k)

    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.write(f"**Top {top_k} classes (LSTM) :**")
        st.dataframe(top_df_lstm)
    with c_l2:
        st.bar_chart(
            top_df_lstm.set_index("id_class")["probabilité"],
            height=250,
        )

    st.markdown("**Analyse automatique des probabilités (LSTM) :**")
    st.info(describe_proba_distribution(proba_df_lstm))


# =====================================================
# 10) Synthèse RF vs LSTM
# =====================================================

st.markdown("---")
st.subheader("🔍 Synthèse de comparaison des modèles pour l'instant sélectionné")

rf_ok = (pred_class_rf == true_class)
lstm_ok = (pred_class_lstm == true_class)
same_pred = (pred_class_rf == pred_class_lstm)

txt_synth = f"""
- 🌲 **RandomForest** : prédiction = `{pred_class_rf}` → {"✅ correcte" if rf_ok else "❌ incorrecte"}  
- 🧠 **LSTM + RMSprop** : prédiction = `{pred_class_lstm}` → {"✅ correcte" if lstm_ok else "❌ incorrecte"}  
- 🔁 Les deux modèles { "donnent la **même** classe." if same_pred else "donnent des **classes différentes**." }
"""
st.markdown(txt_synth)


# =====================================================
# 11) Résumé global (facultatif)
# =====================================================

with st.expander("📊 Résumé des performances globales (sur le jeu de test)", expanded=False):
    st.markdown("""
    **Machine Learning :**
    - 🌲 RandomForest (avec lags) — Accuracy test ≈ **0.996**

    **Deep Learning :**
    - 🧠 MLP + SGD + momentum — Accuracy test ≈ **0.97**
    - 🧠 LSTM + RMSprop — Accuracy test ≈ **0.973**

    L'application permet de comparer les prédictions,
    visualiser la dynamique temporelle, et interpréter les décisions
    via **SHAP** pour le RandomForest.
    """)
