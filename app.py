import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn


# =====================================================
# 1) Définition du modèle LSTM (même archi que notebook)
# =====================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x : (batch, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]           # (batch, hidden_dim)
        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits


# =====================================================
# 2) Chargement des données / modèles (avec cache)
# =====================================================

@st.cache_data
def load_data_and_features():
    """Charge le dataset préparé et la liste des features."""
    df = pd.read_csv("df_ready_for_app.csv")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return df, feature_cols


@st.cache_resource
def load_rf_model():
    """Charge le modèle RandomForest entraîné (Phase 1)."""
    return joblib.load("random_forest.pkl")


@st.cache_resource
def load_lstm_model(input_dim, num_classes, hidden_dim=64, num_layers=1):
    """Charge le modèle LSTM (poids) + retourne le device."""
    device = torch.device("cpu")   # Streamlit Cloud = CPU
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)

    state_dict = torch.load("lstm_rmsprop.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


@st.cache_resource
def load_scaler_lstm():
    """Charge le scaler utilisé pendant l'entraînement du LSTM."""
    scaler = joblib.load("scaler_lstm.pkl")
    return scaler


@st.cache_data
def prepare_sequential_data(df, feature_cols, seq_len=10):
    """
    Trie le dataframe dans l'ordre temporel, prépare X_all (features)
    et pré-calcul X_all_scaled pour le LSTM.
    """
    df_seq = df.sort_values("timestamp").reset_index(drop=True)
    X_all = df_seq[feature_cols].values.astype("float32")

    scaler = load_scaler_lstm()
    X_all_scaled = scaler.transform(X_all)

    return df_seq, X_all, X_all_scaled


# =====================================================
# 3) Fonctions utilitaires
# =====================================================

def create_seq_from_index(X_all_scaled, idx_last, seq_len=10):
    """
    Construit une séquence (seq_len, n_features) terminée à idx_last.
    idx_last doit être >= seq_len-1.
    """
    start = idx_last - seq_len + 1
    X_seq = X_all_scaled[start:idx_last+1, :]
    return X_seq


# =====================================================
# 4) CONFIG Streamlit + état
# =====================================================

st.set_page_config(
    page_title="Time Series – ML & Deep Learning",
    layout="wide"
)

SEQ_LEN = 10

if "idx_last" not in st.session_state:
    st.session_state.idx_last = SEQ_LEN - 1  # première position exploitable


# =====================================================
# 5) En-tête + contexte
# =====================================================

st.title("🕒 Classification de signaux capteurs – ML vs Deep Learning")

with st.expander("🧬 Contexte du projet", expanded=True):
    st.markdown("""
    Ce projet porte sur des **données de capteurs d'une montre** (accélération, gyroscope,
    champ magnétique, angles, etc.), échantillonnées dans le temps.

    - Les données forment une **série temporelle** : chaque ligne = un instant `timestamp`.
    - Nous avons construit une cible **`id_class`** (regroupement d'IDs d'intervalle).
    - **Phase 1 – Machine Learning** : modèles classiques (RandomForest, GradientBoosting, etc.).
    - **Phase 2 – Deep Learning** : MLP, LSTM avec différentes méthodes d'optimisation (SGD, Momentum, RMSprop…).

    L'interface ci-dessous permet de comparer :
    - 🌲 **RandomForest** (modèle tabulaire, non séquentiel)
    - 🧠 **LSTM + RMSprop** (modèle séquentiel qui regarde les 10 derniers instants)
    """)


# =====================================================
# 6) Chargement des données
# =====================================================

df, feature_cols = load_data_and_features()
df_seq, X_all, X_all_scaled = prepare_sequential_data(df, feature_cols, seq_len=SEQ_LEN)

num_features = len(feature_cols)
num_classes = len(np.unique(df["id_class"]))

# mapping id_class -> id_group (si PRESENT)
if "id_group" in df_seq.columns:
    map_df = df_seq[["id_class", "id_group"]].drop_duplicates().sort_values("id_class")
    idclass_to_group = dict(zip(map_df["id_class"], map_df["id_group"]))
else:
    idclass_to_group = {}


# =====================================================
# 7) SIDEBAR – Navigation temporelle + exploration de classes
# =====================================================

st.sidebar.header("⚙️ Navigation temporelle")

max_idx = len(df_seq) - 1
min_idx = SEQ_LEN - 1

# boutons Prev / Next / Random
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

# slider synchronisé
st.session_state.idx_last = st.sidebar.slider(
    "Index temporel (position dans la série)",
    min_value=min_idx,
    max_value=max_idx,
    value=st.session_state.idx_last
)

idx_last = st.session_state.idx_last
st.sidebar.write(f"Index sélectionné : `{idx_last}`")

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
        st.sidebar.warning("Pas d'échantillon pour cette classe dans df_seq.")

# choix de la feature à tracer
feature_to_plot = st.sidebar.selectbox(
    "Feature à afficher sur la fenêtre temporelle :",
    feature_cols,
    index=0
)


# =====================================================
# 8) Affichage info observation
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


# =====================================================
# 9) Contexte temporel (fenêtre LSTM)
# =====================================================

st.subheader("📈 Contexte temporel utilisé par le LSTM")

start_idx = idx_last - SEQ_LEN + 1
window_df = df_seq.iloc[start_idx:idx_last+1].copy()

st.line_chart(
    window_df.set_index("timestamp")[feature_to_plot],
    height=250
)

st.caption(
    f"Évolution de `{feature_to_plot}` sur les {SEQ_LEN} instants "
    f"qui précèdent t (fenêtre d'entrée du LSTM)."
)

st.markdown("---")


# =====================================================
# 10) Onglets : RandomForest vs LSTM
# =====================================================

tab_rf, tab_lstm = st.tabs(["🌲 RandomForest (ML)", "🧠 LSTM + RMSprop (DL)"])


# ---------- Onglet RandomForest ----------
with tab_rf:
    st.subheader("🌲 RandomForest – Machine Learning classique")
    st.markdown("""
    **RandomForest** est un ensemble d'arbres de décision.
    Il ne traite pas directement la structure temporelle, mais des **features tabulaires**
    (accélérations, gyroscopes, champs magnétiques, + lags que nous avons créés).
    """)

    rf = load_rf_model()
    X_rf = row[feature_cols].values.reshape(1, -1)

    pred_class_rf = int(rf.predict(X_rf)[0])
    proba_rf = rf.predict_proba(X_rf)[0]

    st.write(f"**Classe prédite (id_class) :** `{pred_class_rf}`")
    st.write(f"**Classe réelle (id_class) :** `{true_class}`")

    proba_df_rf = pd.DataFrame({
        "id_class": rf.classes_,
        "probabilité": proba_rf
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
            height=250
        )


# ---------- Onglet LSTM ----------
with tab_lstm:
    st.subheader("🧠 LSTM + RMSprop – Modèle séquentiel")
    st.markdown("""
    Le **LSTM** reçoit en entrée une séquence de longueur 10  
    (les 10 derniers instants de la série) et prédit la classe au temps t.
    L'optimiseur utilisé est **RMSprop**, bien adapté aux réseaux récurrents.
    """)

    model_lstm, device = load_lstm_model(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=1
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
        "probabilité": probs
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
            height=250
        )


# =====================================================
# 11) Résumé des performances globales
# =====================================================

st.markdown("---")
with st.expander("📊 Résumé des performances globales (sur le jeu de test)", expanded=False):
    st.markdown("""
    **Modèles Machine Learning :**
    - 🌲 RandomForest (stratifié, avec lags)  
      - Accuracy test ≈ **0.996**
      - Modèle robuste et très performant sur la majorité des classes.

    **Modèles Deep Learning :**
    - 🧠 MLP + SGD + momentum (mini-batch)  
      - Accuracy test ≈ **0.97**
    - 🧠 LSTM + RMSprop  
      - Accuracy test ≈ **0.973**
      - Meilleure prise en compte de la dynamique temporelle (fenêtre de 10 instants).

    👉 L'application Web permet de visualiser, pour un instant donné,
    comment **ML classique** et **Deep Learning séquentiel** se comportent sur les mêmes données.
    """)
