import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn

# =========================================
# 1) Chargement des ressources
# =========================================

@st.cache_data
def load_data_and_features():
    df = pd.read_csv("df_ready_for_app.csv")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return df, feature_cols

@st.cache_resource
def load_rf_model():
    return joblib.load("random_forest.pkl")

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
        out, (hn, cn) = self.lstm(x)
        last_hidden = hn[-1]
        logits = self.fc(last_hidden)
        return logits

@st.cache_resource
def load_lstm_and_scaler(input_dim, num_classes, hidden_dim=64, num_layers=1):
    device = torch.device("cpu")
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)
    state_dict = torch.load("lstm_rmsprop.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    scaler = joblib.load("scaler_lstm.pkl")
    return model, scaler, device

def build_sequences_input(df, feature_cols, seq_len=10):
    df_seq = df.sort_values("timestamp").reset_index(drop=True)
    X_all = df_seq[feature_cols].values.astype("float32")
    return df_seq, X_all

def create_seq_from_index(X_all_scaled, idx_last, seq_len=10):
    start = idx_last - seq_len + 1
    X_seq = X_all_scaled[start:idx_last+1, :]
    return X_seq

# =========================================
# 2) CONFIG et état (session_state)
# =========================================

st.set_page_config(page_title="Time Series ML & DL Demo", layout="wide")

if "idx_last" not in st.session_state:
    st.session_state.idx_last = 9   # par défaut : première position possible

SEQ_LEN = 10

st.title("🕒 Classification de signaux capteurs – ML & Deep Learning")

st.markdown("""
Application interactive pour comparer un **modèle de Machine Learning classique (RandomForest)**  
et un **modèle de Deep Learning séquentiel (LSTM + RMSprop)** sur des données de capteurs temporelles.
""")

df, feature_cols = load_data_and_features()
df_seq, X_all = build_sequences_input(df, feature_cols, seq_len=SEQ_LEN)

num_features = len(feature_cols)
num_classes = len(np.unique(df["id_class"]))

# mapping id_class -> id_group (optionnel)
if "id_group" in df_seq.columns:
    map_df = df_seq[["id_class", "id_group"]].drop_duplicates().sort_values("id_class")
    idclass_to_group = dict(zip(map_df["id_class"], map_df["id_group"]))
else:
    idclass_to_group = {}

# =========================================
# 3) Sidebar : navigation dans le temps
# =========================================

st.sidebar.header("⚙️ Navigation temporelle")

max_idx = len(df_seq) - 1
min_idx = SEQ_LEN - 1

# boutons précédent / suivant / aléatoire
col_b1, col_b2, col_b3 = st.sidebar.columns(3)
with col_b1:
    if st.button("⬅️ Prev"):
        st.session_state.idx_last = max(min_idx, st.session_state.idx_last - 1)
with col_b2:
    if st.button("➡️ Next"):
        st.session_state.idx_last = min(max_idx, st.session_state.idx_last + 1)
with col_b3:
    if st.button("🎲 Random"):
        st.session_state.idx_last = int(np.random.randint(min_idx, max_idx+1))

# slider synchronisé avec session_state
st.session_state.idx_last = st.sidebar.slider(
    "Index temporel (position dans la série)",
    min_value=min_idx,
    max_value=max_idx,
    value=st.session_state.idx_last
)

idx_last = st.session_state.idx_last

st.sidebar.write(f"Index sélectionné : {idx_last}")

# choix de la feature à tracer
feature_to_plot = st.sidebar.selectbox(
    "Feature à afficher sur la fenêtre temporelle :",
    feature_cols,
    index=0
)

# =========================================
# 4) Informations sur l'observation
# =========================================

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

# =========================================
# 5) Contexte temporel (fenêtre LSTM)
# =========================================

st.subheader("📈 Contexte temporel utilisé par le LSTM")

start_idx = idx_last - SEQ_LEN + 1
window_df = df_seq.iloc[start_idx:idx_last+1]

st.line_chart(
    window_df.set_index("timestamp")[feature_to_plot],
    height=250
)
st.caption(f"Évolution de `{feature_to_plot}` sur les {SEQ_LEN} derniers instants avant t (entrée du LSTM).")

st.markdown("---")

# =========================================
# 6) Onglets pour comparer RF vs LSTM
# =========================================

tab_rf, tab_lstm = st.tabs(["🌲 RandomForest (ML)", "🧠 LSTM + RMSprop (DL)"])

# ---------- RandomForest ----------
with tab_rf:
    st.subheader("🌲 Prédiction avec RandomForest (Machine Learning classique)")
    st.markdown("""
    Modèle d'arbres de décision **non séquentiel** :  
    il considère uniquement le vecteur de features à l'instant t (et les lags déjà inclus comme colonnes).
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

    c_rf1, c_rf2 = st.columns(2)
    with c_rf1:
        st.write("**Probabilités par classe :**")
        st.dataframe(proba_df_rf)
    with c_rf2:
        st.bar_chart(
            proba_df_rf.set_index("id_class")["probabilité"],
            height=250
        )

# ---------- LSTM ----------
with tab_lstm:
    st.subheader("🧠 Prédiction avec LSTM + RMSprop (Deep Learning séquentiel)")
    st.markdown("""
    Modèle **LSTM** : il prend en entrée une séquence de longueur 10  
    (les 10 derniers instants avant t) et produit une prédiction pour la classe au temps t.
    """)

    model_lstm, scaler_lstm, device = load_lstm_and_scaler(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=1
    )

    # standardiser toutes les features
    X_all_scaled = scaler_lstm.transform(X_all)
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

    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.write("**Probabilités par classe :**")
        st.dataframe(proba_df_lstm)
    with c_l2:
        st.bar_chart(
            proba_df_lstm.set_index("id_class")["probabilité"],
            height=250
        )

st.markdown("""
---
✅ *Cette interface permet de comparer en direct un modèle **ML tabulaire (RandomForest)**  
et un modèle **DL séquentiel (LSTM)** sur des données de capteurs temporelles.*
""")
