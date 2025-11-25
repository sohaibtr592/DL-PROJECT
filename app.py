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
    """
    Charge le dataset préparé et la liste des features utilisées comme entrée modèle.
    """
    df = pd.read_csv("df_ready_for_app.csv")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return df, feature_cols

@st.cache_resource
def load_rf_model():
    """
    Charge le modèle RandomForest entraîné (Machine Learning classique).
    """
    rf = joblib.load("random_forest.pkl")
    return rf

class LSTMClassifier(nn.Module):
    """
    Même architecture que dans ton notebook d'entraînement.
    """
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

@st.cache_resource
def load_lstm_and_scaler(input_dim, num_classes, hidden_dim=64, num_layers=1):
    """
    Charge le modèle LSTM (poids) + le scaler utilisé à l'entraînement.
    """
    device = torch.device("cpu")   # Streamlit Cloud = CPU

    model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(device)
    state_dict = torch.load("lstm_rmsprop.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load("scaler_lstm.pkl")

    return model, scaler, device

# =========================================
# 2) Fonctions utilitaires pour les séquences LSTM
# =========================================

def build_sequences_input(df, feature_cols, seq_len=10):
    """
    Trie le dataframe dans l'ordre temporel et prépare la matrice de features brute X_all.
    (On ne construit pas toutes les séquences ici, juste les données nécessaires.)
    """
    # On suppose qu'il y a une colonne "timestamp"
    df_seq = df.sort_values("timestamp").reset_index(drop=True)
    X_all = df_seq[feature_cols].values.astype("float32")
    return df_seq, X_all

def create_seq_from_index(X_all_scaled, idx_last, seq_len=10):
    """
    Construit une séquence (seq_len, n_features) qui se termine à idx_last.
    idx_last doit être >= seq_len-1.
    """
    start = idx_last - seq_len + 1
    X_seq = X_all_scaled[start:idx_last+1, :]  # (seq_len, n_features)
    return X_seq

# =========================================
# 3) CONFIG de la page Streamlit
# =========================================

st.set_page_config(page_title="Time Series ML & DL Demo", layout="wide")

st.title("🕒 Classification de signaux capteurs – ML & Deep Learning")

st.markdown("""
Cette application permet de tester deux modèles entraînés sur des **données de capteurs de montre** :

- 🔹 **RandomForest** (Machine Learning classique – Phase 1)  
- 🔹 **LSTM + RMSprop** (Deep Learning – Phase 2)  

Le dataset utilisé est `df_ready` (features + lags + cible `id_class`).
""")

# =========================================
# 4) Chargement des données + features
# =========================================

df, feature_cols = load_data_and_features()

# On prépare un dataframe trié par temps + la matrice X_all (brute)
SEQ_LEN = 10
df_seq, X_all = build_sequences_input(df, feature_cols, seq_len=SEQ_LEN)

num_features = len(feature_cols)
num_classes = len(np.unique(df["id_class"]))

# Mapping id_class -> id_group (optionnel mais plus lisible)
if "id_group" in df_seq.columns:
    map_df = df_seq[["id_class", "id_group"]].drop_duplicates().sort_values("id_class")
    idclass_to_group = dict(zip(map_df["id_class"], map_df["id_group"]))
else:
    idclass_to_group = {}

# =========================================
# 5) Sidebar – choix du modèle et de l’index temporel
# =========================================

st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle à tester :",
    ["RandomForest (ML classique)", "LSTM + RMSprop (Deep Learning)"]
)

# On ne peut choisir qu'un index >= SEQ_LEN-1 pour pouvoir construire une séquence LSTM.
max_idx = len(df_seq) - 1
min_idx = SEQ_LEN - 1

idx_last = st.sidebar.slider(
    "Choisir un index temporel (position dans la série)",
    min_value=min_idx,
    max_value=max_idx,
    value=min_idx
)

st.sidebar.write(f"Index sélectionné : {idx_last}")

# Récupération de la ligne correspondante
row = df_seq.iloc[idx_last]
true_class = int(row["id_class"])
true_group = idclass_to_group.get(true_class, "N/A")

# =========================================
# 6) Affichage des informations de l'observation
# =========================================

st.subheader("🧾 Observation sélectionnée")

col1, col2 = st.columns(2)

with col1:
    st.write("**Timestamp :**", row["timestamp"])
    st.write("**Classe réelle (id_class) :**", true_class)
    st.write("**id_group associé :**", true_group)

with col2:
    st.write("**Features au dernier instant (t)**")
    st.dataframe(row[feature_cols].to_frame().T)

st.markdown("---")

# Petit graphe sur la fenêtre temporelle utilisée par le LSTM
st.subheader("📈 Contexte temporel (fenêtre LSTM)")

start_idx = idx_last - SEQ_LEN + 1
window_df = df_seq.iloc[start_idx:idx_last+1]

feature_to_plot = "AccelerationX"
if feature_to_plot not in window_df.columns:
    # fallback si jamais le nom exact n'existe pas
    feature_to_plot = feature_cols[0]

st.line_chart(
    window_df.set_index("timestamp")[feature_to_plot],
    height=200
)

st.caption(f"Évolution de `{feature_to_plot}` sur les {SEQ_LEN} derniers instants avant t.")

st.markdown("---")

# =========================================
# 7) Prédiction selon le modèle choisi
# =========================================

if model_choice == "RandomForest (ML classique)":
    st.subheader("🔹 Prédiction avec RandomForest")

    rf = load_rf_model()

    # On utilise uniquement le dernier instant, comme pendant l'entraînement
    X_rf = row[feature_cols].values.reshape(1, -1)

    pred_class = int(rf.predict(X_rf)[0])
    proba = rf.predict_proba(X_rf)[0]

    st.write(f"**Classe prédite (id_class) :** `{pred_class}`")
    st.write(f"**Classe réelle (id_class) :** `{true_class}`")

    # Probabilités par classe
    proba_df = pd.DataFrame({
        "id_class": rf.classes_,
        "probabilité": proba
    }).sort_values("probabilité", ascending=False)

    st.write("**Probabilités par classe :**")
    st.dataframe(proba_df)

    st.bar_chart(
        proba_df.set_index("id_class")["probabilité"],
        height=250
    )

elif model_choice == "LSTM + RMSprop (Deep Learning)":
    st.subheader("🔹 Prédiction avec LSTM + RMSprop")

    # Charger modèle + scaler
    model_lstm, scaler_lstm, device = load_lstm_and_scaler(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=1
    )

    # Standardiser toutes les features comme pendant l'entraînement
    X_all_scaled = scaler_lstm.transform(X_all)

    # Construire la séquence qui se termine à idx_last
    X_seq = create_seq_from_index(X_all_scaled, idx_last, seq_len=SEQ_LEN)  # (seq_len, n_features)

    # Passage en tenseur, ajout d'une dimension batch
    X_seq_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, n_features)

    with torch.no_grad():
        logits = model_lstm(X_seq_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    st.write(f"**Classe prédite (id_class) :** `{pred_class}`")
    st.write(f"**Classe réelle (id_class) :** `{true_class}`")

    # Probabilités par classe
    classes = sorted(df["id_class"].unique())
    proba_df = pd.DataFrame({
        "id_class": classes,
        "probabilité": probs
    }).sort_values("probabilité", ascending=False)

    st.write("**Probabilités par classe (LSTM) :**")
    st.dataframe(proba_df)

    st.bar_chart(
        proba_df.set_index("id_class")["probabilité"],
        height=250
    )

st.markdown("""
---
✅ *Cette interface illustre la différence entre un modèle de Machine Learning classique (RandomForest) et un modèle de Deep Learning séquentiel (LSTM) sur des données de capteurs temporelles.*
""")
