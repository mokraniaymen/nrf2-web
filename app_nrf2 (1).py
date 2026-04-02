
# app_nrf2.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, f_oneway
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import tensorflow as tf

# Reproductibilité
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

st.title("Calculateur NRF2")
st.write("Téléversez un CSV ou entrez manuellement un échantillon pour calculer le NRF2_score.")

# Téléversement CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head())
    
    data = data.dropna()
    epsilon = 1e-6
    data["NRF2_score"] = (data["GSH"] + data["SOD"] + data["Catalase"]) / (data["NO"] + data["Arginase"] + epsilon)
    
    variables = ["GSH","SOD","Catalase","NO","Arginase"]
    X = data[variables]
    y = data["NRF2_score"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.write("Variance expliquée par PCA :", pca.explained_variance_ratio_)
    
    st.subheader("Statistiques NRF2")
    st.write(data["NRF2_score"].describe())
    
    st.subheader("Corrélations")
    corr_results = []
    for var in variables:
        if len(data[var]) > 1:
            corr, p_value = pearsonr(data[var], data["NRF2_score"])
            corr_results.append({"Variable": var, "Corrélation": corr, "p-value": p_value})
    if corr_results:
        st.table(pd.DataFrame(corr_results))
    
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Distribution du NRF2_score")
    fig2, ax2 = plt.subplots()
    ax2.hist(data["NRF2_score"], bins=30)
    st.pyplot(fig2)
    
    st.subheader("Relations variables vs NRF2_score")
    for var in variables:
        fig_sc, ax_sc = plt.subplots()
        sns.regplot(x=data[var], y=data["NRF2_score"], ax=ax_sc)
        ax_sc.set_title(f"{var} vs NRF2_score")
        st.pyplot(fig_sc)
    
    data["NRF2_group"] = pd.qcut(data["NRF2_score"], q=3, labels=["Low","Medium","High"])
    st.subheader("Boxplots par groupes NRF2")
    for var in variables:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=data["NRF2_group"], y=data[var], ax=ax_box)
        ax_box.set_title(f"{var} selon groupe NRF2")
        st.pyplot(fig_box)
    
    # Deep Learning
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=150, validation_split=0.2, verbose=0)
    
    y_pred = model.predict(X_test)
    st.subheader("Évaluation Deep Learning")
    st.write("MSE :", mean_squared_error(y_test, y_pred))
    st.write("R2 :", r2_score(y_test, y_pred))
    
    st.subheader("Réel vs Prédit")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred)
    ax3.set_xlabel("Valeurs réelles")
    ax3.set_ylabel("Prédictions")
    st.pyplot(fig3)

# Entrée manuelle
st.subheader("Prédire NRF2 pour un nouvel échantillon")
GSH = st.number_input("GSH", value=5.8)
SOD = st.number_input("SOD", value=1.3)
Catalase = st.number_input("Catalase", value=3.9)
NO = st.number_input("NO", value=0.8)
Arginase = st.number_input("Arginase", value=0.6)

if st.button("Prédire NRF2"):
    try:
        new_sample = pd.DataFrame({"GSH":[GSH], "SOD":[SOD], "Catalase":[Catalase], "NO":[NO], "Arginase":[Arginase]})
        new_scaled = scaler.transform(new_sample)
        pred = model.predict(new_scaled)
        st.write("NRF2 prédit :", float(pred[0][0]))
    except:
        st.write("Veuillez d'abord charger un CSV pour initialiser le modèle.")
