import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

st.title("Tablero de Gestión — Predicción de Churn")
st.markdown("**M71V Maestría en Gestión y Análisis de Datos Financieros** | Segunda Evaluación")
st.markdown("---")

@st.cache_data
def cargar_y_entrenar():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    cat_cols  = [col for col in df.columns if col not in num_cols + ['Churn']]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df['Churn'] = le.fit_transform(df['Churn'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    modelo = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                           use_label_encoder=False, eval_metric='logloss', random_state=42)
    modelo.fit(X_train, y_train)

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return df, X_train, X_test, y_train, y_test, y_pred, y_proba, modelo, num_cols

with st.spinner("Cargando datos y entrenando modelo..."):
    df, X_train, X_test, y_train, y_test, y_pred, y_proba, modelo, num_cols = cargar_y_entrenar()

# ── Sección 1: KPIs ejecutivos ──────────────────────────────────────────────
st.header("1. Resumen ejecutivo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de clientes",      f"{len(df):,}")
col2.metric("Tasa de Churn",          f"{df['Churn'].mean()*100:.1f}%")
col3.metric("Cargo mensual promedio", f"${df['MonthlyCharges'].mean():.2f}")
col4.metric("AUC-ROC del modelo",     f"{roc_auc_score(y_test, y_proba):.4f}")

# ── Sección 2: Análisis descriptivo ─────────────────────────────────────────
st.header("2. Análisis descriptivo")
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(5, 3))
    df['Churn'].value_counts().rename({0: 'No Churn', 1: 'Churn'}).plot(
        kind='bar', ax=ax, color=['steelblue', 'tomato'], edgecolor='white')
    ax.set_title('Distribución de Churn')
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)

with col_b:
    fig, ax = plt.subplots(figsize=(5, 3))
    df[df['Churn']==0]['MonthlyCharges'].hist(alpha=0.6, label='No Churn', ax=ax, color='steelblue', bins=20)
    df[df['Churn']==1]['MonthlyCharges'].hist(alpha=0.6, label='Churn',    ax=ax, color='tomato',    bins=20)
    ax.set_title('MonthlyCharges por Churn')
    ax.legend()
    st.pyplot(fig)

# ── Sección 3: Rendimiento del modelo ───────────────────────────────────────
st.header("3. Rendimiento del modelo")
col_c, col_d = st.columns(2)

with col_c:
    report = classification_report(y_test, y_pred,
                                   target_names=['No Churn', 'Churn'],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

with col_d:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title('Matriz de Confusión')
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')
    st.pyplot(fig)

# ── Sección 4: Importancia de variables ─────────────────────────────────────
st.header("4. Interpretabilidad — Importancia de variables")
fig, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(modelo.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)
importances.head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title('Top 15 variables — XGBoost')
ax.set_xlabel('Importancia')
st.pyplot(fig)

# ── Sección 5: Semáforo de equidad ──────────────────────────────────────────
st.header("5. Gobernanza — Semáforo de equidad (SeniorCitizen)")

X_test_eq = X_test.copy()
X_test_eq['SeniorCitizen_orig'] = df.loc[X_test.index, 'SeniorCitizen'].values
X_test_eq['y_true']  = y_test.values
X_test_eq['y_pred']  = y_pred

resultados_equidad = []
for grupo_val, grupo_nombre in [(0, 'Non-Senior'), (1, 'Senior')]:
    mask = X_test_eq['SeniorCitizen_orig'] == grupo_val
    sub  = X_test_eq[mask]
    tp = ((sub['y_pred']==1) & (sub['y_true']==1)).sum()
    fp = ((sub['y_pred']==1) & (sub['y_true']==0)).sum()
    tn = ((sub['y_pred']==0) & (sub['y_true']==0)).sum()
    fn = ((sub['y_pred']==0) & (sub['y_true']==1)).sum()
    n  = len(sub)
    resultados_equidad.append({
        'Grupo':           grupo_nombre,
        'N':               n,
        'Selection Rate':  round((tp+fp)/n,       4) if n > 0         else 0,
        'TPR (Recall)':    round(tp/(tp+fn),       4) if (tp+fn) > 0  else 0,
        'FPR':             round(fp/(fp+tn),       4) if (fp+tn) > 0  else 0,
        'Accuracy':        round((tp+tn)/n,        4) if n > 0        else 0,
        'PPV (Precision)': round(tp/(tp+fp),       4) if (tp+fp) > 0  else 0,
    })

df_eq = pd.DataFrame(resultados_equidad).set_index('Grupo')

# Tabla de métricas
st.subheader("Métricas por grupo")
st.dataframe(df_eq)

# Semáforo de ratios
st.subheader("Semáforo de ratios (rango equitativo: 0.80 — 1.25)")
metricas_eq = ['Selection Rate', 'TPR (Recall)', 'FPR', 'Accuracy', 'PPV (Precision)']

cols_semaforo = st.columns(len(metricas_eq))
for i, m in enumerate(metricas_eq):
    val_s = df_eq.loc['Senior',     m]
    val_n = df_eq.loc['Non-Senior', m]
    ratio  = val_s / val_n if val_n not in [0, np.nan] else np.nan
    equit  = 0.8 <= ratio <= 1.25
    icono  = "🟢" if equit else "🔴"
    estado = "Equitativo" if equit else "Inequidad"
    cols_semaforo[i].metric(
        label=m,
        value=f"{ratio:.4f}",
        delta=estado,
        delta_color="normal" if equit else "inverse"
    )

# Gráfico de barras comparativo
fig, ax = plt.subplots(figsize=(10, 4))
x     = np.arange(len(metricas_eq))
width = 0.35
ax.bar(x - width/2, df_eq.loc['Non-Senior', metricas_eq].values,
       width, label='Non-Senior', color='steelblue')
ax.bar(x + width/2, df_eq.loc['Senior',     metricas_eq].values,
       width, label='Senior',     color='tomato')
ax.axhline(0.8,  color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(1.25, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(metricas_eq, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title('Métricas de Equidad — Senior vs Non-Senior')
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.caption("Fuente: Elaboración propia con Python | Dataset: Telco Customer Churn (Kaggle)")
