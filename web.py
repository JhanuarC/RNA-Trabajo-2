"""
Aplicación Web de Riesgo Crediticio - Modelo de Red Neuronal (PyTorch)
====================================================================
Streamlit App para predicción de probabilidad de default y Scorecard.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import os

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="RiskScore AI - Evaluador de Riesgo Crediticio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-value {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin: 1rem 0;
    }
    .score-label {
        font-size: 1.2rem;
        text-align: center;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #F0F4F8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E3A5F;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #2E5A8F 0%, #1E3A5F 100%);
    }
    .stProgress > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CLASE DEL MODELO DE RED NEURONAL
# ============================================================================
class CreditRiskNN(nn.Module):
    """
    Red Neuronal para predicción de riesgo crediticio.
    Arquitectura: 3 capas lineales con ReLU y Sigmoid final.
    """
    def __init__(self, input_dim):
        super(CreditRiskNN, self).__init__()
        self.network = nn.Sequential(
            # Capa 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Capa 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Capa 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Salida — probabilidad de default
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

# ============================================================================
# FUNCIONES DE CARGA DE RECURSOS
# ============================================================================
@st.cache_resource
def load_resources():
    try:
        # Cargar esquema
        with open('esquema_modelo.json', 'r') as f:
            esquema = json.load(f)
        
        # --- NUEVA LÓGICA DE LIMPIEZA ---
        # Filtramos las columnas para que coincidan con las 66 que el modelo espera
        columnas_reales = [c for c in esquema['columnas_modelo'] if c not in ["loan_status", "default", "verification_status_joint"]]
        input_size = len(columnas_reales) 
        # --------------------------------
        
        # Cargar scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # Cargar modelo con el tamaño corregido (66)
        model = CreditRiskNN(input_size)
        model.load_state_dict(torch.load('credit_risk_nn.pth', map_location=torch.device('cpu')))
        model.eval()
        
        # Actualizamos el esquema en memoria para que el resto de la app use las 66 columnas
        esquema['columnas_modelo'] = columnas_reales
        esquema['total_features'] = input_size
        
        return esquema, scaler, model
    except Exception as e:
        st.error(f"Error cargando recursos: {e}")
        return None, None, None


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================
def crear_dataframe_usuario(datos, columnas_modelo):
    """
    Crea un DataFrame con el orden EXACTO de columnas que espera el modelo.
    Implementa One-Hot Encoding manual para variables categóricas.
    """
    # Diccionario base con todas las columnas del modelo inicializadas en 0
    registro = {col: 0 for col in columnas_modelo}

    # --- Variables Numéricas ---
    numericas = [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
        "installment", "emp_length", "annual_inc", "pymnt_plan", "zip_code",
        "dti", "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
        "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
        "revol_util", "total_acc", "collections_12_mths_ex_med",
        "mths_since_last_major_derog", "acc_now_delinq", "tot_coll_amt",
        "tot_cur_bal", "total_rev_hi_lim"
    ]
    for col in numericas:
        if col in registro and col in datos:
            registro[col] = float(datos[col])

    # --- One-Hot Encoding Manual: Home Ownership ---
    ho = datos.get("home_ownership", "RENT")
    for opt in ["MORTGAGE", "NONE", "OTHER", "OWN", "RENT"]:
        col_name = f"home_ownership_{opt}"
        if col_name in registro:
            registro[col_name] = 1 if ho == opt else 0

    # --- One-Hot Encoding Manual: Verification Status ---
    vs = datos.get("verification_status", "Not Verified")
    # Mapear valores a columnas del modelo
    vs_map = {
        "Source Verified": "verification_status_Source Verified",
        "Verified": "verification_status_Verified",
        "Not Verified": None  # categoría base (omitted)
    }
    for opt, col_name in vs_map.items():
        if col_name and col_name in registro:
            registro[col_name] = 1 if vs == opt else 0

    # --- One-Hot Encoding Manual: Purpose ---
    purp = datos.get("purpose", "debt_consolidation")
    purpose_options = [
        "credit_card", "debt_consolidation", "educational", "home_improvement",
        "house", "major_purchase", "medical", "moving", "other",
        "renewable_energy", "small_business", "vacation", "wedding"
    ]
    for opt in purpose_options:
        col_name = f"purpose_{opt}"
        if col_name in registro:
            registro[col_name] = 1 if purp == opt else 0

    # --- One-Hot Encoding Manual: Initial List Status ---
    ils = datos.get("initial_list_status", "f")
    col_ils = "initial_list_status_w"
    if col_ils in registro:
        registro[col_ils] = 1 if ils == "w" else 0

    # --- One-Hot Encoding Manual: Application Type ---
    at = datos.get("application_type", "INDIVIDUAL")
    col_at = "application_type_JOINT"
    if col_at in registro:
        registro[col_at] = 1 if at == "JOINT" else 0

    # --- One-Hot Encoding Manual: Employment Title Category ---
    emp_cat = datos.get("emp_title_cat", "Other")
    emp_options = [
        "Company_Name", "Construction", "Education", "Executive",
        "Finance_Legal", "Government_Military", "Healthcare", "Hospitality",
        "Management", "Operations", "Other", "Sales_Retail",
        "Self_Employed", "Social_Community", "Tech", "Transportation", "Unknown"
    ]
    for opt in emp_options:
        col_name = f"emp_title_cat_{opt}"
        if col_name in registro:
            registro[col_name] = 1 if emp_cat == opt else 0

    # --- One-Hot Encoding Manual: Address Region ---
    region = datos.get("addr_region", "South")
    for opt in ["Northeast", "South", "West"]:
        col_name = f"addr_region_{opt}"
        if col_name in registro:
            registro[col_name] = 1 if region == opt else 0

    # Crear DataFrame con orden EXACTO del esquema
    df = pd.DataFrame([registro])[columnas_modelo]
    return df


# ============================================================================
# FUNCIÓN DE CÁLCULO DE SCORECARD
# ============================================================================
def calcular_score(prob_default, target_score=650, target_odds=50, pdo=25):
    """
    Calcula el puntaje de crédito (Scorecard) usando la fórmula:
    Score = offset + (factor * ln(odds))

    Parámetros:
    -----------
    prob_default : float
        Probabilidad de default (entre 0 y 1)
    target_score : int
        Puntaje objetivo (default: 650)
    target_odds : int
        Odds objetivo (default: 50)
    pdo : int
        Puntos para duplicar odds (default: 25)

    Retorna:
    --------
    score : int
        Puntaje de crédito redondeado
    odds : float
        Odds calculados
    factor, offset : float
        Parámetros de la escala
    """
    # Clip de probabilidad para evitar log(0) o división por cero
    probabilidad = np.clip(prob_default, 0.0001, 0.9999)

    # Calcular odds = (1 - p) / p
    odds = (1 - probabilidad) / probabilidad

    # Calcular factor y offset
    factor = pdo / np.log(2)
    offset = target_score - (factor * np.log(target_odds))

    # Calcular score final
    score = offset + (factor * np.log(odds))
    score = int(round(score))

    # Limitar score a rango razonable (300-850)
    score = max(300, min(850, score))

    return score, odds, factor, offset


def interpretar_riesgo(score):
    """Interpreta el nivel de riesgo basado en el puntaje."""
    if score >= 750:
        return "🟢 EXCELENTE", "Riesgo muy bajo. Probabilidad de default mínima. El solicitante califica para las mejores tasas.", "#27AE60"
    elif score >= 650:
        return "🟢 BUENO", "Riesgo bajo. El solicitante tiene un perfil crediticio sólido y favorable.", "#2ECC71"
    elif score >= 600:
        return "🟡 ACEPTABLE", "Riesgo moderado. Se recomienda revisión adicional de documentación.", "#F39C12"
    elif score >= 500:
        return "🟠 RIESGOSO", "Riesgo alto. Se sugiere solicitar garantías adicionales o co-signatario.", "#E67E22"
    else:
        return "🔴 ALTO RIESGO", "Riesgo muy alto. Probabilidad elevada de incumplimiento. Rechazo recomendado.", "#E74C3C"


# ============================================================================
# VISUALIZACIÓN DEL SCORE
# ============================================================================
def render_score_bar(score):
    """Renderiza una barra de color para el puntaje."""
    # Determinar color basado en score
    if score >= 750:
        color = "#27AE60"
        gradient = "linear-gradient(90deg, #27AE60 0%, #2ECC71 100%)"
    elif score >= 650:
        color = "#2ECC71"
        gradient = "linear-gradient(90deg, #2ECC71 0%, #82E0AA 100%)"
    elif score >= 600:
        color = "#F39C12"
        gradient = "linear-gradient(90deg, #F39C12 0%, #F5B041 100%)"
    elif score >= 500:
        color = "#E67E22"
        gradient = "linear-gradient(90deg, #E67E22 0%, #EB984E 100%)"
    else:
        color = "#E74C3C"
        gradient = "linear-gradient(90deg, #E74C3C 0%, #EC7063 100%)"

    # Calcular porcentaje para la barra (rango 300-850)
    pct = ((score - 300) / (850 - 300)) * 100

    html = f"""
    <div style="margin: 1.5rem 0;">
        <div style="background-color: #E8E8E8; border-radius: 15px; height: 35px; position: relative; overflow: hidden;">
            <div style="background: {gradient}; width: {pct}%; height: 100%; border-radius: 15px;
                        transition: width 1s ease-in-out; display: flex; align-items: center;
                        justify-content: flex-end; padding-right: 12px;">
                <span style="color: white; font-weight: 700; font-size: 0.9rem;">{score}</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.8rem; color: #888;">
            <span>300</span>
            <span>425</span>
            <span>550</span>
            <span>675</span>
            <span>850</span>
        </div>
    </div>
    """
    return html, color


# ============================================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ============================================================================
def predecir(datos_usuario, esquema, scaler, modelo):
    """
    Ejecuta el pipeline completo: preprocesamiento -> escalado -> predicción -> score.
    """
    columnas = esquema["columnas_modelo"]

    # 1. Crear DataFrame con orden exacto
    df = crear_dataframe_usuario(datos_usuario, columnas)

    # 2. Escalar con StandardScaler
    X_scaled = scaler.transform(df.values)

    # 3. Convertir a tensor de PyTorch
    X_tensor = torch.FloatTensor(X_scaled)

    # 4. Predicción
    with torch.no_grad():
        prob_default = modelo(X_tensor).item()

    # 5. Calcular Scorecard
    score, odds, factor, offset = calcular_score(
        prob_default, target_score=650, target_odds=50, pdo=25
    )

    return prob_default, score, odds, factor, offset, df


# ============================================================================
# BARRA LATERAL - FORMULARIO DE ENTRADA
# ============================================================================
def render_sidebar():
    """Renderiza el formulario de entrada en la barra lateral."""
    st.sidebar.markdown("## 📝 Datos del Solicitante")
    st.sidebar.markdown("---")

    datos = {}

    # --- SECCIÓN: INFORMACIÓN DEL PRÉSTAMO ---
    st.sidebar.markdown("### 💰 Información del Préstamo")

    datos["loan_amnt"] = st.sidebar.number_input(
        "Monto del Préstamo ($)", min_value=500, max_value=40000,
        value=10000, step=500,
        help="Cantidad solicitada por el prestatario"
    )
    datos["funded_amnt"] = st.sidebar.number_input(
        "Monto Financiado ($)", min_value=500, max_value=40000,
        value=datos["loan_amnt"], step=500,
        help="Monto total comprometido hasta el momento"
    )
    datos["funded_amnt_inv"] = st.sidebar.number_input(
        "Monto Financiado por Inversores ($)", min_value=0, max_value=40000,
        value=datos["funded_amnt"], step=500,
        help="Monto financiado por inversores"
    )
    datos["term"] = st.sidebar.selectbox(
        "Plazo (meses)", options=[36, 60], index=0,
        help="Número de pagos del préstamo"
    )
    datos["int_rate"] = st.sidebar.number_input(
        "Tasa de Interés (%)", min_value=5.0, max_value=35.0,
        value=13.5, step=0.01, format="%.2f",
        help="Tasa de interés del préstamo"
    )
    datos["installment"] = st.sidebar.number_input(
        "Cuota Mensual ($)", min_value=20.0, max_value=1500.0,
        value=350.0, step=10.0, format="%.2f",
        help="Pago mensual del prestatario"
    )
    datos["purpose"] = st.sidebar.selectbox(
        "Propósito del Préstamo",
        options=[
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "car", "medical",
            "moving", "vacation", "house", "wedding", "educational",
            "renewable_energy", "other"
        ],
        index=0,
        help="Categoría del propósito del préstamo"
    )
    datos["application_type"] = st.sidebar.selectbox(
        "Tipo de Solicitud",
        options=["INDIVIDUAL", "JOINT"], index=0,
        help="Indica si el préstamo es individual o conjunto"
    )
    datos["initial_list_status"] = st.sidebar.selectbox(
        "Estado de Lista Inicial",
        options=["f", "w"], index=0,
        help="f = fractional, w = whole"
    )
    datos["pymnt_plan"] = st.sidebar.selectbox(
        "Plan de Pagos",
        options=[0, 1], index=0, format_func=lambda x: "Sí" if x == 1 else "No",
        help="Indica si tiene un plan de pagos"
    )

    st.sidebar.markdown("---")

    # --- SECCIÓN: INFORMACIÓN PERSONAL ---
    st.sidebar.markdown("### 👤 Información Personal")

    datos["annual_inc"] = st.sidebar.number_input(
        "Ingreso Anual ($)", min_value=0, max_value=500000,
        value=65000, step=1000,
        help="Ingreso anual reportado por el prestatario"
    )
    datos["emp_length"] = st.sidebar.selectbox(
        "Antigüedad Laboral (años)",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        index=5,
        help="Años de antigüedad en el empleo actual"
    )
    datos["emp_title_cat"] = st.sidebar.selectbox(
        "Categoría Laboral",
        options=[
            "Other", "Tech", "Healthcare", "Finance_Legal",
            "Education", "Sales_Retail", "Construction",
            "Government_Military", "Executive", "Management",
            "Operations", "Hospitality", "Transportation",
            "Social_Community", "Self_Employed", "Company_Name", "Unknown"
        ],
        index=0,
        help="Sector o categoría laboral del solicitante"
    )
    datos["home_ownership"] = st.sidebar.selectbox(
        "Situación de Vivienda",
        options=["RENT", "MORTGAGE", "OWN", "OTHER", "NONE"],
        index=0,
        help="Estado de propiedad de la vivienda"
    )
    datos["verification_status"] = st.sidebar.selectbox(
        "Estado de Verificación",
        options=["Not Verified", "Source Verified", "Verified"],
        index=0,
        help="Estado de verificación de ingresos"
    )
    datos["addr_region"] = st.sidebar.selectbox(
        "Región",
        options=["South", "Northeast", "West"],
        index=0,
        help="Región geográfica del solicitante"
    )
    datos["zip_code"] = st.sidebar.number_input(
        "Código Postal (3 dígitos)", min_value=0, max_value=999,
        value=100, step=1,
        help="Primeros 3 dígitos del código postal"
    )

    st.sidebar.markdown("---")

    # --- SECCIÓN: INFORMACIÓN CREDITICIA ---
    st.sidebar.markdown("### 💳 Información Crediticia")

    datos["dti"] = st.sidebar.number_input(
        "DTI - Relación Deuda/Ingreso (%)", min_value=0.0, max_value=50.0,
        value=25.0, step=0.1, format="%.2f",
        help="Ratio deuda-ingreso (deuda mensual / ingreso mensual) * 100"
    )
    datos["revol_bal"] = st.sidebar.number_input(
        "Saldo Revolvente ($)", min_value=0, max_value=200000,
        value=12000, step=500,
        help="Saldo total crediticio revolvente"
    )
    datos["revol_util"] = st.sidebar.number_input(
        "Utilización Revolvente (%)", min_value=0.0, max_value=120.0,
        value=45.0, step=1.0, format="%.2f",
        help="Porcentaje de crédito revolvente utilizado"
    )
    datos["open_acc"] = st.sidebar.number_input(
        "Líneas de Crédito Abiertas", min_value=0, max_value=50,
        value=12, step=1,
        help="Número de líneas de crédito abiertas"
    )
    datos["total_acc"] = st.sidebar.number_input(
        "Total Líneas de Crédito", min_value=0, max_value=100,
        value=25, step=1,
        help="Número total de líneas de crédito"
    )
    datos["total_rev_hi_lim"] = st.sidebar.number_input(
        "Límite Total Revolvente ($)", min_value=0, max_value=500000,
        value=25000, step=1000,
        help="Límite máximo de crédito revolvente total"
    )
    datos["tot_cur_bal"] = st.sidebar.number_input(
        "Saldo Total Actual ($)", min_value=0, max_value=500000,
        value=50000, step=1000,
        help="Saldo total actual en todas las cuentas"
    )
    datos["delinq_2yrs"] = st.sidebar.number_input(
        "Morosidades (2 años)", min_value=0, max_value=20,
        value=0, step=1,
        help="Número de morosidades de 30+ días en los últimos 2 años"
    )
    datos["inq_last_6mths"] = st.sidebar.number_input(
        "Consultas (6 meses)", min_value=0, max_value=20,
        value=1, step=1,
        help="Número de consultas de crédito en los últimos 6 meses"
    )
    datos["mths_since_last_delinq"] = st.sidebar.number_input(
        "Meses desde última morosidad", min_value=0, max_value=180,
        value=0, step=1,
        help="Meses desde la última morosidad (0 = nunca)"
    )
    datos["mths_since_last_record"] = st.sidebar.number_input(
        "Meses desde último registro", min_value=0, max_value=180,
        value=0, step=1,
        help="Meses desde el último registro público (0 = nunca)"
    )
    datos["pub_rec"] = st.sidebar.number_input(
        "Registros Públicos", min_value=0, max_value=20,
        value=0, step=1,
        help="Número de registros públicos derogatorios"
    )
    datos["collections_12_mths_ex_med"] = st.sidebar.number_input(
        "Cobranzas (12 meses)", min_value=0, max_value=20,
        value=0, step=1,
        help="Número de cobranzas en los últimos 12 meses (excl. médicas)"
    )
    datos["mths_since_last_major_derog"] = st.sidebar.number_input(
        "Meses desde última derogación mayor", min_value=0, max_value=180,
        value=0, step=1,
        help="Meses desde la última derogación mayor (0 = nunca)"
    )
    datos["acc_now_delinq"] = st.sidebar.number_input(
        "Cuentas actualmente morosas", min_value=0, max_value=20,
        value=0, step=1,
        help="Número de cuentas actualmente en mora"
    )
    datos["tot_coll_amt"] = st.sidebar.number_input(
        "Monto Total en Cobranza ($)", min_value=0, max_value=100000,
        value=0, step=100,
        help="Monto total adeudado en cobranzas"
    )

    st.sidebar.markdown("---")

    # Botón de evaluación
    st.sidebar.markdown("### 🚀 Acción")
    evaluar = st.sidebar.button("📊 EVALUAR RIESGO")

    return datos, evaluar


# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================
def main():
    # Encabezado
    st.markdown('<div class="main-header">📊 RiskScore AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluador Inteligente de Riesgo Crediticio basado en Redes Neuronales</div>', unsafe_allow_html=True)

    # Verificar existencia de archivos necesarios
    archivos_necesarios = ["esquema_modelo.json", "scaler.pkl", "credit_risk_nn.pth"]
    faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]

    if faltantes:
        st.error(f"⚠️ **Archivos faltantes:** {', '.join(faltantes)}")
        st.info("""
        Asegúrate de que los siguientes archivos estén en el mismo directorio que `app.py`:
        - `esquema_modelo.json`: Esquema de columnas del modelo
        - `scaler.pkl`: StandardScaler entrenado
        - `credit_risk_nn.pth`: Pesos del modelo PyTorch
        """)
        st.stop()

    # Cargar recursos
    try:
        esquema, scaler, modelo = load_resources()
    except Exception as e:
        st.error(f"❌ Error cargando recursos: {e}")
        st.stop()

    # Mostrar info del modelo
    with st.expander("ℹ️ Información del Modelo"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Features de Entrada", esquema["total_features"])
        col2.metric("Arquitectura", "128 → 64 → 32 → 1")
        col3.metric("Activación Salida", "Sigmoide")

    # --- FORMULARIO EN SIDEBAR ---
    datos_usuario, evaluar = render_sidebar()

    # --- ÁREA PRINCIPAL: RESULTADOS ---
    st.markdown("---")

    if evaluar:
        with st.spinner("🧠 Analizando riesgo crediticio..."):
            try:
                prob_default, score, odds, factor, offset, df = predecir(
                    datos_usuario, esquema, scaler, modelo
                )

                nivel_riesgo, interpretacion, color = interpretar_riesgo(score)
                score_bar_html, _ = render_score_bar(score)

                # ========== RESULTADOS ==========
                st.markdown("## 📈 Resultados de la Evaluación")

                col_left, col_right = st.columns([1, 1])

                with col_left:
                    # Probabilidad de Default
                    st.markdown("### 🔴 Probabilidad de Default")
                    prob_pct = prob_default * 100
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <div style="font-size: 3rem; font-weight: 800; color: {'#E74C3C' if prob_pct > 50 else '#E67E22' if prob_pct > 25 else '#F39C12' if prob_pct > 15 else '#27AE60'};">
                            {prob_pct:.2f}%
                        </div>
                        <div style="font-size: 0.9rem; color: #888;">Chance de incumplimiento</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gauge visual para probabilidad
                    st.progress(min(prob_default, 1.0))

                    # Odds
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Odds (Razón de probabilidad):</strong> {odds:.4f}<br>
                        <small>Factor: {factor:.4f} | Offset: {offset:.4f}</small>
                    </div>
                    """, unsafe_allow_html=True)

                with col_right:
                    # Scorecard
                    st.markdown("### 🎯 Puntaje de Crédito (Scorecard)")
                    st.markdown(score_bar_html, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem;">
                        <div class="score-value" style="color: {color};">{score}</div>
                        <div class="score-label" style="color: {color};">{nivel_riesgo}</div>
                    </div>
                    <div class="risk-box" style="background-color: {color}15; border-left: 4px solid {color};">
                        {interpretacion}
                    </div>
                    """, unsafe_allow_html=True)

                # --- TABLA DE REFERENCIA ---
                st.markdown("---")
                st.markdown("### 📋 Escala de Puntuación")

                escala_cols = st.columns(5)
                escala_datos = [
                    ("750-850", "EXCELENTE", "#27AE60", "Mínimo riesgo"),
                    ("650-749", "BUENO", "#2ECC71", "Riesgo bajo"),
                    ("600-649", "ACEPTABLE", "#F39C12", "Riesgo moderado"),
                    ("500-599", "RIESGOSO", "#E67E22", "Riesgo alto"),
                    ("300-499", "ALTO RIESGO", "#E74C3C", "Máximo riesgo")
                ]
                for col, (rango, etiqueta, color_e, desc) in zip(escala_cols, escala_datos):
                    with col:
                        st.markdown(f"""
                        <div style="background-color: {color_e}18; border-radius: 8px; padding: 0.8rem; text-align: center; border-top: 3px solid {color_e};">
                            <div style="font-size: 0.8rem; color: {color_e}; font-weight: 700;">{rango}</div>
                            <div style="font-size: 0.85rem; font-weight: 600; margin: 0.3rem 0;">{etiqueta}</div>
                            <div style="font-size: 0.75rem; color: #888;">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # --- RESUMEN DE DATOS INGRESADOS ---
                with st.expander("📄 Ver datos procesados"):
                    st.markdown("**DataFrame final (orden exacto del modelo):**")
                    st.dataframe(df, use_container_width=True)
                    st.markdown(f"**Forma del tensor de entrada:** `(1, {esquema['total_features']})`")

            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")
                st.exception(e)
    else:
        # Estado inicial
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #888;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">👈</div>
            <div style="font-size: 1.2rem;">Completa el formulario en la barra lateral y presiona <strong>EVALUAR RIESGO</strong></div>
            <div style="font-size: 0.9rem; margin-top: 1rem;">El modelo analizará tu perfil y generará una probabilidad de default junto con un puntaje de crédito.</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        RiskScore AI v1.0 | Modelo: Red Neuronal PyTorch | Scorecard: Logistic Odds Scaling<br>
        Desarrollado para evaluación de riesgo crediticio
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    main()
