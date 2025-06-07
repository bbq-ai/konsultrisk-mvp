import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="Konsultriskanalys", layout="wide")
st.title("🔍 Konsultriskanalys – MVP")

# Dummydata
def generate_dummy_data(n=100):
    np.random.seed(42)
    start_date = datetime.today()
    df = pd.DataFrame({
        'Konsult': [f'Konsult {i+1}' for i in range(n)],
        'Dagar kvar på uppdrag': np.random.randint(0, 90, n),
        'Tidigare lediga perioder': np.random.randint(0, 5, n),
        'Kompetensmatch (0-1)': np.round(np.random.rand(n), 2),
        'Aktiva säljcase': np.random.randint(0, 3, n),
        'Faktureringsgrad (%)': np.random.randint(40, 101, n),
        'Förväntad ledtid (dagar)': np.random.randint(5, 30, n),
    })
    df['Risk (1=risk, 0=ingen risk)'] = ((df['Dagar kvar på uppdrag'] < 15) & (df['Aktiva säljcase'] == 0)).astype(int)
    df['Slutdatum uppdrag'] = [start_date + timedelta(days=int(x)) for x in df['Dagar kvar på uppdrag']]
    df['Rekommenderad åtgärdsdag'] = df['Slutdatum uppdrag'] - pd.to_timedelta(df['Förväntad ledtid (dagar)'], unit='d')
    df['Startdatum uppdrag'] = df['Slutdatum uppdrag'] - pd.to_timedelta(np.random.randint(30, 180, n), unit='d')
    return df

# Ladda data
st.sidebar.header("📥 Ladda eller generera data")
data_option = st.sidebar.radio("Välj datakälla:", ["Generera testdata", "Ladda upp egen CSV"])

if data_option == "Generera testdata":
    df = generate_dummy_data(100)
else:
    uploaded_file = st.sidebar.file_uploader("Ladda upp CSV-fil", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['Startdatum uppdrag', 'Slutdatum uppdrag', 'Rekommenderad åtgärdsdag'])
    else:
        st.warning("Vänligen ladda upp en CSV-fil.")
        st.stop()

# Modell 1: Riskprediktion
st.subheader("🤖 Modell 1 – Riskprediktion")
if 'Risk (1=risk, 0=ingen risk)' not in df.columns:
    st.error("Datat måste innehålla en kolumn: 'Risk (1=risk, 0=ingen risk)'")
    st.stop()

X = df[['Dagar kvar på uppdrag', 'Tidigare lediga perioder', 'Kompetensmatch (0-1)', 'Aktiva säljcase', 'Faktureringsgrad (%)']]
y = df['Risk (1=risk, 0=ingen risk)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df['Predikterad risk'] = model.predict(X)
df['Riskprocent (%)'] = (model.predict_proba(X)[:, 1] * 100).round(1)

# Färgkoda efter risk
risk_colors = df['Predikterad risk'].map({1: 'background-color: #ffa07a', 0: ''})

# Interaktiv simulator
st.subheader("🧪 Simulera konsult")
with st.expander("Skapa egen konsultprofil"):
    dagar_kvar = st.slider("Dagar kvar på uppdrag", 0, 120, 30)
    fakt = st.slider("Faktureringsgrad (%)", 0, 100, 80)
    lediga_perioder = st.slider("Tidigare lediga perioder", 0, 10, 2)
    kompetensmatch = st.slider("Kompetensmatch", 0.0, 1.0, 0.7)
    saljcase = st.slider("Aktiva säljcase", 0, 5, 1)
    ledtid = st.slider("Förväntad ledtid (dagar)", 1, 60, 14)

    sim_df = pd.DataFrame([[dagar_kvar, lediga_perioder, kompetensmatch, saljcase, fakt]],
                          columns=X.columns)
    sim_pred = model.predict(sim_df)[0]
    sim_prob = model.predict_proba(sim_df)[0][1] * 100
    åtgärdsdatum = datetime.today() + timedelta(days=dagar_kvar - ledtid)
    st.markdown(f"**Riskprocent:** {sim_prob:.1f}%")
    st.markdown(f"**Rekommenderad åtgärdsdag:** {åtgärdsdatum.date()} ({(åtgärdsdatum - datetime.today()).days} dagar från idag)")
    st.markdown(f"**Riskkategori:** {'⚠️ Risk' if sim_pred == 1 else '✅ Ingen uppenbar risk'}")

# Visualisering
st.subheader("📈 Visualisering av risk och åtgärdsdatum")
fig, ax = plt.subplots()
ax.scatter(df['Slutdatum uppdrag'], df['Riskprocent (%)'], c=df['Predikterad risk'], cmap='coolwarm')
ax.set_xlabel('Slutdatum uppdrag')
ax.set_ylabel('Riskprocent (%)')
ax.set_title('Risknivå över tid')
plt.xticks(rotation=45)
st.pyplot(fig)

# Gantt-diagram
st.subheader("🗓️ Tidslinje: Start, Slut och Åtgärdsdag")
gantt_df = df[['Konsult', 'Startdatum uppdrag', 'Slutdatum uppdrag', 'Rekommenderad åtgärdsdag']].copy()
gantt_long = pd.melt(gantt_df, id_vars='Konsult', var_name='Typ', value_name='Datum')
fig_gantt = px.timeline(gantt_long, x_start='Datum', x_end='Datum', y='Konsult', color='Typ', title='Tidslinje per konsult', hover_name='Konsult')
fig_gantt.update_yaxes(autorange="reversed")
st.plotly_chart(fig_gantt, use_container_width=True)

# Summering och filter
st.subheader("📋 Sammanfattning och filtrering")
today = datetime.today()
df['Åtgärd inom 7 dagar'] = (df['Rekommenderad åtgärdsdag'] <= today + timedelta(days=7))
antal_akuta = df['Åtgärd inom 7 dagar'].sum()
st.info(f"🔔 {antal_akuta} konsulter behöver åtgärd inom 7 dagar")

val = st.selectbox("Filtrera på risknivå:", ["Visa alla", "Endast risk", "Åtgärd inom 7 dagar"])
if val == "Endast risk":
    df_visning = df[df['Predikterad risk'] == 1]
elif val == "Åtgärd inom 7 dagar":
    df_visning = df[df['Åtgärd inom 7 dagar'] == True]
else:
    df_visning = df

# Visa resultat
st.subheader("📌 Resultat")
st.dataframe(df_visning[['Konsult', 'Startdatum uppdrag', 'Slutdatum uppdrag', 'Faktureringsgrad (%)', 'Riskprocent (%)', 'Predikterad risk', 'Rekommenderad åtgärdsdag']].style.apply(lambda _: risk_colors, axis=1))

# Export
st.download_button("📤 Ladda ner resultat som CSV", df_visning.to_csv(index=False), file_name="konsultrisk_resultat.csv")

# Modellutvärdering
st.subheader("📉 Modellutvärdering")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))
