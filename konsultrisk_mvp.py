import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Konsultriskanalys", layout="wide")
st.title("🔍 Konsultriskanalys – MVP")

# Dummydata
def generate_dummy_data(n=100):
    np.random.seed(42)
    df = pd.DataFrame({
        'Konsult': [f'Konsult {i+1}' for i in range(n)],
        'Dagar kvar på uppdrag': np.random.randint(0, 90, n),
        'Tidigare lediga perioder': np.random.randint(0, 5, n),
        'Kompetensmatch (0-1)': np.round(np.random.rand(n), 2),
        'Aktiva säljcase': np.random.randint(0, 3, n),
        'Faktureringsgrad (%)': np.random.randint(40, 101, n),
    })
    df['Risk (1=risk, 0=ingen risk)'] = ((df['Dagar kvar på uppdrag'] < 15) & (df['Aktiva säljcase'] == 0)).astype(int)
    df['Förväntad ledtid (dagar)'] = np.random.randint(5, 30, n)
    return df

# Ladda data
st.sidebar.header("📥 Ladda eller generera data")
data_option = st.sidebar.radio("Välj datakälla:", ["Generera testdata", "Ladda upp egen CSV"])

if data_option == "Generera testdata":
    df = generate_dummy_data(100)
else:
    uploaded_file = st.sidebar.file_uploader("Ladda upp CSV-fil", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Vänligen ladda upp en CSV-fil.")
        st.stop()

# Visa data
st.subheader("📊 Konsultdata")
st.dataframe(df)

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

# Modell 2: När bör man agera?
st.subheader("⏰ Modell 2 – Rekommenderad åtgärdstidpunkt")

def calculate_action_day(row):
    return max(row['Dagar kvar på uppdrag'] - row['Förväntad ledtid (dagar)'], 0)

df['Rekommenderad åtgärdsdag'] = df.apply(calculate_action_day, axis=1)

# Visualisering
st.subheader("📈 Visualisering av risk och åtgärdstid")
fig, ax = plt.subplots()
ax.scatter(df['Dagar kvar på uppdrag'], df['Riskprocent (%)'], c=df['Predikterad risk'], cmap='coolwarm', label='Risk')
ax.set_xlabel('Dagar kvar på uppdrag')
ax.set_ylabel('Riskprocent (%)')
ax.set_title('Risknivå beroende på uppdragets längd')
st.pyplot(fig)

# Visa slutresultat
st.subheader("📌 Resultat")
st.dataframe(df[['Konsult', 'Dagar kvar på uppdrag', 'Faktureringsgrad (%)', 'Riskprocent (%)', 'Predikterad risk', 'Rekommenderad åtgärdsdag']].sort_values(by='Riskprocent (%)', ascending=False))

# Modellutvärdering
st.subheader("📉 Modellutvärdering")
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

# Export
st.download_button("📤 Ladda ner resultat som CSV", df.to_csv(index=False), file_name="konsultrisk_resultat.csv")
