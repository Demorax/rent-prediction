import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Kolik zaplatím za nájem?", layout="centered")

BUILDING_TYPES = {
    "Cihlova": "BRICK",
    "Panelova": "PANEL",
    "Smisena": "MIXED",
    "Skeletova": "SKELET",
    "Kamenna": "STONE",
    "Drevena": "WOODEN",
    "Montovana": "ASSEMBLED",
    "Jina": "OTHER",
    "Neuvedeno": "UNDEFINED",
}

CONDITIONS = {
    "Velmi dobry": "VERY_GOOD",
    "Dobry": "GOOD",
    "Novostavba": "NEW",
    "Po rekonstrukci": "AFTER_RECONSTRUCTION",
    "Pred rekonstrukci": "BEFORE_RECONSTRUCTION",
    "Udrzovany": "MAINTAINED",
    "Neuvedeno": "UNDEFINED",
}

DISPOSITIONS = {
    "1+kk": "DISP_1_KK",
    "1+1": "DISP_1_1",
    "2+kk": "DISP_2_KK",
    "2+1": "DISP_2_1",
    "3+kk": "DISP_3_KK",
    "3+1": "DISP_3_1",
    "4+kk": "DISP_4_KK",
    "4+1": "DISP_4_1",
    "5+kk": "DISP_5_KK",
    "5+1": "DISP_5_1",
}

EQUIPMENT = {"Neuvedeno": "UNDEFINED", "Nezarizeno": "UNFURNISHED", "Castecne": "PARTIALLY", "Plne zarizeno": "FURNISHED"}
PENB = {"G (nejhorsi)": "G", "F": "F", "E": "E", "D": "D", "C": "C", "B": "B", "A (nejlepsi)": "A"}

REGIONS = [
    "Hlavni mesto Praha",
    "Stredocesky kraj",
    "Jihocesky kraj",
    "Plzensky kraj",
    "Karlovarsky kraj",
    "Ustecky kraj",
    "Liberecky kraj",
    "Kralovehradecky kraj",
    "Pardubicky kraj",
    "Kraj Vysocina",
    "Jihomoravsky kraj",
    "Olomoucky kraj",
    "Zlinsky kraj",
    "Moravskoslezsky kraj",
]


def check_api_health():
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def call_prediction_api(data: dict) -> dict | None:
    try:
        resp = httpx.post(f"{API_URL}/predict", json=data, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error: {resp.status_code}")
        return None
    except httpx.ConnectError:
        st.error("API neni dostupne. Spustte backend: uvicorn src.api.main:app")
        return None
    except Exception as e:
        st.error(f"Chyba: {e}")
        return None


def main():
    st.title("Kolik zaplatím za nájem?")
    st.caption("Zadej parametry bytu a model ti řekne, kolik asi zaplatíš.")
    st.divider()

    api_ok = check_api_health()
    if not api_ok:
        st.warning("Backend neni spuštěný. Spusť: `uvicorn src.api.main:app --reload`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lokalita")
        city = st.text_input("Město", value="Praha")
        region = st.selectbox("Kraj", REGIONS, index=0)

        st.subheader("Velikost")
        floor_space = st.slider("Podlahová plocha (m²)", 10, 250, 50)
        disposition_label = st.selectbox("Dispozice", list(DISPOSITIONS.keys()), index=2)

    with col2:
        st.subheader("Typ a stav")
        building_type_label = st.selectbox("Typ budovy", list(BUILDING_TYPES.keys()), index=0)
        condition_label = st.selectbox("Stav bytu", list(CONDITIONS.keys()), index=0)

        st.subheader("Vybavení")
        equipment_label = st.selectbox("Vybavení", list(EQUIPMENT.keys()), index=0)
        penb_label = st.selectbox("Energetická třída", list(PENB.keys()), index=4)

    st.divider()

    if st.button("Odhadnout nájem", type="primary", use_container_width=True):
        if not api_ok:
            st.error("Backend není dostupný.")
            return

        data = {
            "city": city,
            "region": region,
            "floor_space": floor_space,
            "land_space": 0,
            "disposition": DISPOSITIONS[disposition_label],
            "building_type": BUILDING_TYPES[building_type_label],
            "condition": CONDITIONS[condition_label],
            "equipment": EQUIPMENT[equipment_label],
            "penb": PENB[penb_label],
        }

        with st.spinner("Počítám..."):
            result = call_prediction_api(data)

        if result:
            price = result["predicted_rent"]
            st.success(f"Odhadované nájemné: **{price:,.0f} Kč/měsíc**")
            st.info(f"Reálný rozsah bývá {max(0, price - 2500):,.0f} – {price + 2500:,.0f} Kč (MAPE modelu ~15 %)")

            with st.expander("Zadané parametry"):
                st.json(data)

    st.divider()
    with st.expander("O modelu"):
        st.write("""
        **Model:** XGBoost trénovaný na ~20 000 inzerátech z českých realitních portálů

        **Přesnost:** MAPE ~15 % (průměrná relativní odchylka od skutečné ceny)

        **Optimalizace:** hyperparametry nalezeny pomocí Optuna (Bayesian search)
        """)


if __name__ == "__main__":
    main()
