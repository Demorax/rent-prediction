import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# Konfigurace stránky
st.set_page_config(
    page_title="Predikce nájemného bytu",
    page_icon="🏠",
    layout="centered"
)

# Načtení modelů a preprocessing objektů
@st.cache_resource
def load_models():
    model = xgb.XGBRegressor()
    model.load_model('models/xgboost.json')
    preprocessor = joblib.load('models/preprocessor.joblib')
    target_encoder = joblib.load('models/target_encoder.joblib')
    mappings = joblib.load('models/mappings.joblib')
    return model, preprocessor, target_encoder, mappings


# Možnosti výběru
BUILDING_TYPES = ['BRICK', 'PANEL', 'MIXED', 'SKELET', 'UNDEFINED', 'OTHER', 'STONE', 'ASSEMBLED', 'WOODEN']
CONDITIONS = ['VERY_GOOD', 'GOOD', 'NEW', 'AFTER_RECONSTRUCTION', 'UNDEFINED', 'BEFORE_RECONSTRUCTION', 'MAINTAINED']
DISPOSITIONS = ['1+kk', '1+1', '2+kk', '2+1', '3+kk', '3+1', '4+kk', '4+1', '5+kk', '5+1']
EQUIPMENT_OPTIONS = {'Neuvedeno': 0, 'Nezařízeno': 1, 'Částečně zařízeno': 2, 'Plně zařízeno': 3}
PENB_OPTIONS = {'G (nejhorší)': 0, 'F': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A (nejlepší)': 6}

REGIONS = [
    'Praha', 'Středočeský kraj', 'Jihočeský kraj', 'Plzeňský kraj',
    'Karlovarský kraj', 'Ústecký kraj', 'Liberecký kraj', 'Královéhradecký kraj',
    'Pardubický kraj', 'Vysočina', 'Jihomoravský kraj', 'Olomoucký kraj',
    'Zlínský kraj', 'Moravskoslezský kraj'
]

# České popisky pro condition
CONDITION_LABELS = {
    'VERY_GOOD': 'Velmi dobrý',
    'GOOD': 'Dobrý',
    'NEW': 'Novostavba',
    'AFTER_RECONSTRUCTION': 'Po rekonstrukci',
    'UNDEFINED': 'Neuvedeno',
    'BEFORE_RECONSTRUCTION': 'Před rekonstrukcí',
    'MAINTAINED': 'Udržovaný'
}

# České popisky pro building_type
BUILDING_TYPE_LABELS = {
    'BRICK': 'Cihlová',
    'PANEL': 'Panelová',
    'MIXED': 'Smíšená',
    'SKELET': 'Skeletová',
    'UNDEFINED': 'Neuvedeno',
    'OTHER': 'Jiná',
    'STONE': 'Kamenná',
    'ASSEMBLED': 'Montovaná',
    'WOODEN': 'Dřevěná'
}


def main():
    st.title("Predikce měsíčního nájemného")
    st.markdown("Zadejte parametry bytu a zjistěte odhadovanou cenu nájmu v ČR.")
    st.divider()

    # Načtení modelů
    try:
        model, preprocessor, target_encoder, mappings = load_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Chyba při načítání modelů: {e}")
        st.info("Pro správné fungování nahrajte soubory modelů do složky 'models/'")
        models_loaded = False

    # Formulář pro vstupní data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Lokalita")
        city = st.text_input("Město / Čtvrť", value="Praha", help="Název města nebo městské čtvrti")
        region = st.selectbox("Kraj", REGIONS, index=0)

        st.subheader("📐 Velikost")
        floor_space = st.slider("Podlahová plocha (m²)", 10, 250, 50)
        disposition = st.selectbox("Dispozice", DISPOSITIONS, index=2)

    with col2:
        st.subheader("🏗️ Typ a stav")
        building_type_label = st.selectbox(
            "Typ budovy",
            list(BUILDING_TYPE_LABELS.values()),
            index=0
        )
        building_type = [k for k, v in BUILDING_TYPE_LABELS.items() if v == building_type_label][0]

        condition_label = st.selectbox(
            "Stav bytu",
            list(CONDITION_LABELS.values()),
            index=0
        )
        condition = [k for k, v in CONDITION_LABELS.items() if v == condition_label][0]

        st.subheader("⚡ Vybavení")
        equipment_label = st.selectbox("Vybavení", list(EQUIPMENT_OPTIONS.keys()), index=0)
        penb_label = st.selectbox("Energetická třída (PENB)", list(PENB_OPTIONS.keys()), index=2)

    st.divider()

    # Predikce
    if st.button("Predikovat cenu nájmu", type="primary", use_container_width=True):
        if not models_loaded:
            st.error("Modely nejsou načteny. Nelze provést predikci.")
            return

        with st.spinner("Počítám predikci..."):
            try:
                # Příprava dat
                input_data = {
                    'city': city,
                    'region': region,
                    'disposition': disposition,
                    'building_type': building_type,
                    'condition': condition,
                    'floor_space': floor_space,
                    'land_space': 0,
                    'equipment': EQUIPMENT_OPTIONS[equipment_label],
                    'penb': PENB_OPTIONS[penb_label]
                }

                df_input = pd.DataFrame([input_data])

                # Target encoding
                target_cols = ['city', 'region', 'disposition']
                df_input[target_cols] = target_encoder.transform(df_input[target_cols])

                feature_columns = ['city', 'floor_space', 'land_space', 'region', 'disposition',
                                   'equipment', 'penb', 'condition', 'building_type']
                df_for_preprocessing = df_input[feature_columns]

                X_processed = preprocessor.transform(df_for_preprocessing)

                # Predikce
                prediction = model.predict(X_processed)[0]

                st.success(f"### Odhadované měsíční nájemné: **{prediction:,.0f} Kč**")

                st.info(f"Realistický rozsah: **{max(0, prediction - 2500):,.0f} - {prediction + 2500:,.0f} Kč**")

                # Shrnutí parametrů
                with st.expander("Shrnutí zadaných parametrů"):
                    st.markdown(f"""
                    | Parametr | Hodnota |
                    |----------|---------|
                    | Lokalita | {city}, {region} |
                    | Plocha | {floor_space} m² |
                    | Dispozice | {disposition} |
                    | Typ budovy | {building_type_label} |
                    | Stav | {condition_label} |
                    | Vybavení | {equipment_label} |
                    | PENB | {penb_label} |
                    """)

            except Exception as e:
                st.error(f"Chyba při predikci: {e}")
                st.exception(e)

    # Footer
    st.divider()
    with st.expander("O aplikaci"):
        st.markdown("""
        **Model:** XGBoost (Gradient Boosting) s Optuna hyperparameter tuningem

        **Dataset:** ~20 000 bytů z českých realitních portálů (Bezrealitky, Sreality, iDNES)

        **Přesnost:** MAE (Mean Absolute Error) ~2 500 Kč

        **Autor:** [GitHub](https://github.com/Demorax)
        """)


if __name__ == "__main__":
    main()
