# --- Importer un exemple du dataset ---
example_row = None
example_idx = None

st.write("Choisissez une ligne du dataset à importer comme exemple (optionnel):")
row_options = df.index.tolist()
row_display = [f"Ligne {i+1}" for i in row_options]
selected_row = st.selectbox("Sélectionnez le numéro de ligne :", row_display, index=0)
example_button = st.button("Importer la ligne sélectionnée")

if example_button:
    example_idx = row_options[row_display.index(selected_row)]
    st.success(f"Exemple importé : ligne {example_idx+1} du dataset")
    example_row = df.iloc[example_idx]
else:
    example_row = None

user_input = {}

def clean_numeric(val):
    # Try to cast to float or int as appropriate
    try:
        if pd.isnull(val):
            return None
        # Try int first, fallback to float
        try:
            return int(val)
        except Exception:
            return float(val)
    except Exception:
        return None

for feature in features:
    if categorical_map and feature in categorical_map:
        options = categorical_map[feature]
        if example_row is not None:
            # Ensure value is int (handles possible float or str)
            val = example_row[feature]
            val = clean_numeric(val)
            try:
                selection = options[int(val)] if val is not None and int(val) < len(options) else options[0]
            except Exception:
                selection = options[0]
        else:
            selection = options[0]
        selection = st.selectbox(f"{feature}", options, key=feature, index=options.index(selection))
        encoded = options.index(selection)
        user_input[feature] = encoded
    elif page_choice == "Diabetes2 (Custom2)" and feature == "Age":
        original_age_min = int(df["Age"].min())
        original_age_max = int(df["Age"].max())
        if example_row is not None:
            user_val = clean_numeric(example_row["Age"])
            if user_val is None:
                user_val = int(df["Age"].mean())
        else:
            user_val = int(df["Age"].mean())
        user_val = st.number_input(
            f"{feature} (original scale)",
            min_value=original_age_min, max_value=original_age_max,
            value=int(user_val),
            key=feature
        )
        # Normaliser comme dans le script d'origine
        user_input[feature] = (user_val - df["Age"].mean()) / df["Age"].std()
    else:
        if example_row is not None:
            value = clean_numeric(example_row[feature])
        else:
            value = None
        user_input[feature] = show_numeric_input(df, feature, value=value)
