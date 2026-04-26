import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load Models ---
def load_model(filename):
    with open(f"D:/AIML_Clickstream-Customer Conversion/models/{filename}", "rb") as file:
        model = pickle.load(file)
    return model


# Load your models
models = {
    "Linear Regression": load_model("LinearRegression_pipeline.pkl"),
    "Decision Tree (Regression)": load_model("GradientBoosting_pipeline.pkl"),
    "Lasso Regression": load_model("Lasso_pipeline.pkl"),
    "Ridge Regression": load_model("Ridge_pipeline.pkl"),
    "Logistic Regression (Classification)": load_model("LogisticRegression_pipeline.pkl"),
    "Random Forest (Classification)": load_model("RandomForest_pipeline.pkl"),
    "Neural Network (Classification)": load_model("NeuralNet_pipeline.pkl"),
    "XGBoost (Classification)": load_model("XGBoost_pipeline.pkl"),
    "KMeans (Clustering)": load_model("KMeans_clustering.pkl"),
    "DBSCAN (Clustering)": load_model("DBSCAN_clustering.pkl")
}

# --- Preprocessing Function ---
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    if "price" in df.columns:
        df.dropna(subset=["price"], inplace=True)
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df.drop('date', axis=1, inplace=True)
    return df

# --- Streamlit App ---
st.title("🛍️ Customer Analysis App - Regression | Classification | Clustering")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data:")
    st.write(df.head())

    df = preprocess_data(df)

    # Define features and targets
    x_reg = df.drop(["price", "price_2"], axis=1, errors='ignore')
    y_reg = df['price'] if "price" in df.columns else None
    x_cls = df.drop("price_2", axis=1, errors='ignore')
    y_cls = df["price_2"].replace({2:0}) if "price_2" in df.columns else None

    # Columns
    categorical_cols = ["page_2_clothing_model", "order_level"]
    numerical_cols = ["colour", "page_1_main_category", "location", "model_photography", "session_id", "country", "year", "month", "day"]

    # Preprocessor
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # X_processed = preprocessor.fit_transform(x_reg)

    # --- Prediction Section ---
    st.header("🔮 Make Predictions")

    task = st.selectbox("Select Task", ["Regression", "Classification", "Clustering"])

    if task == "Regression":
        model_name = st.selectbox("Select Regression Model", ["Linear Regression", "Decision Tree (Regression)", "Lasso Regression", "Ridge Regression"])
        model = models[model_name]

        if st.button("Predict Price"):
            preds = model.predict(x_reg)
            df['Predicted Price'] = preds
            st.write(df[["price", "Predicted Price"]].head())

    elif task == "Classification":
        model_name = st.selectbox("Select Classification Model", ["Logistic Regression (Classification)", "Random Forest (Classification)", "Neural Network (Classification)", "XGBoost (Classification)"])
        model = models[model_name]

        if st.button("Predict Conversion"):
            preds = model.predict(x_cls)
            df['Predicted Conversion'] = preds
            st.write(df[["price_2", "Predicted Conversion"]].head())

    elif task == "Clustering":
        model_name = st.selectbox("Select Clustering Model", ["KMeans (Clustering)", "DBSCAN (Clustering)"])
        model = models[model_name]

        if st.button("Cluster Customers"):
            if hasattr(model, "predict"):
                clusters = model.predict(x_cls)
            else:
                clusters = model.fit_predict(x_cls)

            df['Cluster'] = clusters
            st.write(df[['Cluster']].head())

            # Visualize clusters
            st.subheader("🧩 Cluster Visualization")
            try:
                import matplotlib.pyplot as plt
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                components = pca.fit_transform(x_cls.toarray() if hasattr(x_cls, "toarray") else X_processed)

                fig, ax = plt.subplots()
                scatter = ax.scatter(components[:, 0], components[:, 1], c=clusters, cmap='tab10')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

    # --- Visualization Section ---
    st.header("📊 Data Visualizations")
    plot_type = st.selectbox("Select Plot Type", ["Bar Chart", "Pie Chart", "Histogram"])

    column = st.selectbox("Select Column for Visualization", df.columns)

    if plot_type == "Bar Chart":
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    elif plot_type == "Pie Chart":
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

    elif plot_type == "Histogram":
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=30)
        st.pyplot(fig)
