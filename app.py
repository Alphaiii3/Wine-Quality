import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========================
# Load model and data
# ========================
@st.cache_data
def load_data():
  df = pd.read_csv("data/winequality-dataset_updated.csv")
  return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


df = load_data()
model= load_model()

# ========================
# App Title & Description
# ========================
st.set_page_config(page_title="Wine Quality Prediction App", layout="wide")
st.title("🍷 Wine Quality Prediction App")
st.markdown("""
This app predicts **wine quality** (good vs not good) from its chemical properties.  
It includes:
- Data exploration  
- Visualizations  
- Real-time predictions  
- Model performance metrics  
""")

# ========================
# Sidebar Navigation
# ========================
menu = st.sidebar.radio(
    "Navigation",
    ["Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# Feature list
features = ['fixed acidity','volatile acidity','citric acid','residual sugar',
            'chlorides','free sulfur dioxide','total sulfur dioxide',
            'density','pH','sulphates','alcohol']

# ========================
# Data Exploration Section
# ========================
if menu == "Data Exploration":
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:", df.dtypes)

    st.markdown("#### Sample Data")
    st.dataframe(df.head())

    # Interactive filter
    st.markdown("#### Filter Data")
    quality_filter = st.multiselect("Select quality values", sorted(df['quality'].unique()))
    if quality_filter:
        st.dataframe(df[df['quality'].isin(quality_filter)])
    else:
        st.dataframe(df)

# ========================
# Visualisation Section
# ========================
elif menu == "Visualisations":
    st.subheader("📈 Visualisations")

    # Chart 1 - Histogram
    col1 = st.selectbox("Select column for histogram", features)
    fig1 = px.histogram(df, x=col1, nbins=30, title=f"Histogram of {col1}")
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 - Scatter plot
    x_axis = st.selectbox("X-axis", features, index=0)
    y_axis = st.selectbox("Y-axis", features, index=1)
    fig2 = px.scatter(df, x=x_axis, y=y_axis, color='quality', title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - Correlation heatmap
    corr = df[features + ['quality']].corr()
    fig3, ax = plt.subplots()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# ========================
# Model Prediction Section
# ========================
elif menu == "Model Prediction":
    st.subheader("🤖 Make a Prediction")

    inputs = []
    for f in features:
        default_val = float(df[f].median())
        val = st.number_input(f"Enter {f}", value=default_val)
        inputs.append(val)

    if st.button("Predict"):
        try:
            with st.spinner("Making prediction..."):
                X_user = np.array(inputs).reshape(1, -1)
                X_user_s = scaler.transform(X_user)
                pred = model.predict(X_user_s)[0]

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_user_s).max()
                    st.success(f"Prediction: {'Good' if pred==1 else 'Not Good'}")
                    st.info(f"Confidence: {proba:.2f}")
                else:
                    st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# ========================
# Model Performance Section
# ========================
elif menu == "Model Performance":
    st.subheader("📊 Model Performance Metrics")

    # Prepare data
X = df[features]
y = (df['quality'] >= 7).astype(int)

# Remove scaler transform step
# X_s = scaler.transform(X)

# Use X directly for prediction
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Not Good', 'Good'], yticklabels=['Not Good', 'Good'], ax=ax)
st.pyplot(fig_cm)

# Dummy model comparison (if you have results for multiple models)
st.markdown("#### Model Comparison")
comp_df = pd.DataFrame({
    'Model': ['RandomForest', 'LogisticRegression'],
    'Accuracy': [acc, 0.85]  # Replace 0.85 with real score
})
st.dataframe(comp_df)
