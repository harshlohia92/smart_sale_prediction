import streamlit as st
import plotly.graph_objects as go
from prediction_helper import predict_sales

st.set_page_config(page_title="Sales Prediction App", layout="centered")

st.title("📊 Sales Prediction Dashboard")

with st.form("sales_form"):
    st.subheader("Enter Input Features")

    quantity = st.number_input("Quantity", min_value=1, value=10)
    unitprice = st.number_input("Unit Price", min_value=1.0, value=20.0)
    country = st.selectbox("Country", [
        'United Kingdom', 'France', 'Australia', 'Germany', 'Norway', 'EIRE',
        'Switzerland', 'Poland', 'Portugal', 'Italy', 'Belgium', 'Lithuania',
        'Japan', 'Iceland', 'Channel Islands', 'Denmark', 'Spain', 'Cyprus',
        'Austria', 'Sweden', 'Netherlands', 'Israel', 'Finland', 'Greece',
        'Hong Kong', 'Singapore', 'Lebanon', 'United Arab Emirates', 'Saudi Arabia',
        'Czech Republic', 'Canada', 'Unspecified', 'Brazil', 'USA',
        'European Community', 'Bahrain', 'Malta', 'RSA'
    ])
    months = st.slider("Month", 1, 12, 6)
    week = st.selectbox("Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday'])
    supplier = st.selectbox("Supplier", ['Supplier_1', 'Supplier_2', 'Supplier_3', 'Supplier_4', 'Supplier_5'])
    supplier_avg_sale = st.number_input("Supplier Avg Sale", min_value=0.0, value=50.0)
    is_christmas = st.selectbox("Is Christmas?", [0, 1])
    is_newyear = st.selectbox("Is New Year?", [0, 1])
    rolling_mean_30_x = st.number_input("Rolling Mean 30 (X)", min_value=0.0, value=20.0)
    is_weekend = st.selectbox("Is Weekend?", [0, 1])

    submitted = st.form_submit_button("🔮 Predict Sales")

if submitted:
    predicted_actual = predict_sales(
        quantity, unitprice, country, months, week, supplier,
        supplier_avg_sale, is_christmas, is_newyear, rolling_mean_30_x, is_weekend
    )


    if predicted_actual < 200:
        level = "Low"
        gauge_color = "red"
    elif predicted_actual < 400:
        level = "Medium"
        gauge_color = "orange"
    else:
        level = "High"
        gauge_color = "green"

    st.subheader("Prediction Results")
    st.write(f"💰 **Predicted Sales (actual units):** {predicted_actual:.2f}")


    color_map = {"Low": "red", "Medium": "orange", "High": "green"}
    st.markdown(f"<h3 style='color:{color_map[level]};'>Sales Level: {level}</h3>", unsafe_allow_html=True)


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_actual,
        title={'text': "Predicted Sales"},
        gauge={
            'axis': {'range': [0, max(1000, predicted_actual * 1.2)]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 200], 'color': "lightcoral"},
                {'range': [200, 400], 'color': "orange"},
                {'range': [400, max(1000, predicted_actual * 1.2)], 'color': "lightgreen"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': predicted_actual
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)