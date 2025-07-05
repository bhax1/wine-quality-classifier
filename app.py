import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt

# Load model
model = joblib.load('wine_quality_model.pkl')

# App configuration
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #6a0dad;
        --secondary: #f8f9fa;
        --accent: #ff6b6b;
        --text: #2d3436;
        --success: #4caf50;
        --warning: #ff9800;
    }
    
    .main {
        background-color: #fef6ff;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #5a0b9d;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px 12px;
    }
    
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 5px solid var(--success);
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
    }
    
    .stWarning {
        background-color: #fff3e0;
        border-left: 5px solid var(--warning);
        border-radius: 0 8px 8px 0;
        padding: 1.5rem;
    }
    
    .input-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    .property-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .property-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    .tabs {
        border-bottom: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2405/2405479.png", width=120)
with col2:
    st.title("Wine Quality Classifier")
    st.markdown("**Your intelligent wine quality classifier**")
    st.caption("Analyze chemical properties to predict wine quality with machine learning")

# Main content columns
main_col, sidebar_col = st.columns([3, 1])

with main_col:
    # Input Section (Single Tab - Removed Quick Input)
    with st.container():
        st.subheader("Enter Wine Characteristics")
        
        cols = st.columns(2)
        with cols[0]:
            with st.container():
                st.markdown("#### Basic Properties")
                fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", 4.0, 16.0, 7.0, 0.1,
                                                help="Non-volatile acids that don't evaporate")
                volatile_acidity = st.number_input("Volatile Acidity (g/dm³)", 0.1, 1.5, 0.5, 0.01,
                                                   help="High levels can lead to unpleasant vinegar taste")
                citric_acid = st.number_input("Citric Acid (g/dm³)", 0.0, 1.0, 0.3, 0.01,
                                              help="Adds freshness and flavor")
                residual_sugar = st.number_input("Residual Sugar (g/dm³)", 0.0, 15.0, 2.0, 0.1,
                                                 help="Amount of sugar remaining after fermentation")
        
        with cols[1]:
            with st.container():
                st.markdown("#### Advanced Properties")
                chlorides = st.number_input("Chlorides (g/dm³)", 0.01, 0.9, 0.05, 0.001,
                                            help="Salt content in the wine")
                sulfur_dioxide = st.number_input("Free SO₂ (mg/dm³)", 1, 80, 15, 1,
                                                 help="Prevents microbial growth and oxidation")
                total_sulfur = st.number_input("Total SO₂ (mg/dm³)", 6, 300, 46, 1,
                                               help="Total sulfur dioxide content")
                density = st.number_input("Density (g/cm³)", 0.99, 1.005, 0.995, 0.0001,
                                          help="Density of the wine")
                pH = st.number_input("pH Level", 2.8, 4.0, 3.3, 0.01,
                                     help="Acidity level on pH scale")
                sulphates = st.number_input("Sulphates (g/dm³)", 0.2, 2.0, 0.5, 0.01,
                                            help="Additives that can affect SO₂ levels")
                alcohol = st.number_input("Alcohol (% vol)", 8.0, 15.0, 10.0, 0.1,
                                          help="Alcohol content percentage")
    
    # Prediction Button
    predict_btn = st.button("🔍 Analyze Wine Quality", type="primary", use_container_width=True)
    
    # Results Section
    if predict_btn:
        features = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": sulfur_dioxide,
            "total sulfur dioxide": total_sulfur,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol
        }
        
        input_data = np.array([list(features.values())]).reshape(1, -1)
        
        try:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            prob = proba.max()
            
            if prediction == 1:
                st.balloons()
                with st.container():
                    st.success(f"""
                    ## 🎉 Excellent Quality Wine!
                    
                    Our analysis indicates this is a **high quality** wine with **{prob:.1%} confidence**.
                    
                    This wine meets all the key chemical markers for excellent taste and aging potential.
                    """)
            else:
                with st.container():
                    st.warning(f"""
                    ## ⚠️ Needs Improvement
                    
                    Our analysis suggests this wine is **below quality standards** with **{prob:.1%} confidence**.
                    
                    Consider adjusting the chemical balance for better results.
                    """)
            
            # Quality Metrics
            st.subheader("Quality Insights")
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Alcohol Balance", f"{alcohol}% vol", 
                         help="Ideal range: 11-13% for reds, 9-12% for whites")
            
            with cols[1]:
                st.metric("Acidity Level", f"{pH:.2f} pH", 
                         help="Ideal range: 3.0-3.4 for balanced taste")
            
            with cols[2]:
                st.metric("Sulfur Balance", f"{total_sulfur} mg/dm³", 
                         help="Ideal range: 30-100 mg/dm³")
            
            # Probability Visualization
            st.subheader("Quality Probability Distribution")
            # Data
            proba_df = pd.DataFrame({
                'Quality': ['Not Good', 'Good'],
                'Probability': [proba[0], proba[1]]
            })
            # Altair chart with horizontal text
            chart = alt.Chart(proba_df).mark_bar().encode(
                x=alt.X('Quality', axis=alt.Axis(labelAngle=0)),  # 0° = horizontal text
                y='Probability',
                color=alt.Color('Quality', scale=alt.Scale(range=['#4caf50', '#ff6b6b']))
            ).properties(
                width=400,
                height=300
            )

            st.altair_chart(chart, use_container_width=True) 
                      
            # Key Factors
            st.subheader("Key Quality Factors")
            st.markdown("""
            - **Volatile Acidity**: Should be < 0.6 g/dm³ (yours: {:.2f})
            - **Sulphates**: Ideal range 0.5-0.8 g/dm³ (yours: {:.2f})
            - **Alcohol**: Higher levels often correlate with quality (yours: {:.1f}%)
            """.format(volatile_acidity, sulphates, alcohol))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Sidebar Content
with sidebar_col:
    with st.container():
        st.subheader("🍷 About Wine Quality Classifier")
        st.markdown("""
        The **Wine Quality Classifier** uses machine learning — specifically the **XGBoost** model — to predict wine quality based on its chemical properties.
        
        ### 🧪 Quality Classification Threshold:
        - 🟢 **Good Quality**: Wine with a quality score **≥ 7**
        - 🔴 **Not Good Quality**: Wine with a quality score **< 7**

        This tool helps users assess wine quality in a simple and efficient way by leveraging data-driven insights.
        """)
    
    with st.expander("📊 Typical Value Ranges"):
        st.markdown("""
        **Red Wines:**
        - Fixed Acidity: 6-8 g/dm³
        - Volatile Acidity: 0.2-0.4 g/dm³
        - pH: 3.0-3.5
        - Alcohol: 11-13% vol
        
        **White Wines:**
        - Residual Sugar: 1-10 g/dm³
        - Total SO₂: 100-200 mg/dm³
        - Alcohol: 9-12% vol
        """)
    
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        1. **Enter** the wine's chemical properties in the input fields  
        2. **Analyze**: The model evaluates 11 key features (e.g., acidity, alcohol, pH)  
        3. **Predict**: Instantly receive a quality classification (Good or Not Good)  
        4. **Explore**: View detailed insights behind the prediction

        The model is powered by **XGBoost**, trained on thousands of labeled wine samples to deliver fast and accurate results.
        """)
    
    st.markdown("---")
    st.caption("""
    **Note:** This tool provides estimates based on chemical properties only. 
    Actual quality may vary based on other factors like vintage, terroir, and personal taste.
    """)