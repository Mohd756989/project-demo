import streamlit as st
import pandas as pd
import joblib
import re

# Load the model and encoders
model = joblib.load('xg_boost_model.pkl')
le_company = joblib.load('le_company.pkl')
le_typename = joblib.load('le_typename.pkl')
le_cpu = joblib.load('le_cpu.pkl')
le_gpu = joblib.load('le_gpu.pkl')
le_opsys = joblib.load('le_opsys.pkl')

# Function to parse memory
def parse_memory(mem):
    mem = mem.replace('GB', '').replace('TB', '000')
    nums = re.findall(r'\d+', mem)
    if nums:
        return int(nums[0])
    return 0

# Streamlit app
st.title('Laptop Price Prediction')

st.sidebar.header('Input Laptop Features')

# Inputs
company = st.sidebar.selectbox('Company', le_company.classes_)
typename = st.sidebar.selectbox('TypeName', le_typename.classes_)
inches = st.sidebar.number_input('Inches', min_value=10.0, max_value=20.0, value=15.6)
ram = st.sidebar.number_input('Ram (GB)', min_value=2, max_value=64, value=8)
memory = st.sidebar.text_input('Memory (e.g., 256GB SSD)', '256GB SSD')
weight = st.sidebar.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0)
cpu = st.sidebar.selectbox('Cpu', le_cpu.classes_)
gpu = st.sidebar.selectbox('Gpu', le_gpu.classes_)
opsys = st.sidebar.selectbox('OpSys', le_opsys.classes_)

# Preprocess inputs
memory_gb = parse_memory(memory)

# Encode categoricals
company_encoded = le_company.transform([company])[0]
typename_encoded = le_typename.transform([typename])[0]
cpu_encoded = le_cpu.transform([cpu])[0]
gpu_encoded = le_gpu.transform([gpu])[0]
opsys_encoded = le_opsys.transform([opsys])[0]

# Create feature array
features = [inches, ram, memory_gb, weight, company_encoded, typename_encoded, cpu_encoded, gpu_encoded, opsys_encoded]

# Predict
if st.sidebar.button('Predict Price'):
    prediction = model.predict([features])
    st.write(f'Predicted Price: ${prediction[0]:.2f}')