import streamlit as st
from helper import predict

st.title('Vehicle_Detection')

uploaded = st.file_uploader('Upload image', type=['jpg','png'])

if uploaded:
    image = "tem.jpg"
    with open(image,"wb") as f:
        f.write(uploaded.getbuffer())
        st.image(uploaded, caption='Uploaded Image',  use_container_width =True)
        prediction = predict(image)
        st.info(f'prediction_class: {prediction}')

