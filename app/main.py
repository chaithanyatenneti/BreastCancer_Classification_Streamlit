import streamlit as st
import pickle  
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():

    # Load the data
    data = pd.read_csv("data/data.csv")
    
    # Drop unnecessary columns
    data = data.drop(columns=["Unnamed: 32", "id"], axis=1)
    
    # Map diagnosis to 0 and 1 
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.title("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)","radius_mean"),
        ("Texture (mean)","texture_mean"),
        ("Perimeter (mean)","perimeter_mean"),
        ("Area (mean)","area_mean"),
        ("Smoothness (mean)","smoothness_mean"),
        ("Compactness (mean)","compactness_mean"),
        ("Concavity (mean)","concavity_mean"),
        ("Concave points(mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"), 
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothenss (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")    
    ]

    input = {}


    for label, feature in slider_labels:
        input[feature] = st.sidebar.slider(label, min_value=data[feature].min(), max_value=data[feature].max(), value=data[feature].mean())

    return input


def scaled_input(input_data):
    data = get_clean_data()

    X = data.drop(columns=["diagnosis"], axis=1)

    scaled_input_data ={}

    for features, values in input_data.items():
        max_value = X[features].max()
        min_value = X[features].min()
        scaled_value = (values - min_value) / (max_value - min_value)
        scaled_input_data[features] = scaled_value

    return scaled_input_data


def get_radar_chart(input_data):

    input_data = scaled_input(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']       
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_data_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_array_scaled = scaler.transform(input_data_array)
    prediction = model.predict(input_data_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_data_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_data_array_scaled)[0][1])

    st.write("This application is developed to help the radiologists with their diagnosis, hence should not be used for final diagnosis.")


def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        page_icon=":xray:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Prediction")
        st.write("This application is used to predict breast cancer using a machine-learning approach using the measurement data of the cell clusters.")

    col1 , col2 =   st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)


    with col2:
        
        add_predictions(input_data)



if __name__ == "__main__":
    main()
