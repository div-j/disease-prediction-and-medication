import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('libs/disease_prediction_model.joblib')

# Load the label encoder
@st.cache_resource
def load_label_encoder():
    return joblib.load('libs/label_encoder.joblib')

#load dataset
@st.cache_data
def load_data():
    description = pd.read_csv('datasets/description.csv')
    precautions = pd.read_csv('datasets/precautions_df.csv')
    diets = pd.read_csv('datasets/diets.csv')
    workout = pd.read_csv('datasets/workout_df.csv')
    medications = pd.read_csv('datasets/medications.csv')
    return description, precautions, diets, workout, medications


def predict_disease(symptoms_list):
    """
    Predicts the disease based on a list of symptoms and retrieves related information.

    Args:
      symptoms_list: A list of strings representing the symptoms.

    Returns:
      A dictionary containing the predicted disease and related
      information (description, precautions, diets, workouts, medications), 
      or an error message if the predicted disease is not found.
    """
    try:
        model = load_model()
        label_encoder = load_label_encoder()
        
        # Convert the list of symptoms to a DataFrame
        symptoms_df = pd.DataFrame([symptoms_list], columns=model.feature_names_in_)
        
        # Predict the disease
        predicted_disease = model.predict(symptoms_df)[0]
        
        # Decode the predicted disease
        decoded_disease = label_encoder.inverse_transform([predicted_disease])[0]
        
        # Load related information
        description, precautions, diets, workout, medications = load_data()
        
        # Initialize result dictionary
        result = {"disease": decoded_disease}
        
        # Debug: Print column names for troubleshooting
        # st.write(f"Debug - Description columns: {description.columns.tolist()}")
        # st.write(f"Debug - Precautions columns: {precautions.columns.tolist()}")
        # st.write(f"Debug - Diets columns: {diets.columns.tolist()}")
        # st.write(f"Debug - Workout columns: {workout.columns.tolist()}")
        # st.write(f"Debug - Medications columns: {medications.columns.tolist()}")
        
        # Get description
        try:
            desc_row = description[description['Disease'] == decoded_disease]
            if not desc_row.empty:
                result["description"] = desc_row['Description'].values[0]
            else:
                result["description"] = "Description not available for this disease."
        except KeyError as e:
            result["description"] = f"Error accessing description data: {str(e)}"
        
        # Get precautions
        try:
            prec_row = precautions[precautions['Disease'] == decoded_disease]
            if not prec_row.empty:
                prec_list = []
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in prec_row.columns:
                        val = prec_row[col].values[0]
                        if pd.notna(val) and str(val).strip() and str(val).strip() != '':
                            prec_list.append(str(val))
                result["precautions"] = prec_list if prec_list else ["No specific precautions available."]
            else:
                result["precautions"] = ["Precautions not available for this disease."]
        except KeyError as e:
            result["precautions"] = [f"Error accessing precautions data: {str(e)}"]
        
        # Get diets
        try:
            diet_row = diets[diets['Disease'] == decoded_disease]
            if not diet_row.empty:
                result["diets"] = diet_row['Diet'].values[0]
            else:
                result["diets"] = "Diet information not available for this disease."
        except KeyError as e:
            result["diets"] = f"Error accessing diet data: {str(e)}"
        
        # Get workout (note: workout_df uses lowercase 'disease')
        try:
            workout_row = workout[workout['disease'] == decoded_disease]
            if not workout_row.empty:
                result["workout"] = workout_row['workout'].values[0]
            else:
                result["workout"] = "Workout information not available for this disease."
        except KeyError as e:
            result["workout"] = f"Error accessing workout data: {str(e)}"
        
        # Get medications
        try:
            med_row = medications[medications['Disease'] == decoded_disease]
            if not med_row.empty:
                result["medications"] = med_row['Medication'].values[0]
            else:
                result["medications"] = "Medication information not available for this disease."
        except KeyError as e:
            result["medications"] = f"Error accessing medication data: {str(e)}"
        
        return result
    
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}


# Streamlit App
def main():
    st.set_page_config(
        page_title="Disease Prediction & Medication System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Disease Prediction & Medication System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                    ["Disease Prediction", "About"])
    
    if app_mode == "Disease Prediction":
        st.header("Disease Prediction Based on Symptoms")
        
        # Load model to get feature names (symptoms)
        try:
            model = load_model()
            available_symptoms = list(model.feature_names_in_)
            
            st.write("Please select the symptoms you are experiencing:")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            selected_symptoms = []
            
            # Split symptoms into two columns for better display
            mid_point = len(available_symptoms) // 2
            
            with col1:
                st.subheader("Symptoms (Part 1)")
                for symptom in available_symptoms[:mid_point]:
                    if st.checkbox(symptom.replace('_', ' ').title(), key=f"sym1_{symptom}"):
                        selected_symptoms.append(symptom)
            
            with col2:
                st.subheader("Symptoms (Part 2)")
                for symptom in available_symptoms[mid_point:]:
                    if st.checkbox(symptom.replace('_', ' ').title(), key=f"sym2_{symptom}"):
                        selected_symptoms.append(symptom)
            
            # Show selected symptoms
            if selected_symptoms:
                st.write("**Selected Symptoms:**")
                st.write(", ".join([s.replace('_', ' ').title() for s in selected_symptoms]))
            
            # Predict button
            if st.button("Predict Disease", type="primary"):
                if selected_symptoms:
                    # Create binary vector for symptoms
                    symptoms_vector = [1 if symptom in selected_symptoms else 0 for symptom in available_symptoms]
                    
                    # Show loading spinner
                    with st.spinner("Analyzing symptoms..."):
                        result = predict_disease(symptoms_vector)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Display results
                        st.success(f"**Predicted Disease: {result['disease']}**")
                        
                        # Create tabs for different information
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Description", "⚠️ Precautions", "🥗 Diet", "💪 Workout", "💊 Medications"])
                        
                        with tab1:
                            st.write("**Description:**")
                            st.write(result["description"])
                        
                        with tab2:
                            st.write("**Precautions:**")
                            if isinstance(result["precautions"], list):
                                for i, precaution in enumerate(result["precautions"], 1):
                                    st.write(f"{i}. {precaution}")
                            else:
                                st.write(result["precautions"])
                        
                        with tab3:
                            st.write("**Diet Recommendations:**")
                            st.write(result["diets"])
                        
                        with tab4:
                            st.write("**Workout Recommendations:**")
                            st.write(result["workout"])
                        
                        with tab5:
                            st.write("**Medication Information:**")
                            st.write(result["medications"])
                            
                        # Disclaimer
                        st.warning("⚠️ **Disclaimer:** This is an AI-based prediction system. Please consult with a healthcare professional for proper diagnosis and treatment.")
                else:
                    st.warning("Please select at least one symptom to make a prediction.")
        
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")
            st.info("Please make sure all model files are in the correct location.")
    
    elif app_mode == "About":
        st.header("About This Application")
        st.write("""
        This Disease Prediction & Medication System is an AI-powered tool that helps predict potential diseases based on symptoms.
        
        **Features:**
        - Disease prediction based on multiple symptoms
        - Detailed disease descriptions
        - Personalized precautions
        - Diet recommendations
        - Workout suggestions
        - Medication information
        
        **How to use:**
        1. Select the symptoms you are experiencing from the checklist
        2. Click the "Predict Disease" button
        3. Review the predicted disease and recommendations
        
        **Important Note:**
        This tool is for educational and informational purposes only. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment.
        """)


if __name__ == "__main__":
    main()
    