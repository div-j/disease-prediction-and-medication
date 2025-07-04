# Disease Prediction & Medication System

A Streamlit web application that predicts diseases based on symptoms and provides comprehensive health recommendations including precautions, diet plans, workout routines, and medication information.

## Features

- **Disease Prediction**: AI-powered disease prediction based on selected symptoms
- **Comprehensive Information**: Get detailed information about predicted diseases including:
  - Disease descriptions
  - Precautionary measures
  - Diet recommendations
  - Workout suggestions
  - Medication information
- **User-Friendly Interface**: Interactive web interface with easy symptom selection
- **Real-time Predictions**: Fast and accurate predictions using pre-trained machine learning models

## Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── datasets/                       # Data files
│   ├── description.csv            # Disease descriptions
│   ├── diets.csv                  # Diet recommendations
│   ├── medications.csv            # Medication information
│   ├── precautions_df.csv         # Precautionary measures
│   └── workout_df.csv             # Workout recommendations
└── libs/                          # Model files
    ├── disease_prediction_model.joblib  # Trained ML model
    └── label_encoder.joblib             # Label encoder for diseases
```

## Installation & Setup

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: This project requires scikit-learn version 1.6.1 specifically to maintain compatibility with the pre-trained models. Using a different version may result in warnings or errors.

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your web browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Troubleshooting

### Version Compatibility Issues
If you see warnings about scikit-learn version mismatches:
- Ensure you're using scikit-learn 1.6.1 as specified in requirements.txt
- If you must use a different version, you may need to retrain the models
- The application includes warning suppression for known compatibility issues

### Common Issues
- **Module not found**: Make sure all dependencies are installed via `pip install -r requirements.txt`
- **File not found**: Ensure you're running the app from the project root directory
- **Model loading errors**: Verify that the model files exist in the `libs/` directory

## How to Use

1. **Select Symptoms**: Choose from the available symptoms by checking the corresponding boxes
2. **Predict Disease**: Click the "Predict Disease" button to get the AI prediction
3. **Review Results**: Browse through the different tabs to see:
   - Disease description
   - Recommended precautions
   - Diet suggestions
   - Workout routines
   - Medication information

## Important Disclaimer

⚠️ **This application is for educational and informational purposes only. The predictions and recommendations provided should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical care.**

## Technical Details

- **Framework**: Streamlit
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Format**: Joblib serialized models

## Requirements

- Python 3.7+
- Streamlit 1.28.1+
- Pandas 2.0+
- NumPy 1.24+
- Scikit-learn 1.6.1 (exact version for model compatibility)
- Joblib 1.3+
