import mysql.connector
from flask import Flask, jsonify, request
import cv2
import numpy as np
import base64
import joblib
from sklearn.random_projection import GaussianRandomProjection

app = Flask(__name__)

# Load the trained k-NN models
with open('knn_model_binary.joblib', 'rb') as model_file:
    binary_knn = joblib.load(model_file)

with open('knn_model_multiclass.joblib', 'rb') as model_file:
    multiclass_knn = joblib.load(model_file)

# Load the StandardScaler used for normalization during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'admin',  # Updated username
    'password': '1234',  # Updated password
    'database': 'PlantDiseaseDB'
}

# Connect to the MySQL database
connection = mysql.connector.connect(**db_config)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()

def table_exists(table_name):
    try:
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        result = cursor.fetchone()
        return bool(result)
    except Exception as e:
        print(f"Error checking if table exists: {e}")
        return False
def get_plant_and_disease(prediction):
    plant_name, disease_name = prediction.split('___')
    print(plant_name , disease_name)
    return plant_name, disease_name.replace('_', ' ')
def get_disease_info(plant_name, disease_name):
    try:
        if table_exists(plant_name + 'Diseases'):
            # Query to fetch disease information
            query = "SELECT SymptomDescription, TreatmentDescription FROM {} WHERE DiseaseName = %s".format(plant_name + 'Diseases')
            cursor.execute(query, (disease_name,))
            result = cursor.fetchone()
            if result:
                symptoms, treatment = result
                return {'disease_name': disease_name, 'plant_name': plant_name, 'symptoms': symptoms, 'treatment': treatment}
            else:
                return jsonify({'error': f'Disease name {disease_name} not found for {plant_name}'})
        else:
            return jsonify({'error': f'No diseases found for {plant_name}'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/')
def index():
    return jsonify({'result': 'Welcome to the server'})

@app.route('/get_disease_info', methods=['POST'])
def get_disease_information():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        base64_image = json_data.get('image', '')

        # Call the function to predict disease from the image
        prediction = predict(base64_image)

        if prediction is not None:
            if prediction == 0:  # Healthy prediction from a binary model
                return jsonify({'result': 'Healthy'})
            else:  # Diseased prediction from a binary model
                print("non healthy")
                plant_name, disease_name = get_plant_and_disease(predict_multiclass(base64_image))
                # Call the function to get disease information
                disease_info = get_disease_info(plant_name, disease_name)
                print(disease_info)
                if disease_info:
                    return jsonify({'result': disease_info})
                else:
                    return jsonify({'error': 'Failed to retrieve disease information'})
        else:
            return jsonify({'error': 'Error during prediction'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict(binary_image):
    # Preprocess and extract features
    features = preprocess_and_extract_features(binary_image)

    if features is not None:
        # Normalize features using the pre-trained scaler
        normalized_features = scaler.transform(np.array([features]))
        # Apply random projection for dimensionality reduction
        random_projection = GaussianRandomProjection(n_components=2000, random_state=42)
        reduced_features = random_projection.fit_transform(normalized_features)

        # Make predictions using the pre-trained binary k-NN model
        prediction = binary_knn.predict(reduced_features)

        return prediction[0]  # Return the predicted class
    else:
        return None

def predict_multiclass(image):
    print('multiclass')
    # Preprocess and extract features
    features = preprocess_and_extract_features(image)
    print(features)
    if features is not None:
        # Normalize features using the pre-trained scaler
        normalized_features = scaler.transform(np.array([features]))
        print(normalized_features)
        random_projection = GaussianRandomProjection(n_components=1000, random_state=42)
        reduced_features = random_projection.fit_transform(normalized_features)
        print(reduced_features)
        # Make predictions using the pre-trained multiclass k-NN model
        prediction = multiclass_knn.predict(reduced_features)
        print (prediction)
        return prediction[0]  # Return the predicted class
    else:
        return None

def preprocess_and_extract_features(image):
    print('preprocess_and_extract_features')
    try:
        # Convert the base64-encoded image to a NumPy array
        image_data = base64.b64decode(image)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image
        gray_image = cv2.resize(gray_image, (128, 128))

        # Calculate LBP features
        lbp_features = calculate_lbp_features(gray_image)

        return lbp_features
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_pixel(img, center, x, y):
    new_value = 0
    if img[x][y] >= center:
        new_value = 1
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def calculate_lbp_features(image):
    rows, cols = image.shape
    lbp_values = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            lbp_values[i, j] = lbp_calculated_pixel(image, i, j)

    return lbp_values.flatten()

if __name__ == '__main__':
    app.run(debug=True)
