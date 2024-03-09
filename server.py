import joblib
import cv2
import numpy as np
import base64
import socket
import threading

# Load the trained k-NN model
with open('knn2_model.pkl', 'rb') as model_file:
    best_knn = joblib.load(model_file)

# Load the StandardScaler used for normalization during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

# Variable to store the server's IP address
server_ip = None

def handle_client(client_socket):
    global server_ip

    # Get the client's IP address and port
    client_address = client_socket.getpeername()
    client_ip, client_port = client_address

    print(f"[*] Accepted connection from {client_ip}:{client_port}")

    # Store the client's IP address in a variable
    client_ip_variable = client_ip

    # Update the server_ip variable with the server's IP address
    server_ip = client_socket.getsockname()[0]

    # Receive message from the client (base64-encoded image)
    request_data = client_socket.recv(1024)
    image_data = request_data.decode('utf-8')
    print(f"Received from client ({client_ip_variable}:{client_port}): {image_data}")

    # Make prediction
    prediction = predict(image_data)

    if prediction is not None:
        # Respond back to the client if needed
        client_socket.send(str(prediction).encode('utf-8'))

    # Close the connection
    client_socket.close()
    print(f"Connection from {client_address} closed")

def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Use a common port within the specified range
    server.listen(5)

    print("[*] Socket server listening on 0.0.0.0:5000")

    while True:
        client, addr = server.accept()
        client_handler = threading.Thread(target=handle_client, args=(client,))
        client_handler.start()

def predict(image):
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

        # Normalize features using the pre-trained scaler
        normalized_features = scaler.transform(np.array([lbp_features]))

        # Make predictions using the pre-trained k-NN model
        prediction = best_knn.predict(normalized_features)

        return bool(prediction[0])  # Return the predicted class (0 or 1)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def calculate_lbp_features(image):
    rows, cols = image.shape
    lbp_values = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            lbp_values[i, j] = lbp_calculated_pixel(image, i, j)

    return lbp_values.flatten()

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

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

if __name__ == '__main__':
    # Start the socket server in a separate thread
    socket_thread = threading.Thread(target=start_socket_server)
    socket_thread.start()

    # Keep the main thread free for other tasks, you can add your additional logic here if needed
    while True:
        pass
