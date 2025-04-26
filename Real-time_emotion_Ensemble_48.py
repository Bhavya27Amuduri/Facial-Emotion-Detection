import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_detection_model_ensemble.h5")

# Define the emotion labels
emotion_labels = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Create a VideoCapture object to access the camera
cap = cv2.VideoCapture(0)  # Change the parameter to the appropriate camera index if multiple cameras are connected

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, display "No face found" on the frame
    if len(faces) == 0:
        cv2.putText(frame, "No face found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Preprocess the face region
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            # Make predictions
            result = model.predict(reshaped, verbose=0)
            emotion_index = np.argmax(result)
            emotion_label = emotion_labels[emotion_index]

            # Get the confidence percentage
            confidence = result[0][emotion_index] * 100

            # Create the label with the emotion and confidence percentage
            label = f"{emotion_label} ({confidence:.2f}%)"

            # Display the emotion label and confidence on the frame
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the display window
cap.release()
cv2.destroyAllWindows()
