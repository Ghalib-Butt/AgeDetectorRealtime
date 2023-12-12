import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Load a pre-trained age estimation model (you need to have one)
age_model = load_model("advanced_age_model.keras")

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector(frame)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extract the face region
        face_roi = frame[y : y + h, x : x + w]

        # Preprocess the face image for age estimation (resize, normalize, etc.)
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # Predict the age
        age_pred = age_model.predict(face_roi)
        age = int(age_pred[0])

        # Draw a rectangle around the face and display the age
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Age: {age}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Display the frame
    cv2.imshow("Age Estimation", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
