import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Initialize variables
face_count = 0
previous_faces = set()

# Variable to control the loop
running = True

def on_key(event):
    global running
    if event.key == 'q':
        running = False

# Connect the key press event to the handler
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

try:
    while running:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Create a set for current face positions
        current_faces = set()

        # Draw rectangles around detected faces and add them to the set
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            current_faces.add((x, y, w, h))

        # Compare current faces with previous faces to update count
        if current_faces != previous_faces:
            face_count = len(current_faces)

        # Update previous faces for the next iteration
        previous_faces = current_faces

        # Show the count on the frame
        cv2.putText(frame, f'Face Count: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with the detected faces using matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Face Detection')
        plt.axis('off')
        plt.pause(0.001)
        plt.clf()

except KeyboardInterrupt:
    print("Interrupted by user")

# Release the capture and close windows
cap.release()
plt.close()
cv2.destroyAllWindows()