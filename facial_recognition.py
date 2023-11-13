import face_recognition
import cv2

# Load the known images and get the face encodings
known_image = face_recognition.load_image_file("known_image.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

# Load an image to check
unknown_image = face_recognition.load_image_file("unknown_image.jpg")
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to BGR color (OpenCV uses BGR)
unknown_image = unknown_image[:, :, ::-1]

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Check if the face is a match for the known face(s)
    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

    label = "Unknown"
    if matches[0]:
        label = "Known"

    # Draw a box around the face
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_image, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the resulting image
cv2.imshow('Image', unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
