import cv2 # type: ignore
import mediapipe as mp # type: ignore
import pygame # type: ignore
import sys
import numpy as np

# Initialize Mediapipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Initialize Pygame
pygame.init()

# Load the image and detect landmarks
def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        sys.exit()

    # Convert BGR image to RGB for Mediapipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        annotated_image = image.copy()

        # Extract landmarks for visualization
        height, width, _ = image.shape
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)

        return annotated_image
    else:
        print("No face detected in the image.")
        sys.exit()

# Pygame visualization and animation
def display_in_pygame(annotated_image):
    # Convert image to RGB for Pygame
    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    pygame_image = pygame.surfarray.make_surface(np.rot90(rgb_image))

    # Set up Pygame screen
    screen = pygame.display.set_mode((annotated_image.shape[1], annotated_image.shape[0]))
    pygame.display.set_caption("Face to Animation")

    # Run the Pygame event loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(pygame_image, (0, 0))
        pygame.display.flip()

    pygame.quit()

# Main function
if __name__ == "__main__":
    image_path = input("Enter the path to the face image: ")

    # Process the image to extract face landmarks
    annotated_image = process_image(image_path)

    # Display the annotated image in Pygame
    display_in_pygame(annotated_image)
