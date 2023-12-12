import cv2
import os

# Create a directory to store the dataset
dataset_dir = "age_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Counter to keep track of collected images
image_count = 0

while True:
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Collecting Images", frame)

    # Save the frame as an image when the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("s"):
        # Prompt the user to enter the age
        age = input("Enter the age of the person: ")

        # Ensure the age is a valid integer
        try:
            age = int(age)
        except ValueError:
            print("Invalid age input. Please enter a valid integer.")
            continue

        # Create a subfolder for the entered age if it doesn't exist
        age_folder = os.path.join(dataset_dir, f"age_{age}")
        os.makedirs(age_folder, exist_ok=True)

        # Save the frame as an image in the age-specific subfolder
        image_count += 1
        image_filename = os.path.join(age_folder, f"image_{image_count}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Saved {image_filename}")

    # Exit the loop when the 'Esc' key (key code 27) is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
