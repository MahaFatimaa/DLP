from PIL import Image
import os

# Define the path to the dataset
dataset_path = 'trainfolder-original'

# Loop through each class folder in your dataset
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)

    if os.path.isdir(class_path):
        # Loop through each image in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust file formats as needed
                img_path = os.path.join(class_path, filename)

                # Open the image
                img = Image.open(img_path)

                # Split the filename and the extension
                name, ext = os.path.splitext(filename)

                # Create the rotated versions and save with appropriate names

                # 1. Right rotation (90 degrees clockwise) - Add one zero
                right_rotated_img = img.rotate(90, expand=True)
                right_rotated_name = name + "0" + ext
                right_rotated_img.save(os.path.join(class_path, right_rotated_name))
                print(f"Image {filename} rotated right and saved as {right_rotated_name}.")

                # 2. Left rotation (90 degrees counterclockwise) - Add two zeros
                left_rotated_img = img.rotate(-90, expand=True)
                left_rotated_name = name + "00" + ext
                left_rotated_img.save(os.path.join(class_path, left_rotated_name))
                print(f"Image {filename} rotated left and saved as {left_rotated_name}.")

                # 3. 180-degree flip - Add three zeros
                flipped_img = img.rotate(180, expand=True)
                flipped_name = name + "000" + ext
                flipped_img.save(os.path.join(class_path, flipped_name))
                print(f"Image {filename} flipped and saved as {flipped_name}.")

