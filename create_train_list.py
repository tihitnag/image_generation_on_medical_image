import os

# Path to your main images directory
images_dir = 'data/your_dataset/imagefile/imagefile'

# Define the class folders
class_folders = ['abnormal', 'normal']

# Create train.txt
with open('data/your_dataset/train.txt', 'w') as f:
    total_images = 0
    
    for class_folder in class_folders:
        class_path = os.path.join(images_dir, class_folder)
        if os.path.exists(class_path):
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Write paths for each image
            for image in image_files:
                # Write the full path to the image
                f.write(f'{os.path.join(class_path, image)}\n')
                total_images += 1

print(f"Created train.txt with {total_images} images")
print(f"Images found in:")
for class_folder in class_folders:
    class_path = os.path.join(images_dir, class_folder)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"- {class_folder}: {count} images")
    else:
        print(f"- {class_folder}: directory not found at {class_path}") 