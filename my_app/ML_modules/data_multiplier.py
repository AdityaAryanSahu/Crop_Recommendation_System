import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, save_img


# Configure your input and output folders
input_base_dir = "D:\Soil_type_dataset"
output_base_dir = "D:\Soil_type_dataset"

classes_to_augment = ['Black', 'Red']
target_augmented_count = 1000  # desired number per class (including originals)

# Define your augmentation strategy
augmentor = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

for soil_class in classes_to_augment:
    input_dir = os.path.join(input_base_dir, soil_class)
    output_dir = os.path.join(output_base_dir, soil_class)
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images in '{soil_class}'")

    generated = 0
    while generated < (target_augmented_count - len(images)):
        for image_file in images:
            if generated >= (target_augmented_count - len(images)):
                break

            img_path = os.path.join(input_dir, image_file)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate one augmented image
            prefix = os.path.splitext(image_file)[0]
            for batch in augmentor.flow(x, batch_size=1):
                aug_filename = f"{prefix}_aug_{generated}.jpg"
                save_path = os.path.join(output_dir, aug_filename)
                save_img(save_path, array_to_img(batch[0]))
                generated += 1
                break  # generate only 1 per original image at a time

    print(f"✓ Augmented '{soil_class}' to ~{target_augmented_count} images.")
