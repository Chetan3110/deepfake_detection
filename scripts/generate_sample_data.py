import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

# Paths
sample_data_dir = 'data/sample_data/'
synthetic_data_dir = 'data/synthetic_data/'

# Create synthetic_data directory if it doesn't exist
os.makedirs(synthetic_data_dir + 'df/', exist_ok=True)
os.makedirs(synthetic_data_dir + 'real/', exist_ok=True)

# Augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_images(category):
    input_dir = sample_data_dir + category + '/'
    output_dir = synthetic_data_dir + category + '/'
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            # Load and prepare image
            img_path = os.path.join(input_dir, filename)
            img = load_img(img_path)  # Load image as PIL object
            img_array = img_to_array(img)  # Convert to NumPy array
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for datagen
            
            # Generate and save augmented images
            prefix = filename.split('.')[0]
            counter = 0
            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir,
                                      save_prefix=prefix, save_format='jpg'):
                counter += 1
                if counter >= 5:  # Generate 5 images per input
                    break

# Generate synthetic data for both categories
print("Generating synthetic data for 'df' category...")
augment_images('df')

print("Generating synthetic data for 'real' category...")
augment_images('real')

print("Synthetic data generation complete!")
print(f"Synthetic data saved to: {synthetic_data_dir}")
