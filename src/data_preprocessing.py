import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image

def preprocess_and_save_images(input_dir: Path, output_dir: Path, img_size=(256, 256)):
    """
    Preprocess images: Resize, normalize, and save to the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Loop over each class (cats, dogs)
    for class_name in ['cats', 'dogs']:
        class_input_dir = input_dir / class_name
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)  # Create class-specific output dir
        
        # Process each image in the class directory
        for img_path in class_input_dir.iterdir():
            if img_path.suffix in ['.jpg', '.jpeg', '.png']:  # Only process image files
                img = image.load_img(img_path, target_size=img_size)  # Load image and resize
                img_array = image.img_to_array(img)  # Convert image to array
                img_array /= 255.0  # Normalize image to [0, 1] range

                # Save the processed image
                output_img_path = class_output_dir / img_path.name
                image.save_img(str(output_img_path), img_array)  # Save image

def main():
    # Define the raw data directory
    raw_data_path = Path('data/raw')
    
    # Define the directories for input and output (train/test, cats/dogs)
    input_train_dir = raw_data_path / 'train'
    input_test_dir = raw_data_path / 'test'
    output_train_dir = Path('data/processed/train')
    output_test_dir = Path('data/processed/test')

    # Preprocess and save images from train and test directories
    preprocess_and_save_images(input_train_dir, output_train_dir)
    preprocess_and_save_images(input_test_dir, output_test_dir)

if __name__ == "__main__":
    main()
