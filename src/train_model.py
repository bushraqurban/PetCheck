import tensorflow as tf
from pathlib import Path

def train_model(train_data, val_data, model_save_path):
    # Ensure that the output directory exists
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=10, validation_data=val_data)
    
    # Save the model
    model.save(model_save_path)

def main():
    # Load your preprocessed train and validation data
    train_cats_dir = Path('data/processed/train/cats')
    train_dogs_dir = Path('data/processed/train/dogs')
    test_cats_dir = Path('data/processed/test/cats')
    test_dogs_dir = Path('data/processed/test/dogs')
    
    # Here we assume you've already written code for loading the datasets (with a function like `load_data`)
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        Path('data/processed/train'),
        image_size=(256, 256),
        batch_size=32
    )
    
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        Path('data/processed/test'),
        image_size=(256, 256),
        batch_size=32
    )
    
    # Set the model save path
    model_save_path = Path('models/model.h5')
    
    # Call the train_model function with the necessary arguments
    train_model(train_data, val_data, model_save_path)

if __name__ == "__main__":
    main()