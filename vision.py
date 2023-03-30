import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CSV file
csv_path = 'gfwtraining.csv'
df = pd.read_csv(csv_path, header=None)
df.columns = ['filename', 'class']

# Update the 'filename' column to include the .png extension and the directory path
df['filename'] = df['filename'].apply(lambda x: f'gfwtrain/{x}.png')

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Use 20% of the data for validation
)

# Set batch size and image dimensions
batch_size = 32
img_height, img_width = 299, 299

# Create train and validation generators
train_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

import tensorflow as tf

# Define your model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(30, activation='softmax')  # 30 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10
)

model.save_weights('model_weights.h5')  
