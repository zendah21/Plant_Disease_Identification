
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

# Define the paths to the augmented and non-augmented data directories
augmented_path = 'dataset3/Plant_leave_diseases_dataset_with_augmentation'
non_augmented_path = 'dataset3/Plant_leave_diseases_dataset_without_augmentation'

# data augmentation and normalization
# helps prevent overfitting and allows the model to learn more generalized features.
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Apply shearing transformations
    zoom_range=0.2,  # Randomly zoom into images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Use the nearest pixels for filling gaps when rotating or shifting
    validation_split=0.2
)

# training and validation generators
train_generator = train_datagen.flow_from_directory(
    augmented_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical', # not binary this time  i have 6 class
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    non_augmented_path,  # This should point to the validation data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  # 6 class for this time apple and crops
    subset='validation'
)

#  CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax') # I changed sigmoid to softmax

])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # training cycles
    validation_data=validation_generator,
)

# Save the model
model.save('apple_crops_disease_model.keras')

# Plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
