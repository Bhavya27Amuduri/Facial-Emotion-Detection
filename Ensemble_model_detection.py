import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# Build the ensemble model
def build_ensemble_model(input_shape=(48, 48, 3)):
    # Define a single input layer
    input_layer = Input(shape=input_shape)

    # ResNet50 Branch
    resnet = ResNet50(weights="imagenet", include_top=False, input_tensor=input_layer)
    resnet_output = GlobalAveragePooling2D()(resnet.output)

    # MobileNetV2 Branch
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_layer)
    mobilenet_output = GlobalAveragePooling2D()(mobilenet.output)

    # Concatenate outputs from both branches
    combined = Concatenate()([resnet_output, mobilenet_output])

    # Add fully connected layers
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.5)(x)
    output = Dense(7, activation='softmax')(x)  # 7 emotion classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Unfreeze all layers to allow full training
    for layer in model.layers:
        layer.trainable = True

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
model = build_ensemble_model()

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'Fer2013/train',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'Fer2013/test',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10  
)

#Training data plot
def plot_accuracy_loss(history):
    # Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_accuracy_loss(history)


# Save the model
model.save("emotion_detection_model_ensemble.h5")
