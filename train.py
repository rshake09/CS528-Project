import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ---------------- CONFIG ----------------
input_dir = './processed_data'
batch_size = 32
image_size = (224, 224)
epochs = 30

# ---------------- LOAD DATA ----------------
train_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print("Classes:", class_names)

# ---------------- DATA PIPELINE ----------------
AUTOTUNE = tf.data.AUTOTUNE

# Data augmentation (VERY important)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

train_data = train_data.map(lambda x, y: (preprocess_input(x), y))
val_data = val_data.map(lambda x, y: (preprocess_input(x), y))

train_data = train_data.cache().shuffle(1000).prefetch(AUTOTUNE)
val_data = val_data.cache().prefetch(AUTOTUNE)

# ---------------- MODEL ----------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=image_size + (3,)
)

# Freeze base model
base_model.trainable = False

# Build model
inputs = tf.keras.Input(shape=image_size + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    factor=0.5,
    min_lr=1e-6
)

# ---------------- TRAIN ----------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stop, lr_scheduler]
)

# ---------------- OPTIONAL FINE-TUNING ----------------
print("\n[INFO] Fine-tuning last layers...")

base_model.trainable = True

# freeze most layers, only train top ones
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

# ---------------- SAVE ----------------
os.makedirs('./models', exist_ok=True)
model.save('./models/asl_model.keras')

print("[INFO] Model saved.")