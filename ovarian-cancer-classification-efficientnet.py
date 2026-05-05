
# STEP 1: Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')

drive_dataset_path = '/content/drive/MyDrive/Colab Notebooks/dataset'



# STEP 2: Copy Dataset Locally

import os, shutil

local_dataset_path = '/content/dataset'

if not os.path.exists(local_dataset_path):
    shutil.copytree(drive_dataset_path, local_dataset_path)
    print(" Dataset copied to Colab local storage.")
else:
    print("Dataset already exists locally.")

train_dir = os.path.join(local_dataset_path, 'train')
test_dir = os.path.join(local_dataset_path, 'test')

print("Train classes:", os.listdir(train_dir))
print("Test classes:", os.listdir(test_dir))



# STEP 3: Data Pipeline

import tensorflow as tf

IMG_SIZE = (224, 224)  # Full size for best accuracy
BATCH_SIZE = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
train_data = train_data.prefetch(tf.data.AUTOTUNE)
test_data = test_data.prefetch(tf.data.AUTOTUNE)

print("Class Names:", class_names)



# STEP 4: Data Augmentation

from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])



# STEP 5: Full Fine-Tuning EfficientNetB0

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
base_model.trainable = True  # Fine-tuning entire model

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=True)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)  # Keep dropout for regularization
outputs = layers.Dense(len(class_names), activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# STEP 6: Train Model

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=25  # Enough for convergence
)



# STEP 7: Plot Accuracy & Loss

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training & Validation Loss')

plt.show()



# STEP 8: Single Image Inference

import numpy as np
from tensorflow.keras.preprocessing import image

img_path = "/content/dataset/test/HGSC/10013.png"  # Change if needed

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print(f"🔍 Predicted Class: {class_names[predicted_class]}")
print(f"📊 Confidence: {np.max(predictions)*100:.2f}%")



# STEP 9: Evaluate with F1, Precision, Recall & Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Collect predictions on test set
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification report
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



# STEP X: Save Entire Model (.keras format)

model.save('/content/drive/MyDrive/Colab Notebooks/saved_model/efficientnet_b0_model.keras')
print("✅ Model saved in .keras format!")
