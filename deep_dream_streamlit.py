import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using GPU for computations.")
else:
    print("Using CPU for computations.")

# Load your pre-trained model
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# Filter layer names to include only those that contain "mixed"
layer_names = [
    layer.name for layer in base_model.layers
    if 'mixed' in layer.name and not isinstance(layer, (tf.keras.layers.Concatenate, tf.keras.layers.Add))
]

def calc_loss(image, model, layer_names):
    img_batch = tf.expand_dims(image, axis=0)
    losses = []
    for layer_name in layer_names:
        layer_output = model.get_layer(layer_name)(img_batch)
        losses.append(tf.reduce_mean(layer_output))
    return tf.reduce_sum(losses)

@tf.function
def deepdream(model, image, step_size, layer_names):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = calc_loss(image, model, layer_names)

    gradients = tape.gradient(loss, image)
    gradients /= tf.math.reduce_std(gradients) + 1e-8  # Avoid division by zero

    image = image + gradients * step_size
    image = tf.clip_by_value(image, -1, 1)

    return loss, image

def deprocess(image):
    return tf.cast(255 * (image + 1.0) / 2.0, tf.uint8)

def run_deep_dream(image, steps=100, step_size=0.01, layer_names=None):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    output_images = []  # List to store images at each step
    for i in range(steps):
        _, image = deepdream(base_model, image, step_size, layer_names)
        if i % 100 == 0:  # Store every 100 steps
            output_images.append(deprocess(image))
    return output_images

# Streamlit App
st.title("Deep Dream Image Generator")
st.write("Upload an image to create a dream-like effect!")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    original_size = image.size  # Store original size
    image = image.resize((225, 375))  # Resize for the model
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)  # Convert to TensorFlow tensor

    # Layer selection
    layer_names_selected = st.multiselect("Choose layers to maximize (only 'mixed' layers):", layer_names)

    # Run the Deep Dream process
    if st.button("Generate Dream Image"):
        if layer_names_selected:
            dream_images = run_deep_dream(image_tensor, steps=2000, step_size=0.01, layer_names=layer_names_selected)

            # Display input image in original size
            st.image(image, caption='Input Image', use_column_width=False, width=original_size[0])

            # Present dream images as a list with original size
            st.write("Generated Dream Images:")
            for step, dream_image in enumerate(dream_images):
                st.image(dream_image.numpy(), caption=f'Step {step * 100}', use_column_width=False, width=original_size[0])
        else:
            st.warning("Please select at least one layer to maximize.")
