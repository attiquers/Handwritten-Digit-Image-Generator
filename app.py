# app.py - Updated to match the screenshot UI and address deprecation warnings

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import io

# --- Model Definition (Must match your trained model) ---
# Define the Generator architecture EXACTLY as it was defined during training.
# This must match the ConditionalGenerator class in your training script.
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, num_classes, img_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh() # Output in range [-1, 1]
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels.long()) # Ensure labels are long type for embedding
        input_tensor = torch.cat([noise, c], 1)
        return self.main(input_tensor).view(-1, 1, 28, 28)

# --- Model Loading (Cached for efficiency) ---
@st.cache_resource
def load_generator_model(model_path="conditional_generator_mnist.pth", z_dim=100, num_classes=10, img_dim=28*28):
    generator = ConditionalGenerator(z_dim, num_classes, img_dim)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval() # Set to evaluation mode (disables dropout, batchnorm etc.)
        st.success(f"Generator model loaded from {model_path}")
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory as app.py and your GitHub repo.")
        st.stop() # Stop the app if model can't be loaded
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()
    return generator

# Instantiate the generator. This will call load_generator_model and cache the result.
generator = load_generator_model()

# --- Streamlit App UI ---
st.set_page_config(layout="centered") # Default layout is centered, but explicit for clarity

# Title
st.title("Handwritten Digit Image Generator")

# Introduction text
st.write("Generate synthetic MNIST-like images using your trained model.")

# Space for separation (optional, but can help visual spacing)
st.write("")

# User input: select a digit using a selectbox (dropdown)
st.write("Choose a digit to generate (0-9):")
selected_digit = st.selectbox(
    " ", # Empty label because the prompt is given by st.write above
    options=list(range(10)),
    index=2 # Default to '2' as seen in the screenshot
)

# Generate Button
if st.button("Generate Images"):
    # Information message when generation starts
    st.info(f"Generating 5 images for digit: **{selected_digit}**...")

    # --- Image Generation Logic ---
    num_images_to_generate = 5
    z_dim = 100 # Must match z_dim used in training
    num_classes = 10 # Must match num_classes used in training

    with torch.no_grad(): # Disable gradient calculations for inference, saves memory and speeds up
        noise = torch.randn(num_images_to_generate, z_dim) # Generate random noise for diversity
        labels = torch.full((num_images_to_generate,), selected_digit, dtype=torch.long) # Create labels for the specific digit
        generated_images_tensor = generator(noise, labels)

    # Denormalize images from [-1, 1] to [0, 255] and convert to PIL Image for display
    display_images = []
    for i in range(num_images_to_generate):
        img_tensor = generated_images_tensor[i].squeeze() # Remove channel dimension (1, 28, 28) -> (28, 28)
        img_tensor = (img_tensor + 1) / 2 # Denormalize from [-1, 1] to [0, 1]
        img_np = (img_tensor.numpy() * 255).astype(np.uint8) # Convert to 0-255 pixel values
        display_images.append(Image.fromarray(img_np, 'L')) # 'L' specifies grayscale mode for PIL

    # --- Display Generated Images ---
    st.markdown(f"## Generated images of digit {selected_digit}") # Using markdown for larger header

    # Display images in a grid/row using st.columns
    # Adjust the column widths if needed; a list of 5 equal parts works well for 5 images.
    cols = st.columns(num_images_to_generate)
    for i, img in enumerate(display_images):
        with cols[i]:
            # Updated: Replaced 'use_column_width' with 'use_container_width'
            st.image(img, caption="", use_container_width=False, width=150)

# Optional: Add a footer or attribution
st.markdown("---")
st.caption("Developed using PyTorch and Streamlit.")