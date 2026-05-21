"""
Streamlit UI for Secure Medical Image Processing
Run with: streamlit run app.py
"""
import io
import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Add src to path
sys.path.append('src')

from evaluation_metrics import calculate_psnr, calculate_ssim
from image_denoising import denoise_image
from image_optimization import compress_image, enhance_image
from image_security import decrypt_image, encrypt_image, generate_key
from image_segmentation import threshold_segmentation
from user_authentication import hash_password, verify_password
from watermarking import add_watermark

# Page configuration
st.set_page_config(
    page_title="Secure Medical Image Processing",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Simple & Clean
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None

# Create necessary directories
os.makedirs('temp', exist_ok=True)
os.makedirs('results', exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory"""
    if uploaded_file is not None:
        file_path = os.path.join('temp', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

def get_dataset_images(data_path='data/breast-histopathology-images', limit=50):
    """Get list of images from dataset"""
    images = []
    
    if not os.path.exists(data_path):
        return images
    
    patient_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()]
    
    for patient_id in patient_dirs[:10]:  # First 10 patients
        patient_path = os.path.join(data_path, patient_id)
        
        for class_label in ['0', '1']:
            class_path = os.path.join(patient_path, class_label)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path)[:5]:  # 5 images per class
                    if img_file.endswith('.png'):
                        full_path = os.path.join(class_path, img_file)
                        images.append({
                            'path': full_path,
                            'name': img_file,
                            'patient': patient_id,
                            'label': 'Cancer' if class_label == '1' else 'No Cancer'
                        })
                        if len(images) >= limit:
                            return images
    
    return images

def display_image_comparison(original_path, processed_path, original_label="Original", processed_label="Processed"):
    """Display two images side by side"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_path, caption=original_label, use_column_width=True)
    with col2:
        st.image(processed_path, caption=processed_label, use_column_width=True)

# Main App
def main():
    # Sidebar
    st.sidebar.markdown("# 🏥 Navigation")
    
    if not st.session_state.authenticated:
        page = st.sidebar.radio(
            "Select Module",
            ["Login", "User Management"]
        )
    else:
        page = st.sidebar.radio(
            "Select Module",
            ["Home", "Image Processing", "Disease Detection", "Security", "Evaluation Metrics", "Explainable AI", "Interactive Visualizations", "Automated Training", "User Management"]
        )
    # Interactive Visualizations Page
    if page == "Interactive Visualizations":
        st.markdown('<h1 class="main-header">📊 Interactive Visualizations</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        Visualize model performance with confusion matrix and ROC curve.
        """)
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, confusion_matrix, roc_curve
        y_true = st.text_area("True Labels (comma-separated)", "0,1,1,0,1")
        y_pred = st.text_area("Predicted Labels (comma-separated)", "0,1,0,0,1")
        y_true = [int(x.strip()) for x in y_true.split(",") if x.strip().isdigit()]
        y_pred = [int(x.strip()) for x in y_pred.split(",") if x.strip().isdigit()]
        if st.button("Show Confusion Matrix"):
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], va='center', ha='center')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig)
        if st.button("Show ROC Curve"):
            if len(set(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc='lower right')
                st.pyplot(fig)
            else:
                st.warning("ROC curve requires binary classification labels.")

    # User Management Page
    if page == "User Management":
        st.markdown('<h1 class="main-header">👤 User Management</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        Register new users and assign roles.
        """)
        if 'users' not in st.session_state:
            st.session_state.users = {}
        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["user", "admin", "doctor", "researcher"])
        if st.button("Register User"):
            if username and password:
                from user_authentication import hash_password
                st.session_state.users[username] = {
                    'password': hash_password(password),
                    'role': role
                }
                st.success(f"User '{username}' registered as {role}!")
            else:
                st.warning("Please enter username and password.")

    # Automated Training Page
    if page == "Automated Training":
        st.markdown('<h1 class="main-header">🤖 Automated Model Training</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        Launch a new model training run with custom parameters. This will use the training pipeline and save the best model.
        """)
        import importlib
        train_pipeline = importlib.import_module("train_pipeline")
        from disease_detection import build_resnet_model
        data_dir = st.text_input("Data directory", "data/breast-histopathology-images")
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32)
        save_path = st.text_input("Model save path", "models/resnet_model.h5")
        max_images = st.number_input("Max images (optional, 0=all)", min_value=0, value=0)
        if st.button("Start Training"):
            st.info("Training started. Check terminal for progress logs.")
            model = build_resnet_model(num_classes=2, freeze_base=True)
            callbacks = None
            if max_images == 0:
                max_images = None
            history = train_pipeline.train_model(
                data_dir,
                model,
                epochs=epochs,
                batch_size=batch_size,
                save_path=save_path,
                max_images=max_images,
                callbacks=callbacks
            )
            st.success(f"Training complete. Best model saved to: {save_path}")
    
    # Login Page
    if page == "Login":
        st.markdown('<h1 class="main-header">🏥 Secure Medical Image Processing</h1>', unsafe_allow_html=True)
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 User Authentication")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username and password:
                    # Check against registered users in session state
                    users = st.session_state.get('users', {})
                    user = users.get(username)
                    if user and verify_password(user['password'], password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
                else:
                    st.warning("Please enter username and password")
    
    # Explainable AI Page
    elif page == "Explainable AI":
        st.markdown('<h1 class="main-header">🧠 Explainable AI (Grad-CAM)</h1>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        Visualize model predictions using Grad-CAM for interpretability.
        """)
        import os

        from explainable_ai import grad_cam

        # Select image and model
        image_path = st.text_input("Path to image", "data/breast-histopathology-images/10253/sample.png")
        model_path = st.text_input("Path to model", "models/resnet_model.h5")
        layer_name = st.text_input("Layer name for Grad-CAM", "conv5_block3_out")
        class_index = st.number_input("Class index (e.g. 0=Normal, 1=Cancer)", min_value=0, max_value=10, value=1)
        if st.button("Generate Grad-CAM Overlay"):
            if os.path.exists(image_path) and os.path.exists(model_path):
                overlay = grad_cam(image_path, model_path, layer_name, class_index)
                st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
                st.success("Grad-CAM visualization generated!")
            else:
                st.error("Image or model file not found.")

    # Home Page
    elif page == "Home":
        st.markdown('<h1 class="main-header">🏥 Secure Medical Image Processing</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("""
        ### Welcome to the Secure Medical Image Processing System
        
        This platform provides comprehensive tools for medical image analysis and security.
        
        #### 🎯 Features:
        - **Image Processing**: Denoise, enhance, and segment medical images
        - **Disease Detection**: AI-powered cancer detection from histopathology images
        - **Security**: Encrypt/decrypt images with AES encryption
        - **Evaluation**: Calculate PSNR, SSIM, and other quality metrics
        - **Watermarking**: Add authentication marks to images
        
        #### 📊 Tech Stack:
        - OpenCV for image processing
        - TensorFlow for deep learning
        - Cryptography for security
        - Streamlit for UI
        
        Use the sidebar to navigate between different modules.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🔒 **Secure**\n\nAES encryption for image protection")
        with col2:
            st.info("🤖 **Intelligent**\n\nML-based disease detection")
        with col3:
            st.info("⚡ **Fast**\n\nOptimized processing pipeline")
        with col1:
            st.markdown("""
            - 🖼️ **Image Processing**: Denoise, enhance, and segment
            - 🔬 **Disease Detection**: AI-powered cancer detection
            - 🔐 **Security**: Encrypt/decrypt with AES-256
            """)
        with col2:
            st.markdown("""
            - 📊 **Evaluation Metrics**: PSNR, SSIM calculations
            - 💧 **Watermarking**: Add authentication marks
            - � **Dataset Integration**: Direct dataset access
            """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Technology Stack")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        with tech_col1:
            st.markdown("**� Python**")
            st.caption("Core language")
        with tech_col2:
            st.markdown("**👁️ OpenCV**")
            st.caption("Image processing")
        with tech_col3:
            st.markdown("**� TensorFlow**")
            st.caption("Deep learning")
        with tech_col4:
            st.markdown("**🔐 Cryptography**")
            st.caption("Security")
        
        st.markdown("<br><p style='text-align: center; color: #667eea; font-weight: 600;'>Use the sidebar to navigate between different modules 👈</p>", unsafe_allow_html=True)
    
    # Image Processing Page
    elif page == "Image Processing":
        st.markdown('<h1 class="main-header">🖼️ Image Processing</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Option to choose between dataset or upload
        source = st.radio("Image Source:", ["📁 Use Dataset Images", "📤 Upload Image"], horizontal=True)
        
        file_path = None
        
        if source == "📁 Use Dataset Images":
            # Get images from dataset
            dataset_images = get_dataset_images()
            
            if dataset_images:
                # Create a selection box with image info
                image_options = [f"{img['name']} - Patient {img['patient']} - {img['label']}" for img in dataset_images]
                selected_idx = st.selectbox("Select an image from dataset:", range(len(image_options)), 
                                           format_func=lambda x: image_options[x])
                
                file_path = dataset_images[selected_idx]['path']
                st.image(file_path, caption=f"Selected: {dataset_images[selected_idx]['name']} ({dataset_images[selected_idx]['label']})", width=400)
            else:
                st.warning("No images found in dataset. Please add images to data/breast-histopathology-images/")
        
        else:  # Upload Image
            uploaded_file = st.file_uploader("Upload Medical Image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                st.session_state.temp_image_path = file_path
                st.image(file_path, caption="Uploaded Image", width=400)
            
            st.markdown("### Select Processing Operations:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧹 Denoise"):
                    output_path = 'results/denoised.png'
                    denoise_image(file_path, output_path)
                    st.success("✅ Image denoised!")
                    display_image_comparison(file_path, output_path, "Original", "Denoised")
            
            with col2:
                if st.button("✨ Enhance"):
                    output_path = 'results/enhanced.png'
                    enhance_image(file_path, output_path)
                    st.success("✅ Image enhanced!")
                    display_image_comparison(file_path, output_path, "Original", "Enhanced")
            
            with col3:
                if st.button("🔪 Segment"):
                    output_path = 'results/segmented.png'
                    threshold_segmentation(file_path, output_path)
                    st.success("✅ Image segmented!")
                    display_image_comparison(file_path, output_path, "Original", "Segmented")
            
            col4, col5 = st.columns(2)
            
            with col4:
                watermark_text = st.text_input("Watermark Text", "CONFIDENTIAL")
                if st.button("💧 Add Watermark"):
                    output_path = 'results/watermarked.png'
                    add_watermark(file_path, watermark_text, output_path)
                    st.success("✅ Watermark added!")
                    display_image_comparison(file_path, output_path, "Original", "Watermarked")
            
            with col5:
                quality = st.slider("Compression Quality", 1, 100, 85)
                if st.button("📦 Compress"):
                    output_path = 'results/compressed.jpg'
                    compress_image(file_path, output_path, quality)
                    st.success("✅ Image compressed!")
                    display_image_comparison(file_path, output_path, "Original", "Compressed")
    
    # Disease Detection Page
    elif page == "Disease Detection":
        st.markdown('<h1 class="main-header">🔬 Disease Detection</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Option to choose between dataset or upload
        source = st.radio("Image Source:", ["📁 Use Dataset Images", "📤 Upload Image"], horizontal=True, key="disease_source")
        
        file_path = None
        actual_label = None
        
        if source == "📁 Use Dataset Images":
            # Get images from dataset
            dataset_images = get_dataset_images()
            
            if dataset_images:
                # Create a selection box with image info
                image_options = [f"{img['name']} - Patient {img['patient']} - {img['label']}" for img in dataset_images]
                selected_idx = st.selectbox("Select an image from dataset:", range(len(image_options)), 
                                           format_func=lambda x: image_options[x], key="disease_select")
                
                file_path = dataset_images[selected_idx]['path']
                actual_label = dataset_images[selected_idx]['label']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(file_path, caption=f"Selected: {dataset_images[selected_idx]['name']}", use_column_width=True)
                    st.info(f"**Actual Label:** {actual_label}")
            else:
                st.warning("No images found in dataset. Please add images to data/breast-histopathology-images/")
        
        else:  # Upload Image
            uploaded_file = st.file_uploader("Upload Histopathology Image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(file_path, caption="Uploaded Image", use_column_width=True)
        
        if file_path:
            col1, col2 = st.columns(2)
            with col2:
                if st.button("🔍 Detect Cancer"):
                    model_path = 'models/cancer_detection_model.h5'
                    
                    if os.path.exists(model_path):
                        from tensorflow.keras.models import load_model

                        # Load model
                        model = load_model(model_path)
                        
                        # Preprocess
                        img = cv2.imread(file_path)
                        img = cv2.resize(img, (50, 50))
                        img = img / 255.0
                        img = np.expand_dims(img, axis=0)
                        
                        # Predict
                        prediction = model.predict(img)[0][0]
                        
                        if prediction > 0.5:
                            result = "Cancer Detected"
                            confidence = prediction * 100
                            st.error(f"⚠️ **{result}**")
                        else:
                            result = "No Cancer"
                            confidence = (1 - prediction) * 100
                            st.success(f"✅ **{result}**")
                        
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Show actual label if from dataset
                        if actual_label:
                            if (actual_label == "Cancer" and result == "Cancer Detected") or \
                               (actual_label == "No Cancer" and result == "No Cancer"):
                                st.success(f"✅ Correct! Actual: {actual_label}")
                            else:
                                st.error(f"❌ Incorrect! Actual: {actual_label}")
                        
                        # Progress bar
                        st.progress(int(confidence))
                    else:
                        st.warning("⚠️ Model not found. Please train the model first by running `python src/train_model.py`")
    
    # Security Page
    elif page == "Security":
        st.markdown('<h1 class="main-header">🔒 Image Security</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Encrypt", "🔓 Decrypt"])
        
        with tab1:
            st.markdown("### Encrypt Medical Image")
            uploaded_file = st.file_uploader("Upload Image to Encrypt", type=['png', 'jpg', 'jpeg'], key="encrypt")
            
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
                st.image(file_path, caption="Image to Encrypt", width=300)
                
                if st.button("🔐 Encrypt Image"):
                    key_path = 'results/secret.key'
                    if not os.path.exists(key_path):
                        generate_key(key_path)
                        st.info("🔑 Encryption key generated")
                    
                    encrypted_path = 'results/encrypted.img'
                    encrypt_image(file_path, key_path, encrypted_path)
                    
                    st.success("✅ Image encrypted successfully!")
                    st.download_button(
                        label="📥 Download Encrypted File",
                        data=open(encrypted_path, 'rb').read(),
                        file_name='encrypted.img'
                    )
                    st.download_button(
                        label="🔑 Download Encryption Key",
                        data=open(key_path, 'rb').read(),
                        file_name='secret.key'
                    )
        
        with tab2:
            st.markdown("### Decrypt Medical Image")
            
            col1, col2 = st.columns(2)
            with col1:
                encrypted_file = st.file_uploader("Upload Encrypted File (.img)", type=['img'], key="decrypt_img")
            with col2:
                key_file = st.file_uploader("Upload Encryption Key (.key)", type=['key'], key="decrypt_key")
            
            if encrypted_file and key_file:
                enc_path = save_uploaded_file(encrypted_file)
                key_path = save_uploaded_file(key_file)
                
                # Get image dimensions
                width = st.number_input("Image Width", value=50, min_value=1)
                height = st.number_input("Image Height", value=50, min_value=1)
                channels = st.selectbox("Channels", [3, 1], index=0)
                
                if st.button("🔓 Decrypt Image"):
                    try:
                        decrypted_path = 'results/decrypted.png'
                        decrypt_image(enc_path, key_path, shape=(height, width, channels), output_path=decrypted_path)
                        st.success("✅ Image decrypted successfully!")
                        st.image(decrypted_path, caption="Decrypted Image", width=300)
                    except Exception as e:
                        st.error(f"❌ Decryption failed: {e}")
    
    # Evaluation Metrics Page
    elif page == "Evaluation Metrics":
        st.markdown('<h1 class="main-header">📊 Evaluation Metrics</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Compare Two Images")
        
        # Option to use dataset or upload
        source = st.radio("Image Source:", ["📁 Use Dataset Images", "📤 Upload Images"], horizontal=True, key="metrics_source")
        
        path1 = None
        path2 = None
        
        if source == "📁 Use Dataset Images":
            dataset_images = get_dataset_images()
            
            if dataset_images and len(dataset_images) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Image:**")
                    image_options = [f"{img['name']} - {img['label']}" for img in dataset_images]
                    selected_idx1 = st.selectbox("Select first image:", range(len(image_options)), 
                                               format_func=lambda x: image_options[x], key="metrics_img1")
                    path1 = dataset_images[selected_idx1]['path']
                    st.image(path1, caption=f"Image 1", use_column_width=True)
                
                with col2:
                    st.markdown("**Processed/Comparison Image:**")
                    selected_idx2 = st.selectbox("Select second image:", range(len(image_options)), 
                                               format_func=lambda x: image_options[x], key="metrics_img2")
                    path2 = dataset_images[selected_idx2]['path']
                    st.image(path2, caption=f"Image 2", use_column_width=True)
            else:
                st.warning("Need at least 2 images in dataset for comparison.")
        
        else:  # Upload Images
            col1, col2 = st.columns(2)
            
            with col1:
                img1 = st.file_uploader("Upload Original Image", type=['png', 'jpg', 'jpeg'], key="img1")
                if img1:
                    path1 = save_uploaded_file(img1)
                    st.image(path1, caption="Image 1", use_column_width=True)
            
            with col2:
                img2 = st.file_uploader("Upload Processed Image", type=['png', 'jpg', 'jpeg'], key="img2")
                if img2:
                    path2 = save_uploaded_file(img2)
                    st.image(path2, caption="Image 2", use_column_width=True)
        
        if path1 and path2:
            if st.button("📊 Calculate Metrics"):
                try:
                    psnr = calculate_psnr(path1, path2)
                    ssim = calculate_ssim(path1, path2)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PSNR (Peak Signal-to-Noise Ratio)", f"{psnr:.2f} dB")
                        st.caption("Higher is better (>30 dB is good)")
                    
                    with col2:
                        st.metric("SSIM (Structural Similarity Index)", f"{ssim:.4f}")
                        st.caption("Closer to 1 is better")
                    
                    # Interpretation
                    if psnr > 30:
                        st.success("✅ Excellent image quality!")
                    elif psnr > 20:
                        st.info("ℹ️ Good image quality")
                    else:
                        st.warning("⚠️ Image quality needs improvement")
                
                except Exception as e:
                    st.error(f"Error calculating metrics: {e}")
    
    # Logout button in sidebar
    if st.session_state.authenticated:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

if __name__ == "__main__":
    main()
