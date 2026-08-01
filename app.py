import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Page Header Layout
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
# 1. Page Header Layout
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# --- COPY AND PASTE THIS NEW STYLING BLOCK ---
st.markdown("""
    <style>
    /* Premium clinical dark slate background */
    .stApp {
        background-color: #1E222B !important;
    }
    
    /* Force all text headers and paragraphs to crisp off-white for readability */
    .stApp p, .stApp li, .stApp span, .stApp h1, .stApp h2, .stApp h3 {
        color: #F3F4F6 !important;
    }
    
    /* Clean, professional card box for the file uploader */
    [data-testid="stFileUploader"] {
        background-color: #282C34;
        border: 1px solid #3F4451;
        border-radius: 8px;
        padding: 15px;
    }

    /* FIX: Force the "Browse files" button text and label inside the uploader to be highly visible */
    [data-testid="stFileUploader"] button {
        background-color: #3B82F6 !important; /* Premium medical blue button */
        color: #FFFFFF !important;            /* High-contrast crisp white text */
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }

    /* Add a subtle hover animation effect when mouse scrolls over the button */
    [data-testid="stFileUploader"] button:hover {
        background-color: #2563EB !important; /* Darker blue shade on hover */
    }

    /* Force standard uploader subtexts ("Limit 200MB per file") to stay readable */
    [data-testid="stFileUploader"] section {
        color: #9CA3AF !important;
    }
    </style>
    """, unsafe_allow_html=True)





st.title("🧠 Brain Tumor Detection AI Assistant")
st.write("Upload a brain MRI scan image below to let the neural network analyze it.")



# 2. Load your pre-trained model file safely
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("brain_tumor_cnn_model.keras")

model = load_my_model()
CATEGORIES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# 3. Create the Visual Upload Button Widget
uploaded_file = st.file_uploader("Choose a brain MRI image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
    st.write("🔄 Analyzing scan pattern structure...")
    
    # 4. Image Preprocessing Matrix Alignment
    img_array = np.array(image.convert('L')) # Convert to grayscale matrix
    resized_array = cv2.resize(img_array, (128, 128)) # Format dimensions
    normalized_img = resized_array.astype('float32') / 255.0 # Scale pixels
    final_input = np.expand_dims(np.expand_dims(normalized_img, axis=-1), axis=0) # Add batch dimension
    
    # 5. Execute Prediction
    predictions = model.predict(final_input)
    predicted_class_index = np.argmax(predictions)
    diagnosis = CATEGORIES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    
    
    # 6. Translate Multi-Class labels into clear Binary Output
    st.subheader("📋 Diagnostic Conclusion:")
        
    if diagnosis == 'no_tumor':
        # FIX: Changed from confidence[predicted_class_index] to confidence
        st.success(f"✅ Healthy Brain Pattern Detected! (Confidence: {confidence:.4f}%)")

        st.write("""
        **Scan Summary:** The neural network analyzed the structural layout of this MRI and found **no definitive signs** of aggressive cellular masses or tissue anomalies. 
                 
        **General Brain Health Information:**
        * A typical healthy brain MRI shows symmetrical structures, clean tissue boundaries, and clear ventricles without any fluid buildup or blockages.
        * Even if the AI predicts a normal scan, regular checkups are important if physical symptoms (like persistent headaches or dizziness) continue.
        """)
        
    else:
        # FIX: Changed from confidence[predicted_class_index] to confidence
        st.error(f"🚨 Tumor Detected: {diagnosis.replace('_', ' ').title()} ({confidence:.4f}%)")

        st.write(f"""
        **Scan Summary:** The network observed localized geometric variations and tissue density changes consistent with an abnormal mass growth classified as a **{diagnosis.replace('_', ' ').title()}**.
                 
        **About This Classification:**
        * **Glioma Tumor:** Growth originating in the glial cells that support neurons. They require detailed scanning to track tissue integration.
        * **Meningioma Tumor:** A tumor arising from the meninges (the protective layers covering the brain and spinal cord). They are often slow-growing but can exert pressure on surrounding brain tissue.
        * **Pituitary Tumor:** An abnormal growth in the pituitary gland at the base of the brain, which can occasionally affect hormone levels or vision pathways.
        """)
        
     # Standard Medical Disclaimer Anchor
    st.caption("⚠️ **Disclaimer:** This analysis is generated automatically by a deep learning model for educational and screening assistance. It does not replace a professional evaluation by a radiologist or neurologist.")

        
     

