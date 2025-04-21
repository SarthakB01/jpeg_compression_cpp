import streamlit as st
import os
from PIL import Image
import tempfile
from jpeg_wrapper import compress_jpeg
import io
import base64

def main():

    if not os.path.exists('jpeg_compressor'):
        st.error("""
        JPEG compressor engine not found! Please ensure:
        1. The 'jpeg_compressor' binary exists
        2. It has execute permissions
        """)
        return
    
    
    st.title("JPEG Compression Tool")
    st.write("Upload an image and adjust compression quality")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a JPEG image", type=["jpg", "jpeg"])
    
    # Compression quality slider
    quality = st.slider(
        "Compression Quality", 
        min_value=0.002, 
        max_value=1.0, 
        value=0.5,
        help="1.0 = Best quality (low compression), 0.5 = Balanced, 0.1 = High compression, 0.002 = Extreme"
    )
    
    if uploaded_file is not None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Save uploaded file to temporary location
            temp_input = os.path.join(temp_dir, "input.jpg")
            temp_output = os.path.join(temp_dir, "output.jpg")
            
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Size information
            original_size = os.path.getsize(temp_input)
            st.write(f"Original file size: {original_size/1024:.2f} KB")
            
            # Compress button
            if st.button("Compress Image"):
                with st.spinner("Compressing..."):
                    # Call the wrapped compression function
                    success, output = compress_jpeg(temp_input, temp_output, quality)
                    
                    if success and os.path.exists(temp_output):
                        # Display compressed image
                        compressed_image = Image.open(temp_output)
                        st.image(compressed_image, caption=f"Compressed Image (Quality: {quality})", use_column_width=True)
                        
                        # Size information
                        compressed_size = os.path.getsize(temp_output)
                        st.write(f"Compressed file size: {compressed_size/1024:.2f} KB")
                        st.write(f"Compression ratio: {original_size/compressed_size:.2f}:1")
                        
                        # Show compression details
                        st.text_area("Compression Details", output, height=100)
                        
                        # Download button
                        with open(temp_output, "rb") as file:
                            btn = st.download_button(
                                label="Download Compressed Image",
                                data=file,
                                file_name="compressed.jpg",
                                mime="image/jpeg"
                            )
                    else:
                        st.error(f"Compression failed: {output}")

if __name__ == "__main__":
    main()