"""
Batch Face Swap Script for Surveillance Dataset
Uses InsightFace with minimal dependencies
"""

import os
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
from pathlib import Path
import random

# Configuration
INPUT_FOLDER = "surveillance_images"  # Your original images folder
OUTPUT_FOLDER = "deepfake_images"     # Where swapped images will be saved
MODEL_PATH = "./checkpoints/inswapper_128.onnx"  # Face swap model path

class BatchFaceSwapper:
    def __init__(self, model_path):
        """Initialize the face swapper with minimal dependencies"""
        print("Initializing Face Analysis...")
        
        # Initialize face analysis for detection
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load the face swap model
        print(f"Loading face swap model from {model_path}")
        self.swapper = insightface.model_zoo.get_model(model_path, download=False)
        
    def get_faces(self, image):
        """Detect all faces in an image"""
        faces = self.app.get(image)
        return faces
    
    def swap_face(self, source_img, target_img, source_face, target_face):
        """Perform the actual face swap"""
        result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
        return result
    
    def process_batch(self, input_folder, output_folder, swap_mode='random'):
        """
        Process all images in the input folder
        
        Args:
            input_folder: Path to original surveillance images
            output_folder: Path where deepfakes will be saved
            swap_mode: 'random' (swap with random face) or 'sequential' (swap pairs sequentially)
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        
        if len(image_files) == 0:
            print(f"No images found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Store all faces from the dataset
        print("Extracting faces from all images...")
        face_database = []
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            faces = self.get_faces(img)
            if len(faces) > 0:
                face_database.append({
                    'path': img_path,
                    'image': img,
                    'faces': faces
                })
        
        print(f"Extracted faces from {len(face_database)} images")
        
        if len(face_database) < 2:
            print("Need at least 2 images with detectable faces for face swapping")
            return
        
        # Perform face swapping
        print("Starting face swap process...")
        swap_count = 0
        
        for i, item in enumerate(face_database):
            target_img = item['image']
            target_faces = item['faces']
            original_filename = item['path'].name
            
            # Skip if no face detected
            if len(target_faces) == 0:
                continue
            
            # Select source face based on mode
            if swap_mode == 'random':
                # Pick a random image that's not the current one
                source_idx = random.choice([j for j in range(len(face_database)) if j != i])
            else:  # sequential
                # Swap with the next image (circular)
                source_idx = (i + 1) % len(face_database)
            
            source_faces = face_database[source_idx]['faces']
            
            if len(source_faces) == 0:
                continue
            
            # Use the first face from each image
            source_face = source_faces[0]
            target_face = target_faces[0]
            
            try:
                # Perform face swap
                result_img = self.swap_face(
                    face_database[source_idx]['image'],
                    target_img,
                    source_face,
                    target_face
                )
                
                # Save the result
                output_path = os.path.join(output_folder, f"swapped_{original_filename}")
                cv2.imwrite(output_path, result_img)
                
                swap_count += 1
                print(f"[{swap_count}/{len(face_database)}] Processed: {original_filename}")
                
            except Exception as e:
                print(f"Error processing {original_filename}: {str(e)}")
                continue
        
        print(f"\nCompleted! {swap_count} face-swapped images saved to {output_folder}")
    
    def process_single_pair(self, source_img_path, target_img_path, output_path):
        """Swap faces between two specific images"""
        source_img = cv2.imread(source_img_path)
        target_img = cv2.imread(target_img_path)
        
        source_faces = self.get_faces(source_img)
        target_faces = self.get_faces(target_img)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            print("No faces detected in one or both images")
            return None
        
        result = self.swap_face(source_img, target_img, source_faces[0], target_faces[0])
        cv2.imwrite(output_path, result)
        print(f"Saved face-swapped image to {output_path}")
        return result


def download_model():
    """Download the inswapper model if not present"""
    import urllib.request
    
    os.makedirs("checkpoints", exist_ok=True)
    model_url = "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading face swap model (this may take a few minutes)...")
        try:
            urllib.request.urlretrieve(model_url, MODEL_PATH)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please download manually from:")
            print(model_url)
            return False
    else:
        print("Model already exists!")
    
    return True


def main():
    """Main function to run the batch face swapper"""
    print("=" * 60)
    print("Batch Face Swapper for Surveillance Dataset")
    print("=" * 60)
    
    # Download model if needed
    if not download_model():
        return
    
    # Initialize the face swapper
    try:
        swapper = BatchFaceSwapper(MODEL_PATH)
    except Exception as e:
        print(f"Error initializing face swapper: {e}")
        print("\nMake sure you have installed the required packages:")
        print("pip install insightface opencv-python onnxruntime")
        return
    
    # Process the batch
    swapper.process_batch(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        swap_mode='random'  # Can be 'random' or 'sequential'
    )
    
    print("\nAll done! Check the output folder for deepfake images.")


if __name__ == "__main__":
    main()
