# Augmentation Script - Create random variations of extracted frames
# FIXED: Handles subdirectories properly
# Random blur, low-light, noise, compression

import cv2
import numpy as np
from pathlib import Path
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / 'Desktop' / 'augmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# RANDOM AUGMENTATION FUNCTIONS
# ============================================================================

class RandomAugmentation:
    """Apply random low-light effects and blur to create variations."""
    
    @staticmethod
    def add_random_blur(image: np.ndarray) -> np.ndarray:
        """Apply random blur (motion blur or Gaussian blur)."""
        blur_type = random.choice(['motion', 'gaussian'])
        
        if blur_type == 'motion':
            # Motion blur
            kernel_size = random.choice([3, 5, 7])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            return cv2.filter2D(image, -1, kernel)
        else:
            # Gaussian blur
            kernel_size = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def add_random_low_light(image: np.ndarray) -> np.ndarray:
        """Apply random low-light filter with varying intensity."""
        # Random intensity between 0.1 and 0.5
        intensity = random.uniform(0.1, 0.5)
        
        alpha = 1.0 - intensity
        beta = -intensity * 40
        
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def add_random_noise(image: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise."""
        noise_level = random.uniform(0.01, 0.08)
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_random_compression(image: np.ndarray) -> np.ndarray:
        """Add random JPEG compression."""
        quality = random.randint(50, 85)
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    @staticmethod
    def add_random_brightness(image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness."""
        brightness_factor = random.uniform(0.7, 1.3)
        adjusted = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_random_augmentation(image: np.ndarray, apply_probability: float = 0.7) -> np.ndarray:
        """
        Apply random augmentations to image.
        Each augmentation has apply_probability of being applied.
        """
        # Randomly decide which augmentations to apply
        if random.random() < apply_probability:
            image = RandomAugmentation.add_random_blur(image)
        
        if random.random() < apply_probability:
            image = RandomAugmentation.add_random_low_light(image)
        
        if random.random() < apply_probability:
            image = RandomAugmentation.add_random_noise(image)
        
        if random.random() < apply_probability:
            image = RandomAugmentation.add_random_compression(image)
        
        if random.random() < apply_probability:
            image = RandomAugmentation.add_random_brightness(image)
        
        return image


class AugmentationPipeline:
    """Batch augmentation for real and fake directories (handles subdirectories)."""
    
    def __init__(self, input_dir: str, output_dir: str, num_augmentations: int = 2):
        """
        Initialize augmentation pipeline.
        
        Args:
            input_dir: Directory with real/ and fake/ folders (can have subdirectories)
            output_dir: Where to save augmented frames (flat structure)
            num_augmentations: How many augmented versions per image
        """
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'real').mkdir(exist_ok=True)
        (self.output_dir / 'fake').mkdir(exist_ok=True)
        
        self.num_augmentations = num_augmentations
    
    def get_all_images_recursive(self, folder_path: Path):
        """
        Recursively find all images in folder and subdirectories.
        """
        image_files = sorted(
            list(folder_path.glob('**/*.jpg')) + 
            list(folder_path.glob('**/*.png'))
        )
        return image_files
    
    def augment_folder(self, folder_type: str = 'real') -> dict:
        """
        Augment all images in a folder (real or fake).
        Handles subdirectories automatically.
        
        Args:
            folder_type: 'real' or 'fake'
        
        Returns:
            Statistics dictionary
        """
        folder_path = self.input_dir / folder_type
        output_folder = self.output_dir / folder_type
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            return {'error': f'{folder_path} not found'}
        
        # Get all images (recursively from subdirectories)
        image_files = self.get_all_images_recursive(folder_path)
        
        if not image_files:
            logger.error(f"No images found in {folder_path} (including subdirectories)")
            return {'error': f'No images in {folder_path}'}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Augmenting {folder_type.upper()} images: {len(image_files)} found")
        logger.info(f"{'='*70}")
        logger.info(f"Source: {folder_path}")
        logger.info(f"Output: {output_folder}")
        
        stats = {
            'folder': folder_type,
            'original_images': len(image_files),
            'augmented_images': 0,
            'total_images': 0,
            'errors': 0
        }
        
        for idx, img_path in enumerate(image_files):
            try:
                # Read original image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to read: {img_path.name}")
                    stats['errors'] += 1
                    continue
                
                # Create unique filename to avoid collisions
                # Use full path relative to source to create unique names
                relative_path = img_path.relative_to(folder_path)
                unique_name = str(relative_path).replace('/', '_').replace('\\', '_')
                
                # Save original (with unique name in flat structure)
                original_output = output_folder / unique_name
                cv2.imwrite(str(original_output), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                stats['total_images'] += 1
                
                # Create augmented versions
                for aug_num in range(self.num_augmentations):
                    # Apply random augmentations
                    augmented = RandomAugmentation.apply_random_augmentation(image)
                    
                    # Save augmented image
                    base_name = Path(unique_name).stem
                    ext = Path(unique_name).suffix
                    aug_filename = f"{base_name}_aug_{aug_num}{ext}"
                    aug_output = output_folder / aug_filename
                    
                    cv2.imwrite(str(aug_output), augmented, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    stats['augmented_images'] += 1
                    stats['total_images'] += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"  ✓ Processed {idx + 1}/{len(image_files)} images")
            
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                stats['errors'] += 1
        
        logger.info(f"\n✓ {folder_type.upper()} AUGMENTATION COMPLETE")
        logger.info(f"  Original images: {stats['original_images']}")
        logger.info(f"  Augmented versions: {stats['augmented_images']}")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Errors: {stats['errors']}")
        
        return stats
    
    def augment_all(self) -> dict:
        """Augment both real and fake folders."""
        logger.info("\n" + "="*70)
        logger.info("AUGMENTATION PIPELINE - HANDLING SUBDIRECTORIES")
        logger.info("="*70)
        
        real_stats = self.augment_folder('real')
        fake_stats = self.augment_folder('fake')
        
        total_images = real_stats.get('total_images', 0) + fake_stats.get('total_images', 0)
        
        logger.info("\n" + "="*70)
        logger.info("AUGMENTATION PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Real images: {real_stats.get('total_images', 0):,}")
        logger.info(f"Fake images: {fake_stats.get('total_images', 0):,}")
        logger.info(f"Total images created: {total_images:,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Output structure: FLAT (all images in real/ and fake/)")
        logger.info("="*70 + "\n")
        
        return {'real': real_stats, 'fake': fake_stats, 'total': total_images}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SURVEILLANCE DEEPFAKE DETECTION - AUGMENTATION TOOL")
    print("FIXED VERSION: Handles subdirectories")
    print("="*70 + "\n")
    
    # Configuration
    input_directory = '~/Desktop/extracted_frames'
    output_directory = '~/Desktop/augmented_frames'
    num_augmentations = 2  # Create 2 augmented versions per image
    
    logger.info(f"Input directory: {Path(input_directory).expanduser()}")
    logger.info(f"Output directory: {Path(output_directory).expanduser()}")
    logger.info(f"Augmentations per image: {num_augmentations}")
    logger.info(f"\nHandling subdirectory structure:")
    logger.info(f"  extracted_frames/real/video_001/ → augmented_frames/real/ (flat)")
    logger.info(f"  extracted_frames/fake/deepfake_001/ → augmented_frames/fake/ (flat)")
    
    # Run augmentation
    augmenter = AugmentationPipeline(
        input_dir=input_directory,
        output_dir=output_directory,
        num_augmentations=num_augmentations
    )
    
    result = augmenter.augment_all()
    
    logger.info("\nAugmentation complete! Next step: Run finetune.py to train the model.")
    logger.info(f"\nTo verify, check:")
    logger.info(f"  ls -la ~/Desktop/augmented_frames/real/ | head")
    logger.info(f"  ls -la ~/Desktop/augmented_frames/fake/ | head")
