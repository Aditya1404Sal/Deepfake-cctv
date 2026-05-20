# Modified FaceForensics++ Frame Extraction for Desktop
# Specifically configured for: ~/Desktop/original and ~/Desktop/Deepfakes

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/Desktop/extraction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SURVEILLANCE DEGRADATION
# ============================================================================

class SurveillanceDegradation:
    """Apply realistic surveillance degradation to frames."""
    
    @staticmethod
    def add_low_light_filter(image: np.ndarray, intensity: float = 0.35) -> np.ndarray:
        """Reduce brightness for low-light surveillance."""
        alpha = 1.0 - intensity
        beta = -intensity * 40
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, noise_level: float = 0.04) -> np.ndarray:
        """Add camera sensor noise."""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_compression_artifacts(image: np.ndarray, quality: int = 75) -> np.ndarray:
        """Add JPEG compression artifacts."""
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    @staticmethod
    def apply_degradation(image: np.ndarray, preset: str = 'mid_quality') -> np.ndarray:
        """Apply surveillance degradation based on preset."""
        if preset == 'mid_quality':
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.35)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.04)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=75)
        elif preset == 'high_quality':
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.2)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.02)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=85)
        elif preset == 'low_quality':
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.45)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.06)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=65)
        else:  # extreme
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.55)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.08)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=50)
        
        return img


# ============================================================================
# FRAME EXTRACTOR
# ============================================================================

class FrameExtractor:
    """Extract frames from MP4 videos with degradation."""
    
    def __init__(self, input_dir: str, output_dir: str, preset: str = 'mid_quality',
                 frame_skip: int = 2, max_frames: Optional[int] = 150, delete_videos: bool = True):
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preset = preset
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.delete_videos = delete_videos
        
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'space_freed_mb': 0.0,
            'start_time': datetime.now().isoformat()
        }
    
    def extract_video(self, video_path: Path) -> dict:
        """Extract frames from a single video."""
        result = {
            'video': video_path.name,
            'frames': 0,
            'success': False,
            'error': None,
            'size_mb': 0.0
        }
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result['error'] = "Could not open video"
                logger.error(f"✗ Failed to open: {video_path.name}")
                return result
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Processing: {video_path.name} ({total_frames} frames)")
            
            # Create output directory for this video
            video_output_dir = self.output_dir / video_path.stem
            video_output_dir.mkdir(exist_ok=True)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Stop if max frames reached
                if self.max_frames and extracted_count >= self.max_frames:
                    break
                
                # Apply degradation
                degraded_frame = SurveillanceDegradation.apply_degradation(frame, self.preset)
                
                # Save frame
                frame_filename = f"{video_path.stem}_{extracted_count:06d}.jpg"
                frame_path = video_output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), degraded_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                extracted_count += 1
                frame_count += 1
                
                if extracted_count % 50 == 0:
                    logger.info(f"  ✓ Extracted {extracted_count} frames...")
            
            cap.release()
            
            # Get file size and delete
            video_size_mb = video_path.stat().st_size / (1024 * 1024)
            result['size_mb'] = video_size_mb
            
            if self.delete_videos:
                try:
                    video_path.unlink()
                    logger.info(f"  Deleted: {video_path.name} (freed {video_size_mb:.2f} MB)")
                    result['space_freed'] = video_size_mb
                except Exception as e:
                    logger.warning(f"Could not delete {video_path.name}: {e}")
            
            result['success'] = True
            result['frames'] = extracted_count
            
            return result
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"✗ Error processing {video_path.name}: {e}")
            return result
    
    def process_directory(self):
        """Process all MP4 files in directory."""
        mp4_files = sorted(list(self.input_dir.glob('**/*.mp4')))
        
        if not mp4_files:
            logger.error(f"No MP4 files found in {self.input_dir}")
            return self.stats
        
        logger.info(f"Found {len(mp4_files)} MP4 files")
        self.stats['total_videos'] = len(mp4_files)
        
        for idx, video_path in enumerate(mp4_files, 1):
            logger.info(f"\n[{idx}/{len(mp4_files)}] Processing...")
            result = self.extract_video(video_path)
            
            if result['success']:
                self.stats['processed_videos'] += 1
                self.stats['total_frames'] += result['frames']
                if 'space_freed' in result:
                    self.stats['space_freed_mb'] += result['space_freed']
                logger.info(f"✓ Success: {result['frames']} frames extracted")
            else:
                self.stats['failed_videos'] += 1
        
        self.print_summary()
        return self.stats
    
    def print_summary(self):
        """Print and save statistics."""
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*70)
        logger.info(f"Videos processed: {self.stats['processed_videos']}/{self.stats['total_videos']}")
        logger.info(f"Failed videos: {self.stats['failed_videos']}")
        logger.info(f"Total frames extracted: {self.stats['total_frames']:,}")
        logger.info(f"Space freed: {self.stats['space_freed_mb']:.2f} MB")
        logger.info(f"Output location: {self.output_dir}")
        logger.info("="*70 + "\n")
        
        # Save stats
        stats_file = self.output_dir / 'stats.json'
        self.stats['end_time'] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Stats saved to: {stats_file}")


# ============================================================================
# SIMPLIFIED DESKTOP PROCESSING
# ============================================================================

def process_desktop_folders():
    """Process original and Deepfakes folders from Desktop."""
    
    desktop = Path('~/Desktop').expanduser()
    original_dir = desktop / 'original'
    deepfakes_dir = desktop / 'Deepfakes'
    output_base = desktop / 'extracted_frames'
    
    logger.info("="*70)
    logger.info("FaceForensics++ Desktop Frame Extraction")
    logger.info("="*70)
    
    # Check if directories exist
    if not original_dir.exists():
        logger.error(f"Original folder not found: {original_dir}")
        return
    
    if not deepfakes_dir.exists():
        logger.error(f"Deepfakes folder not found: {deepfakes_dir}")
        return
    
    logger.info(f"Source directories:")
    logger.info(f"  Real videos: {original_dir}")
    logger.info(f"  Fake videos: {deepfakes_dir}")
    logger.info(f"Output directory: {output_base}")
    
    # Process REAL videos
    logger.info("\n" + "-"*70)
    logger.info("PROCESSING REAL VIDEOS (original)")
    logger.info("-"*70)
    real_extractor = FrameExtractor(
        input_dir=str(original_dir),
        output_dir=str(output_base / 'real'),
        preset='mid_quality',
        frame_skip=2,
        max_frames=20,
        delete_videos=False  # Set to True to delete originals
    )
    real_extractor.process_directory()
    
    # Process FAKE videos (Deepfakes)
    logger.info("\n" + "-"*70)
    logger.info("PROCESSING FAKE VIDEOS (Deepfakes)")
    logger.info("-"*70)
    fake_extractor = FrameExtractor(
        input_dir=str(deepfakes_dir),
        output_dir=str(output_base / 'fake'),
        preset='mid_quality',
        frame_skip=2,
        max_frames=20,
        delete_videos=False  # Set to True to delete originals
    )
    fake_extractor.process_directory()
    
    # Final summary
    total_real_frames = real_extractor.stats['total_frames']
    total_fake_frames = fake_extractor.stats['total_frames']
    total_frames = total_real_frames + total_fake_frames
    
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Real frames extracted: {total_real_frames:,}")
    logger.info(f"Fake frames extracted: {total_fake_frames:,}")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Dataset location: {output_base}")
    logger.info(f"Log file: ~/Desktop/extraction.log")
    logger.info("="*70)
    
    # Directory structure
    logger.info(f"\nDataset structure:")
    logger.info(f"  {output_base}/")
    logger.info(f"  ├── real/")
    logger.info(f"  │   ├── video_001/")
    logger.info(f"  │   │   ├── video_001_000000.jpg")
    logger.info(f"  │   │   └── ...")
    logger.info(f"  │   └── video_N/")
    logger.info(f"  └── fake/")
    logger.info(f"      ├── deepfake_001/")
    logger.info(f"      │   ├── deepfake_001_000000.jpg")
    logger.info(f"      │   └── ...")
    logger.info(f"      └── deepfake_N/")


# ============================================================================
# QUICK UTILITIES
# ============================================================================

def count_frames_in_dataset(dataset_dir: str = '~/Desktop/extracted_frames'):
    """Count total frames in extracted dataset."""
    dataset_path = Path(dataset_dir).expanduser()
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    real_frames = len(list((dataset_path / 'real').glob('**/*.jpg')))
    fake_frames = len(list((dataset_path / 'fake').glob('**/*.jpg')))
    
    print(f"\nDataset Statistics:")
    print(f"  Real frames: {real_frames:,}")
    print(f"  Fake frames: {fake_frames:,}")
    print(f"  Total frames: {real_frames + fake_frames:,}")
    print(f"  Balance: {real_frames/(real_frames+fake_frames)*100:.1f}% real, {fake_frames/(real_frames+fake_frames)*100:.1f}% fake")


def get_dataset_size(dataset_dir: str = '~/Desktop/extracted_frames'):
    """Get total disk size of dataset."""
    dataset_path = Path(dataset_dir).expanduser()
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    total_size = sum(f.stat().st_size for f in dataset_path.glob('**/*') if f.is_file())
    size_gb = total_size / (1024**3)
    
    print(f"Dataset size: {size_gb:.2f} GB")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("FaceForensics++ Desktop Frame Extraction Tool")
    print("="*70 + "\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'extract':
            # Start extraction
            process_desktop_folders()
        
        elif command == 'count':
            # Count frames
            count_frames_in_dataset()
        
        elif command == 'size':
            # Get size
            get_dataset_size()
        
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python script.py extract   - Extract frames from Desktop folders")
            print("  python script.py count     - Count extracted frames")
            print("  python script.py size      - Get dataset disk size")
    
    else:
        # Default: run extraction
        process_desktop_folders()
