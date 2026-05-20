# FaceForensics++ c23 Frame Extraction + Low-Light Degradation Pipeline
# Extracts frames from MP4 videos, applies surveillance degradation, deletes videos to save space

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import argparse
from datetime import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOW-LIGHT & SURVEILLANCE DEGRADATION FUNCTIONS
# ============================================================================

class SurveillanceDegradation:
    """Apply realistic surveillance-quality degradation to frames."""
    
    @staticmethod
    def add_low_light_filter(image: np.ndarray, intensity: float = 0.35) -> np.ndarray:
        """
        Reduce brightness to simulate low-light surveillance.
        intensity: 0.0 (original) to 1.0 (completely dark)
        For mid-quality surveillance: 0.3-0.4
        """
        alpha = 1.0 - intensity
        beta = -intensity * 40
        low_light = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return low_light
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, noise_level: float = 0.04) -> np.ndarray:
        """
        Add Gaussian noise to simulate camera sensor noise.
        noise_level: standard deviation as fraction of 255
        For mid-quality: 0.03-0.05
        """
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_motion_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Add motion blur (optional, adds processing time)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        blurred = cv2.filter2D(image, -1, kernel)
        return blurred
    
    @staticmethod
    def add_compression_artifacts(image: np.ndarray, quality: int = 75) -> np.ndarray:
        """
        Add JPEG compression artifacts (most important for realism).
        quality: 10-100 (lower = more artifacts)
        For mid-quality surveillance: 70-80
        """
        _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded
    
    @staticmethod
    def apply_surveillance_degradation(
        image: np.ndarray,
        preset: str = 'mid_quality',
        skip_motion_blur: bool = True  # Skip to save processing time
    ) -> np.ndarray:
        """
        Apply realistic surveillance degradation.
        
        preset options:
        - 'mid_quality': Balanced (recommended for training)
        - 'high_quality': Light degradation
        - 'low_quality': Heavy degradation
        - 'extreme': Very heavy (too extreme)
        
        skip_motion_blur: Set True to speed up processing
        """
        
        if preset == 'mid_quality':
            # Low-light + noise + compression (balanced)
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.35)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.04)
            if not skip_motion_blur:
                img = SurveillanceDegradation.add_motion_blur(img, kernel_size=3)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=75)
            
        elif preset == 'high_quality':
            # Light degradation (easier detection)
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.2)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.02)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=85)
            
        elif preset == 'low_quality':
            # Heavy degradation (harder detection, more realistic low-light)
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.45)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.06)
            if not skip_motion_blur:
                img = SurveillanceDegradation.add_motion_blur(img, kernel_size=5)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=65)
            
        elif preset == 'extreme':
            # Extreme degradation (for testing limits)
            img = SurveillanceDegradation.add_low_light_filter(image, intensity=0.55)
            img = SurveillanceDegradation.add_gaussian_noise(img, noise_level=0.08)
            if not skip_motion_blur:
                img = SurveillanceDegradation.add_motion_blur(img, kernel_size=7)
            img = SurveillanceDegradation.add_compression_artifacts(img, quality=50)
        
        return img


# ============================================================================
# FRAME EXTRACTION FROM MP4 VIDEO
# ============================================================================

class FaceForensicsFrameExtractor:
    """Extract frames from FaceForensics++ c23 MP4 videos."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        degradation_preset: str = 'mid_quality',
        frame_skip: int = 1,
        max_frames_per_video: Optional[int] = None,
        delete_after_extraction: bool = True,
        skip_motion_blur: bool = True
    ):
        """
        Initialize extractor.
        
        Args:
            input_dir: Path to FaceForensics++ c23 videos
            output_dir: Where to save extracted frames
            degradation_preset: 'mid_quality', 'high_quality', 'low_quality', 'extreme'
            frame_skip: Extract every Nth frame (1=all frames, 5=every 5th frame)
            max_frames_per_video: Limit frames per video (None=all)
            delete_after_extraction: Delete MP4 after processing
            skip_motion_blur: Skip motion blur to speed up
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.degradation_preset = degradation_preset
        self.frame_skip = frame_skip
        self.max_frames_per_video = max_frames_per_video
        self.delete_after_extraction = delete_after_extraction
        self.skip_motion_blur = skip_motion_blur
        
        # Create subdirectories for real and fake videos
        self.real_output_dir = self.output_dir / 'real'
        self.fake_output_dir = self.output_dir / 'fake'
        self.real_output_dir.mkdir(exist_ok=True)
        self.fake_output_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames_extracted': 0,
            'total_size_freed_mb': 0
        }
    
    def extract_frames_from_video(
        self,
        video_path: str,
        is_real: bool = True
    ) -> dict:
        """
        Extract frames from a single MP4 video.
        
        Args:
            video_path: Path to MP4 file
            is_real: True if real video, False if fake/deepfake
        
        Returns:
            Dictionary with extraction statistics
        """
        video_path = Path(video_path)
        result = {
            'video': video_path.name,
            'frames_extracted': 0,
            'success': False,
            'error': None
        }
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result['error'] = "Could not open video"
                logger.error(f"Failed to open: {video_path.name}")
                return result
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Processing: {video_path.name} ({total_frames} frames, {frame_width}x{frame_height})")
            
            # Determine output directory (real or fake)
            output_video_dir = self.real_output_dir if is_real else self.fake_output_dir
            video_name = video_path.stem
            frames_output_dir = output_video_dir / video_name
            frames_output_dir.mkdir(exist_ok=True)
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if needed
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Stop if max frames reached
                if self.max_frames_per_video and extracted_count >= self.max_frames_per_video:
                    break
                
                # Apply degradation
                degraded_frame = SurveillanceDegradation.apply_surveillance_degradation(
                    frame,
                    preset=self.degradation_preset,
                    skip_motion_blur=self.skip_motion_blur
                )
                
                # Save frame
                frame_filename = f"{video_name}_{extracted_count:06d}.jpg"
                frame_path = frames_output_dir / frame_filename
                
                # Use JPEG quality 85 for storage (already degraded, high quality storage)
                cv2.imwrite(
                    str(frame_path),
                    degraded_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                
                extracted_count += 1
                frame_count += 1
                
                # Progress logging every 100 frames
                if extracted_count % 100 == 0:
                    logger.info(f"  Extracted {extracted_count} frames from {video_path.name}")
            
            cap.release()
            
            # Get file size before deletion
            video_size_mb = video_path.stat().st_size / (1024 * 1024)
            
            # Delete video file after extraction
            if self.delete_after_extraction:
                try:
                    video_path.unlink()
                    logger.info(f"Deleted: {video_path.name} (freed {video_size_mb:.2f} MB)")
                    result['space_freed_mb'] = video_size_mb
                except Exception as e:
                    logger.warning(f"Failed to delete {video_path.name}: {e}")
            
            result['success'] = True
            result['frames_extracted'] = extracted_count
            result['video_size_mb'] = video_size_mb
            
            return result
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {video_path.name}: {e}")
            return result
    
    def process_directory(self) -> dict:
        """
        Process all MP4 files in input directory.
        Automatically categorizes as real or fake based on filename/path structure.
        """
        # Find all MP4 files
        mp4_files = list(self.input_dir.glob('**/*.mp4'))
        
        if not mp4_files:
            logger.error(f"No MP4 files found in {self.input_dir}")
            return self.stats
        
        logger.info(f"Found {len(mp4_files)} MP4 files to process")
        self.stats['total_videos'] = len(mp4_files)
        
        # Process each video
        for idx, video_path in enumerate(mp4_files, 1):
            logger.info(f"[{idx}/{len(mp4_files)}] Processing: {video_path.name}")
            
            # Determine if real or fake based on path/filename
            # FaceForensics structure: dataset_name/manipulated_sequences/method/videos
            is_real = 'original' in str(video_path).lower() or 'manipulated' not in str(video_path).lower()
            
            result = self.extract_frames_from_video(str(video_path), is_real=is_real)
            
            if result['success']:
                self.stats['processed_videos'] += 1
                self.stats['total_frames_extracted'] += result['frames_extracted']
                if 'space_freed_mb' in result:
                    self.stats['total_size_freed_mb'] += result['space_freed_mb']
                
                logger.info(f"✓ Success: {result['frames_extracted']} frames extracted")
            else:
                self.stats['failed_videos'] += 1
                logger.error(f"✗ Failed: {result['error']}")
        
        return self.stats
    
    def print_summary(self):
        """Print processing summary."""
        logger.info("=" * 80)
        logger.info("FRAME EXTRACTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total videos processed: {self.stats['processed_videos']}/{self.stats['total_videos']}")
        logger.info(f"Failed videos: {self.stats['failed_videos']}")
        logger.info(f"Total frames extracted: {self.stats['total_frames_extracted']:,}")
        logger.info(f"Space freed: {self.stats['total_size_freed_mb']:.2f} MB")
        logger.info(f"Degradation preset: {self.degradation_preset}")
        logger.info(f"Frame skip: {self.frame_skip}")
        logger.info("=" * 80)
        
        # Save stats to JSON
        stats_file = self.output_dir / 'extraction_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Stats saved to: {stats_file}")


# ============================================================================
# BATCH PROCESSING WITH RESUMPTION
# ============================================================================

class BatchProcessor:
    """Process multiple directories with resumption capability."""
    
    def __init__(self, base_input_dir: str, base_output_dir: str):
        self.base_input_dir = Path(base_input_dir)
        self.base_output_dir = Path(base_output_dir)
        self.processed_log = self.base_output_dir / 'processed_videos.txt'
        
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_processed_videos(self) -> set:
        """Get list of already processed videos (for resumption)."""
        if self.processed_log.exists():
            with open(self.processed_log, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def log_processed_video(self, video_name: str):
        """Log that a video was processed."""
        with open(self.processed_log, 'a') as f:
            f.write(f"{video_name}\n")
    
    def process_all(
        self,
        degradation_preset: str = 'mid_quality',
        frame_skip: int = 1,
        max_frames_per_video: Optional[int] = None,
        resume: bool = True
    ):
        """Process all videos with resumption support."""
        
        extractor = FaceForensicsFrameExtractor(
            input_dir=str(self.base_input_dir),
            output_dir=str(self.base_output_dir),
            degradation_preset=degradation_preset,
            frame_skip=frame_skip,
            max_frames_per_video=max_frames_per_video,
            delete_after_extraction=True,
            skip_motion_blur=True
        )
        
        stats = extractor.process_directory()
        extractor.print_summary()
        
        return stats


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from FaceForensics++ c23 videos with surveillance degradation'
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to FaceForensics++ c23 videos directory'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for extracted frames'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['high_quality', 'mid_quality', 'low_quality', 'extreme'],
        default='mid_quality',
        help='Surveillance degradation preset (default: mid_quality)'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Extract every Nth frame (1=all frames, 5=every 5th) (default: 1)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Max frames per video (None=all, 100=100 frames per video)'
    )
    
    parser.add_argument(
        '--no-delete',
        action='store_true',
        help='Do NOT delete MP4 files after extraction'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last processed video'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FaceForensics++ Frame Extraction Pipeline")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Degradation preset: {args.preset}")
    logger.info(f"Frame skip: {args.frame_skip}")
    logger.info(f"Max frames per video: {args.max_frames if args.max_frames else 'unlimited'}")
    logger.info(f"Delete MP4s after extraction: {not args.no_delete}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info("=" * 80)
    
    # Create extractor
    extractor = FaceForensicsFrameExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        degradation_preset=args.preset,
        frame_skip=args.frame_skip,
        max_frames_per_video=args.max_frames,
        delete_after_extraction=not args.no_delete,
        skip_motion_blur=True  # Always skip for speed
    )
    
    # Process directory
    stats = extractor.process_directory()
    extractor.print_summary()


if __name__ == '__main__':
    main()
