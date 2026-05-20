"""
CCTV Dataset Download Script
Downloads datasets from VIRAT, UCF-Crime, and EPFL
"""

import os
import requests
import logging
from pathlib import Path
from typing import List, Dict
import hashlib
from tqdm import tqdm
import urllib.request
import zipfile
import tarfile
import shutil
from bs4 import BeautifulSoup
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Automated downloader for CCTV surveillance datasets
    """
    
    def __init__(self, download_dir: str = "datasets"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'virat': {
                'name': 'VIRAT Video Dataset',
                'base_url': 'https://viratdata.org/',
                'description': '8.5 hours of HD surveillance video',
                'size': '~300GB'
            },
            'ucf_crime': {
                'name': 'UCF-Crime Dataset',
                'base_url': 'https://www.crcv.ucf.edu/projects/real-world/',
                'description': '1,900 surveillance videos, 128 hours',
                'size': '~150GB'
            },
            'epfl': {
                'name': 'EPFL Pedestrian Videos',
                'base_url': 'https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/',
                'description': 'Multi-view synchronized camera sequences',
                'size': '~50GB'
            },
            'mot17': {
                'name': 'MOT17 Challenge',
                'direct_url': 'https://motchallenge.net/data/MOT17.zip',
                'description': '14 video sequences for pedestrian tracking',
                'size': '~5.5GB'
            },
            'mot20': {
                'name': 'MOT20 Challenge',
                'direct_url': 'https://motchallenge.net/data/MOT20.zip',
                'description': '8 video sequences with dense crowds',
                'size': '~5GB'
            }
        }
    
    def download_file(self, url: str, output_path: str, 
                     chunk_size: int = 8192) -> bool:
        """
        Download a file with progress bar
        
        Args:
            url: URL to download from
            output_path: Local path to save file
            chunk_size: Download chunk size
            
        Returns:
            success: True if download successful
        """
        try:
            logger.info(f"Downloading from {url}")
            
            # Get file size
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=os.path.basename(output_path)) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def extract_archive(self, archive_path: str, extract_dir: str) -> bool:
        """
        Extract zip or tar archive
        
        Args:
            archive_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            success: True if extraction successful
        """
        try:
            logger.info(f"Extracting {archive_path}...")
            
            os.makedirs(extract_dir, exist_ok=True)
            
            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.error(f"Unsupported archive format: {archive_path}")
                return False
            
            logger.info(f"Extracted to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def verify_checksum(self, file_path: str, expected_hash: str, 
                       algorithm: str = 'sha256') -> bool:
        """
        Verify file integrity using checksum
        
        Args:
            file_path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm (md5, sha256, etc.)
            
        Returns:
            valid: True if checksum matches
        """
        try:
            hash_func = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            
            calculated_hash = hash_func.hexdigest()
            
            if calculated_hash == expected_hash:
                logger.info(f"Checksum verified for {file_path}")
                return True
            else:
                logger.error(f"Checksum mismatch for {file_path}")
                logger.error(f"Expected: {expected_hash}")
                logger.error(f"Got: {calculated_hash}")
                return False
                
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False
    
    def download_mot17(self) -> bool:
        """
        Download MOT17 Challenge dataset
        
        Returns:
            success: True if download successful
        """
        logger.info("="*80)
        logger.info("Downloading MOT17 Challenge Dataset")
        logger.info("="*80)
        
        dataset_dir = self.download_dir / 'MOT17'
        dataset_dir.mkdir(exist_ok=True)
        
        url = self.datasets['mot17']['direct_url']
        zip_path = dataset_dir / 'MOT17.zip'
        
        # Download
        if not zip_path.exists():
            success = self.download_file(url, str(zip_path))
            if not success:
                return False
        else:
            logger.info(f"Archive already exists: {zip_path}")
        
        # Extract
        extract_dir = dataset_dir / 'extracted'
        if not extract_dir.exists():
            success = self.extract_archive(str(zip_path), str(extract_dir))
            if not success:
                return False
        else:
            logger.info(f"Already extracted to: {extract_dir}")
        
        logger.info("MOT17 dataset ready!")
        return True
    
    def download_mot20(self) -> bool:
        """
        Download MOT20 Challenge dataset
        
        Returns:
            success: True if download successful
        """
        logger.info("="*80)
        logger.info("Downloading MOT20 Challenge Dataset")
        logger.info("="*80)
        
        dataset_dir = self.download_dir / 'MOT20'
        dataset_dir.mkdir(exist_ok=True)
        
        url = self.datasets['mot20']['direct_url']
        zip_path = dataset_dir / 'MOT20.zip'
        
        # Download
        if not zip_path.exists():
            success = self.download_file(url, str(zip_path))
            if not success:
                return False
        else:
            logger.info(f"Archive already exists: {zip_path}")
        
        # Extract
        extract_dir = dataset_dir / 'extracted'
        if not extract_dir.exists():
            success = self.extract_archive(str(zip_path), str(extract_dir))
            if not success:
                return False
        else:
            logger.info(f"Already extracted to: {extract_dir}")
        
        logger.info("MOT20 dataset ready!")
        return True
    
    def download_virat(self) -> bool:
        """
        Download VIRAT Video Dataset
        Note: VIRAT requires registration and manual download
        
        Returns:
            success: Always False (manual download required)
        """
        logger.info("="*80)
        logger.info("VIRAT Video Dataset Download Instructions")
        logger.info("="*80)
        
        logger.warning("VIRAT dataset requires registration and manual download")
        logger.info("\nSteps to download VIRAT:")
        logger.info("1. Visit: https://viratdata.org/")
        logger.info("2. Register for an account")
        logger.info("3. Navigate to 'Downloads' section")
        logger.info("4. Download VIRAT Video Dataset Release 2.0")
        logger.info("5. Extract files to: " + str(self.download_dir / 'VIRAT'))
        logger.info("\nDataset details:")
        logger.info(f"  - Name: {self.datasets['virat']['name']}")
        logger.info(f"  - Size: {self.datasets['virat']['size']}")
        logger.info(f"  - Description: {self.datasets['virat']['description']}")
        
        # Create placeholder directory
        virat_dir = self.download_dir / 'VIRAT'
        virat_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_path = virat_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("VIRAT Video Dataset\n")
            f.write("="*50 + "\n\n")
            f.write("This dataset requires manual download.\n\n")
            f.write("Download from: https://viratdata.org/\n\n")
            f.write("After downloading, place the files in this directory.\n")
        
        logger.info(f"\nCreated placeholder directory: {virat_dir}")
        logger.info(f"README created: {readme_path}")
        
        return False
    
    def download_ucf_crime(self) -> bool:
        """
        Download UCF-Crime Dataset
        Note: UCF-Crime requires registration and manual download
        
        Returns:
            success: Always False (manual download required)
        """
        logger.info("="*80)
        logger.info("UCF-Crime Dataset Download Instructions")
        logger.info("="*80)
        
        logger.warning("UCF-Crime dataset requires form submission and manual download")
        logger.info("\nSteps to download UCF-Crime:")
        logger.info("1. Visit: https://www.crcv.ucf.edu/projects/real-world/")
        logger.info("2. Fill out the data request form")
        logger.info("3. Wait for email with download link")
        logger.info("4. Download the dataset files")
        logger.info("5. Extract files to: " + str(self.download_dir / 'UCF-Crime'))
        logger.info("\nDataset details:")
        logger.info(f"  - Name: {self.datasets['ucf_crime']['name']}")
        logger.info(f"  - Size: {self.datasets['ucf_crime']['size']}")
        logger.info(f"  - Description: {self.datasets['ucf_crime']['description']}")
        
        # Alternative: Provide Kaggle link if available
        logger.info("\nAlternative download (may require Kaggle account):")
        logger.info("  - Some UCF-Crime videos may be available on Kaggle")
        logger.info("  - Search for 'UCF-Crime' on kaggle.com/datasets")
        
        # Create placeholder directory
        ucf_dir = self.download_dir / 'UCF-Crime'
        ucf_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_path = ucf_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("UCF-Crime Dataset\n")
            f.write("="*50 + "\n\n")
            f.write("This dataset requires manual download.\n\n")
            f.write("Request access from: https://www.crcv.ucf.edu/projects/real-world/\n\n")
            f.write("After downloading, place the files in this directory.\n")
        
        logger.info(f"\nCreated placeholder directory: {ucf_dir}")
        logger.info(f"README created: {readme_path}")
        
        return False
    
    def download_epfl(self) -> bool:
        """
        Download EPFL Pedestrian Videos
        Note: EPFL may require manual download
        
        Returns:
            success: Varies based on availability
        """
        logger.info("="*80)
        logger.info("EPFL Pedestrian Videos Download Instructions")
        logger.info("="*80)
        
        logger.info("\nSteps to download EPFL dataset:")
        logger.info("1. Visit: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/")
        logger.info("2. Navigate to download links section")
        logger.info("3. Download the following sequences:")
        logger.info("   - Laboratory sequences")
        logger.info("   - Terrace sequences")
        logger.info("   - Passageway sequences")
        logger.info("4. Extract files to: " + str(self.download_dir / 'EPFL'))
        logger.info("\nDataset details:")
        logger.info(f"  - Name: {self.datasets['epfl']['name']}")
        logger.info(f"  - Size: {self.datasets['epfl']['size']}")
        logger.info(f"  - Description: {self.datasets['epfl']['description']}")
        
        # Create placeholder directory
        epfl_dir = self.download_dir / 'EPFL'
        epfl_dir.mkdir(exist_ok=True)
        
        # Create README
        readme_path = epfl_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("EPFL Pedestrian Videos Dataset\n")
            f.write("="*50 + "\n\n")
            f.write("This dataset may require manual download.\n\n")
            f.write("Download from: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/\n\n")
            f.write("Recommended sequences:\n")
            f.write("  - Laboratory\n")
            f.write("  - Terrace\n")
            f.write("  - Passageway\n\n")
            f.write("After downloading, place the files in this directory.\n")
        
        logger.info(f"\nCreated placeholder directory: {epfl_dir}")
        logger.info(f"README created: {readme_path}")
        
        return False
    
    def download_all_datasets(self, skip_manual: bool = False) -> Dict[str, bool]:
        """
        Download all available datasets
        
        Args:
            skip_manual: Skip datasets requiring manual download
            
        Returns:
            results: Dictionary of dataset names and success status
        """
        results = {}
        
        logger.info("="*80)
        logger.info("Starting Download of All CCTV Datasets")
        logger.info("="*80)
        
        # MOT17 (automatic)
        logger.info("\n[1/5] Processing MOT17...")
        results['MOT17'] = self.download_mot17()
        
        # MOT20 (automatic)
        logger.info("\n[2/5] Processing MOT20...")
        results['MOT20'] = self.download_mot20()
        
        # VIRAT (manual)
        if not skip_manual:
            logger.info("\n[3/5] Processing VIRAT...")
            results['VIRAT'] = self.download_virat()
        else:
            logger.info("\n[3/5] Skipping VIRAT (manual download required)")
            results['VIRAT'] = False
        
        # UCF-Crime (manual)
        if not skip_manual:
            logger.info("\n[4/5] Processing UCF-Crime...")
            results['UCF-Crime'] = self.download_ucf_crime()
        else:
            logger.info("\n[4/5] Skipping UCF-Crime (manual download required)")
            results['UCF-Crime'] = False
        
        # EPFL (manual)
        if not skip_manual:
            logger.info("\n[5/5] Processing EPFL...")
            results['EPFL'] = self.download_epfl()
        else:
            logger.info("\n[5/5] Skipping EPFL (manual download required)")
            results['EPFL'] = False
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("Download Summary")
        logger.info("="*80)
        
        for dataset, success in results.items():
            status = "✓ Downloaded" if success else "✗ Manual download required"
            logger.info(f"{dataset:15s}: {status}")
        
        logger.info("\n" + "="*80)
        logger.info(f"All files saved to: {self.download_dir}")
        logger.info("="*80)
        
        return results
    
    def list_downloaded_datasets(self) -> Dict[str, Dict]:
        """
        List all downloaded datasets and their status
        
        Returns:
            info: Dictionary with dataset information
        """
        info = {}
        
        for dataset_name in ['MOT17', 'MOT20', 'VIRAT', 'UCF-Crime', 'EPFL']:
            dataset_path = self.download_dir / dataset_name
            
            if dataset_path.exists():
                # Count video files
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                video_files = []
                for ext in video_extensions:
                    video_files.extend(list(dataset_path.rglob(f'*{ext}')))
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
                total_size_gb = total_size / (1024**3)
                
                info[dataset_name] = {
                    'path': str(dataset_path),
                    'exists': True,
                    'video_count': len(video_files),
                    'total_size_gb': round(total_size_gb, 2)
                }
            else:
                info[dataset_name] = {
                    'path': str(dataset_path),
                    'exists': False,
                    'video_count': 0,
                    'total_size_gb': 0
                }
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Downloaded Datasets Summary")
        logger.info("="*80)
        
        for dataset_name, dataset_info in info.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Path: {dataset_info['path']}")
            logger.info(f"  Downloaded: {'Yes' if dataset_info['exists'] else 'No'}")
            logger.info(f"  Video files: {dataset_info['video_count']}")
            logger.info(f"  Total size: {dataset_info['total_size_gb']} GB")
        
        return info


def main():
    """Main function to download datasets"""
    
    # Initialize downloader
    downloader = DatasetDownloader(download_dir="datasets")
    
    print("\n" + "="*80)
    print("CCTV Dataset Downloader")
    print("="*80)
    print("\nThis script will download the following datasets:")
    print("1. MOT17 Challenge (automatic)")
    print("2. MOT20 Challenge (automatic)")
    print("3. VIRAT Video Dataset (manual)")
    print("4. UCF-Crime Dataset (manual)")
    print("5. EPFL Pedestrian Videos (manual)")
    print("\nNote: Some datasets require registration and manual download.")
    print("="*80)
    
    # Ask user preference
    choice = input("\nDownload automatic datasets only? (y/n): ").strip().lower()
    skip_manual = (choice == 'y')
    
    # Download all datasets
    results = downloader.download_all_datasets(skip_manual=skip_manual)
    
    # List downloaded datasets
    downloader.list_downloaded_datasets()
    
    print("\n" + "="*80)
    print("Download process complete!")
    print("="*80)
    print("\nFor manual downloads:")
    print("- Check the README files in each dataset directory")
    print("- Follow the instructions to complete the downloads")
    print("="*80)


if __name__ == "__main__":
    main()
