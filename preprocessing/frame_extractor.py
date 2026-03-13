"""
Frame Extractor for FaceForensics++ Dataset
Extracts frames from videos at 5 FPS and generates metadata CSV
Optimized with multiprocessing and efficient frame extraction
"""

import cv2
import argparse
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from configs.loggings import logger, setup_logging
from configs.settings import settings

# Suppress OpenCV warnings
warnings.filterwarnings('ignore')


def process_video_standalone(
    video_path_str: str,
    category: str,
    output_dir_str: str,
    base_output_path_str: str,
    target_fps: int,
    jpeg_quality: int,
    categories: Dict[str, str]
) -> List[Dict]:
    """
    Standalone function for multiprocessing - process single video
    Must be at module level for Windows multiprocessing compatibility
    
    Args:
        video_path_str: String path to video file
        category: Category name
        output_dir_str: String path to output directory
        base_output_path_str: String path to base output directory
        target_fps: Target FPS for extraction
        jpeg_quality: JPEG quality (0-100)
        categories: Dict mapping categories to labels
        
    Returns:
        List of metadata records
    """
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    base_output_path = Path(base_output_path_str)
    
    records = []
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return records
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        if video_fps == 0 or total_frames == 0:
            cap.release()
            return records
        
        # Calculate frame interval
        frame_interval = int(video_fps / target_fps)
        if frame_interval < 1:
            frame_interval = 1
        
        video_name = video_path.stem
        extracted_count = 0
        
        # OPTIMIZATION: Seek to specific frames instead of reading all
        for frame_idx in range(0, total_frames, frame_interval):
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Create frame filename
            frame_filename = f"{video_name}_frame_{extracted_count:05d}.jpg"
            frame_path = output_dir / frame_filename
            
            # Save frame with quality control
            cv2.imwrite(
                str(frame_path), 
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            
            # Calculate timestamp
            timestamp = frame_idx / video_fps
            
            # Create metadata record
            record = {
                'frame_path': str(frame_path.relative_to(base_output_path)),
                'video_name': video_name,
                'category': category,
                'label': categories[category],
                'frame_number': extracted_count,
                'original_frame_index': frame_idx,
                'timestamp': round(timestamp, 2),
                'video_fps': round(video_fps, 2),
                'extraction_fps': target_fps,
                'width': width,
                'height': height,
                'video_duration': round(duration, 2),
                'total_video_frames': total_frames,
                'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            records.append(record)
            extracted_count += 1
        
        cap.release()
        
    except Exception as e:
        pass  # Silent fail in worker process
    
    return records


class FrameExtractor:
    """Extract frames from videos and generate metadata"""
    
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        fps: Optional[int] = None,
        metadata_csv: Optional[str] = None,
        jpeg_quality: Optional[int] = None,
        num_workers: Optional[int] = None
    ):
        """
        Initialize FrameExtractor
        
        Args:
            dataset_path: Path to FaceForensics++ dataset (default: from settings)
            output_path: Path to save extracted frames (default: from settings)
            fps: Frames per second to extract (default: from settings)
            metadata_csv: Name of metadata CSV file (default: from settings)
            jpeg_quality: JPEG quality 0-100 (default: from settings)
            num_workers: Number of parallel workers (default: from settings)
        """
        self.dataset_path = dataset_path or settings.RAW_DATA_DIR
        self.output_path = output_path or settings.FRAME_DATA_DIR
        self.target_fps = fps or settings.TARGET_FPS
        self.metadata_csv = metadata_csv or settings.FRAME_EXTRACTION_METADATA_CSV
        self.jpeg_quality = jpeg_quality or settings.JPEG_QUALITY
        self.num_workers = num_workers or settings.NUM_WORKERS
        
        # Categories in the dataset
        self.categories = settings.DATASET_CATEGORIES

    def _load_video_properties_for_category(self, category: str) -> Dict[str, Dict]:
        """Load lightweight video properties used to rebuild metadata."""
        video_props: Dict[str, Dict] = {}
        category_path = self.dataset_path / category

        if not category_path.exists():
            return video_props

        video_files = list(category_path.glob("*.mp4"))
        for video_path in tqdm(video_files, desc=f"Indexing {category} videos", leave=False):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            duration = total_frames / fps if fps > 0 else 0
            video_props[video_path.stem] = {
                "video_fps": round(fps, 2) if fps else 0,
                "total_video_frames": total_frames,
                "width": width,
                "height": height,
                "video_duration": round(duration, 2),
            }

        return video_props

    def rebuild_metadata_from_frames(self) -> None:
        """Rebuild metadata CSV from existing extracted frame files only."""
        csv_path = self.output_path / self.metadata_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame_path",
            "video_name",
            "category",
            "label",
            "frame_number",
            "original_frame_index",
            "timestamp",
            "video_fps",
            "extraction_fps",
            "width",
            "height",
            "video_duration",
            "total_video_frames",
            "extraction_date",
        ]

        total_rows = 0
        real_rows = 0
        fake_rows = 0
        category_counts: Dict[str, int] = {}

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for category, label in self.categories.items():
                frame_dir = self.output_path / category
                if not frame_dir.exists():
                    logger.warning(f"Frame directory not found: {frame_dir}")
                    continue

                video_props = self._load_video_properties_for_category(category)
                frame_files = sorted(frame_dir.glob("*.jpg"))
                category_counts[category] = 0

                for frame_path in tqdm(frame_files, desc=f"Rebuilding {category}"):
                    stem = frame_path.stem
                    if "_frame_" not in stem:
                        continue

                    video_name, frame_idx_str = stem.rsplit("_frame_", 1)
                    try:
                        frame_number = int(frame_idx_str)
                    except ValueError:
                        continue

                    props = video_props.get(video_name, {})
                    video_fps = props.get("video_fps", 0)
                    frame_interval = max(1, int(video_fps / self.target_fps)) if video_fps else 1
                    original_frame_index = frame_number * frame_interval
                    timestamp = (original_frame_index / video_fps) if video_fps else 0

                    extraction_date = datetime.fromtimestamp(frame_path.stat().st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    row = {
                        "frame_path": str(frame_path.relative_to(self.output_path)),
                        "video_name": video_name,
                        "category": category,
                        "label": label,
                        "frame_number": frame_number,
                        "original_frame_index": original_frame_index,
                        "timestamp": round(timestamp, 2),
                        "video_fps": video_fps,
                        "extraction_fps": self.target_fps,
                        "width": props.get("width", 0),
                        "height": props.get("height", 0),
                        "video_duration": props.get("video_duration", 0),
                        "total_video_frames": props.get("total_video_frames", 0),
                        "extraction_date": extraction_date,
                    }
                    writer.writerow(row)

                    total_rows += 1
                    category_counts[category] += 1
                    if label == "real":
                        real_rows += 1
                    else:
                        fake_rows += 1

        logger.info(f"Metadata rebuilt and saved to {csv_path}")
        logger.info(f"Total frames indexed: {total_rows}")
        logger.info(f"Real frames: {real_rows}")
        logger.info(f"Fake frames: {fake_rows}")
        logger.info("\nFrames by category:")
        for category in sorted(category_counts.keys()):
            logger.info(f"  {category}: {category_counts[category]} frames")
        
    def extract_frames_from_video(
        self,
        video_path: Path,
        category: str,
        output_dir: Path
    ) -> List[Dict]:
        """
        Extract frames from a single video (optimized with frame seeking)
        
        Args:
            video_path: Path to video file
            category: Video category (original, Deepfakes, etc.)
            output_dir: Directory to save frames
            
        Returns:
            List of metadata records for extracted frames
        """
        return process_video_standalone(
            str(video_path),
            category,
            str(output_dir),
            str(self.output_path),
            self.target_fps,
            self.jpeg_quality,
            self.categories
        )
    
    def process_category(self, category: str) -> List[Dict]:
        """
        Process all videos in a category with multiprocessing
        
        Args:
            category: Category name (original, Deepfakes, etc.)
            
        Returns:
            List of all metadata records
        """
        category_path = self.dataset_path / category
        
        if not category_path.exists():
            logger.warning(f"Category path does not exist: {category_path}")
            return []
        
        # Create output directory for this category
        output_dir = self.output_path / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all video files
        video_files = list(category_path.glob("*.mp4"))
        
        # Limit for testing if configured
        if settings.MAX_VIDEOS_PER_CATEGORY:
            video_files = video_files[:settings.MAX_VIDEOS_PER_CATEGORY]
            logger.info(f"Testing mode: Limited to {len(video_files)} videos")
        
        if not video_files:
            logger.warning(f"No videos found in {category_path}")
            return []
        
        logger.info(f"Processing {len(video_files)} videos from {category} with {self.num_workers} workers")
        
        all_records = []
        
        if self.num_workers > 1:
            # OPTIMIZATION: ProcessPoolExecutor for Windows compatibility
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_video = {
                    executor.submit(
                        process_video_standalone,
                        str(video_path),
                        category,
                        str(output_dir),
                        str(self.output_path),
                        self.target_fps,
                        self.jpeg_quality,
                        self.categories
                    ): video_path
                    for video_path in video_files
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(video_files), desc=f"Processing {category}") as pbar:
                    for future in as_completed(future_to_video):
                        try:
                            records = future.result()
                            all_records.extend(records)
                        except Exception as e:
                            video_path = future_to_video[future]
                            logger.error(f"Error processing {video_path}: {str(e)}")
                        pbar.update(1)
        else:
            # Single-threaded processing
            for video_path in tqdm(video_files, desc=f"Processing {category}"):
                records = self.extract_frames_from_video(
                    video_path,
                    category,
                    output_dir
                )
                all_records.extend(records)
        
        return all_records
    
    def save_metadata(self, all_records: List[Dict]) -> None:
        """
        Save metadata to CSV file
        
        Args:
            all_records: List of all metadata records from all videos
        """
        if not all_records:
            logger.warning("No metadata to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        
        # Save to CSV
        csv_path = self.output_path / self.metadata_csv
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Metadata saved to {csv_path}")
        logger.info(f"Total frames extracted: {len(df)}")
        logger.info(f"Real frames: {len(df[df['label'] == 'real'])}")
        logger.info(f"Fake frames: {len(df[df['label'] == 'fake'])}")
        
        # Print summary by category
        logger.info("\nFrames by category:")
        for category in sorted(df['category'].unique()):
            count = len(df[df['category'] == category])
            logger.info(f"  {category}: {count} frames")
    
    def extract_all(self) -> None:
        """Extract frames from all categories (optimized)"""
        logger.info("=" * 60)
        logger.info("Starting OPTIMIZED frame extraction process")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Output path: {self.output_path}")
        logger.info(f"Target FPS: {self.target_fps}")
        logger.info(f"JPEG Quality: {self.jpeg_quality}")
        logger.info(f"Parallel Workers: {self.num_workers}")
        logger.info("=" * 60)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Store all records
        all_records = []
        
        # Process each category
        for category in self.categories.keys():
            records = self.process_category(category)
            all_records.extend(records)
            logger.info(f"Completed {category}: {len(records)} frames extracted")
        
        # Save metadata
        self.save_metadata(all_records)
        
        logger.info("=" * 60)
        logger.info("Frame extraction completed!")
        logger.info("=" * 60)


def main():
    """Main function to run frame extraction"""
    parser = argparse.ArgumentParser(description="Frame extractor and metadata builder")
    parser.add_argument(
        "--rebuild-csv-only",
        action="store_true",
        help="Rebuild metadata CSV from existing extracted frames without extracting again",
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create extractor with default settings from configs
    extractor = FrameExtractor()

    if args.rebuild_csv_only:
        extractor.rebuild_metadata_from_frames()
    else:
        # Extract frames
        extractor.extract_all()


if __name__ == "__main__":
    main()
