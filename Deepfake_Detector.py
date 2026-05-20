"""
CCTV Deepfake Detection Pipeline
Subsystem 2: Detect deepfakes in CCTV footage using ResNext-50 + BiLSTM
"""

import os
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import glob
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, accuracy_score, 
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CCTVDeepfakeDetector:
    """
    Complete pipeline for detecting CCTV deepfakes using ResNext-50 + BiLSTM
    """
    
    def __init__(self, num_frames: int = 40, input_shape: Tuple = (256, 256, 3)):
        self.num_frames = num_frames
        self.input_shape = input_shape
        self.model = None
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self):
        """Create CCTV-specific augmentation pipeline"""
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Lighting variations (critical for CCTV)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            
            # Color transformations
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.5
            ),
            A.ToGray(p=0.3),
            
            # Noise and blur (CCTV artifacts)
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.3
            ),
            
            # Compression artifacts
            A.ImageCompression(
                quality_lower=60,
                quality_upper=90,
                compression_type=A.ImageCompression.ImageCompressionType.JPEG,
                p=0.5
            ),
            
            # Resolution degradation
            A.Downscale(
                scale_min=0.5,
                scale_max=0.9,
                interpolation=cv2.INTER_LINEAR,
                p=0.2
            ),
            
            # Advanced degradations
            A.MotionBlur(blur_limit=7, p=0.2),
            A.MedianBlur(blur_limit=5, p=0.1),
            A.Posterize(num_bits=4, p=0.1),
        ], p=1.0)
    
    def build_model(self, freeze_base_layers: bool = True) -> Model:
        """
        Build ResNext-50 + BiLSTM detector for CCTV deepfakes
        
        Args:
            freeze_base_layers: Whether to freeze lower CNN layers
            
        Returns:
            model: Complete Keras model
        """
        logger.info("Building CCTV Deepfake Detector model...")
        
        # Base CNN for spatial features (ResNext-50)
        base_model = tf.keras.applications.ResNeXt50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze lower layers, train upper 30 layers
        if freeze_base_layers:
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            for layer in base_model.layers[-30:]:
                layer.trainable = True
        
        # Input for video sequences
        video_input = layers.Input(
            shape=(self.num_frames, *self.input_shape),
            name='video_input'
        )
        
        # TimeDistributed wrapper for frame-by-frame processing
        frame_features = layers.TimeDistributed(
            base_model,
            name='spatial_features'
        )(video_input)
        
        # Temporal modeling with BiLSTM
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                256,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3,
                name='lstm_1'
            ),
            name='bilstm_1'
        )(frame_features)
        
        lstm_out = layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3,
                name='lstm_2'
            ),
            name='bilstm_2'
        )(lstm_out)
        
        # Attention mechanism
        attention = layers.Attention(
            name='attention'
        )([lstm_out, lstm_out])
        
        # Global average pooling over time dimension
        attention_pooled = layers.GlobalAveragePooling1D(
            name='temporal_pooling'
        )(attention)
        
        # Classification head
        dense1 = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name='dense_1'
        )(attention_pooled)
        dropout1 = layers.Dropout(0.5, name='dropout_1')(dense1)
        
        dense2 = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            name='dense_2'
        )(dropout1)
        dropout2 = layers.Dropout(0.5, name='dropout_2')(dense2)
        
        # Output layer
        output = layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(dropout2)
        
        # Build model
        model = Model(
            inputs=video_input,
            outputs=output,
            name='CCTV_DeepFake_Detector'
        )
        
        self.model = model
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def prepare_training_data(self, real_videos_dir: str, fake_videos_dir: str,
                             test_size: float = 0.15, val_size: float = 0.15):
        """
        Prepare train/val/test splits with balanced classes
        
        Args:
            real_videos_dir: Directory containing real CCTV videos
            fake_videos_dir: Directory containing generated deepfakes
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            train_data, val_data, test_data: Lists of (video_path, label) tuples
        """
        logger.info("Preparing training data...")
        
        # Collect video paths
        real_videos = glob.glob(os.path.join(real_videos_dir, "*.mp4"))
        fake_videos = glob.glob(os.path.join(fake_videos_dir, "*.mp4"))
        
        logger.info(f"Found {len(real_videos)} real videos")
        logger.info(f"Found {len(fake_videos)} fake videos")
        
        # Create balanced dataset
        dataset = []
        for video_path in real_videos:
            dataset.append((video_path, 0))  # 0 = real
        for video_path in fake_videos:
            dataset.append((video_path, 1))  # 1 = fake
        
        # Shuffle dataset
        random.seed(42)
        random.shuffle(dataset)
        
        # Split: 70% train, 15% val, 15% test
        train_data, temp_data = train_test_split(
            dataset,
            test_size=(test_size + val_size),
            stratify=[label for _, label in dataset],
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(test_size / (test_size + val_size)),
            stratify=[label for _, label in temp_data],
            random_state=42
        )
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def create_data_generator(self, data_list: List, batch_size: int = 8,
                             augment: bool = False, shuffle: bool = True):
        """
        Create data generator for video sequences
        
        Args:
            data_list: List of (video_path, label) tuples
            batch_size: Batch size
            augment: Apply augmentation
            shuffle: Shuffle data
            
        Returns:
            generator: VideoDataGenerator instance
        """
        return VideoDataGenerator(
            data_list=data_list,
            batch_size=batch_size,
            num_frames=self.num_frames,
            input_shape=self.input_shape,
            augment=augment,
            augmentation_pipeline=self.augmentation_pipeline if augment else None,
            shuffle=shuffle
        )
    
    def train(self, train_data: List, val_data: List, 
              epochs: int = 25, batch_size: int = 8,
              checkpoint_dir: str = 'checkpoints'):
        """
        Train the deepfake detection model
        
        Args:
            train_data: Training data list
            val_data: Validation data list
            epochs: Number of training epochs
            batch_size: Batch size
            checkpoint_dir: Directory to save checkpoints
        """
        logger.info("Starting model training...")
        
        # Create data generators
        train_generator = self.create_data_generator(
            train_data, batch_size=batch_size, augment=True, shuffle=True
        )
        val_generator = self.create_data_generator(
            val_data, batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            TensorBoard(
                log_dir='logs',
                histogram_freq=1
            ),
            CSVLogger('training_log.csv', append=True)
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete!")
        return history
    
    def fine_tune(self, train_data: List, val_data: List,
                  epochs: int = 10, batch_size: int = 8):
        """
        Fine-tune upper layers with lower learning rate
        
        Args:
            train_data: Training data list
            val_data: Validation data list
            epochs: Number of fine-tuning epochs
            batch_size: Batch size
        """
        logger.info("Starting fine-tuning...")
        
        # Unfreeze all LSTM and Dense layers
        for layer in self.model.layers:
            if any(name in layer.name for name in ['lstm', 'bilstm', 'dense', 'attention']):
                layer.trainable = True
        
        # Unfreeze top 50 layers of base CNN
        base_model = self.model.layers[1].layer  # TimeDistributed base model
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Create data generators
        train_generator = self.create_data_generator(
            train_data, batch_size=batch_size, augment=True, shuffle=True
        )
        val_generator = self.create_data_generator(
            val_data, batch_size=batch_size, augment=False, shuffle=False
        )
        
        # Fine-tune
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                ModelCheckpoint(
                    filepath='checkpoints/finetuned_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                )
            ],
            verbose=1
        )
        
        logger.info("Fine-tuning complete!")
        return history
    
    def evaluate(self, test_data: List, output_dir: str = 'evaluation'):
        """
        Comprehensive evaluation with metrics and visualizations
        
        Args:
            test_data: Test data list
            output_dir: Directory to save evaluation results
        """
        logger.info("Starting comprehensive evaluation...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test generator
        test_generator = self.create_data_generator(
            test_data, batch_size=8, augment=False, shuffle=False
        )
        
        # Predict
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = np.array([label for _, label in test_data])
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=['Real', 'Fake'],
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc,
            'true_positives': int(cm[1, 1]),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0])
        }
        
        print("\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation complete! Results saved to {output_dir}")
        return metrics
    
    def predict_video(self, video_path: str) -> Tuple[float, str]:
        """
        Predict if a single video is real or fake
        
        Args:
            video_path: Path to video file
            
        Returns:
            confidence: Prediction confidence (0-1)
            label: 'Real' or 'Fake'
        """
        # Extract frames
        frames = self._extract_frames_from_video(video_path)
        
        # Normalize
        frames = frames.astype(np.float32) / 255.0
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        
        # Predict
        pred_proba = self.model.predict(frames, verbose=0)[0][0]
        
        label = 'Fake' if pred_proba > 0.5 else 'Real'
        confidence = pred_proba if pred_proba > 0.5 else 1 - pred_proba
        
        return confidence, label
    
    def _extract_frames_from_video(self, video_path: str) -> np.ndarray:
        """Extract frames from video for prediction"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.input_shape[:2])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return np.array(frames)


class VideoDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for video sequences"""
    
    def __init__(self, data_list, batch_size=8, num_frames=40,
                 input_shape=(256, 256, 3), augment=False,
                 augmentation_pipeline=None, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.input_shape = input_shape
        self.augment = augment
        self.augmentation_pipeline = augmentation_pipeline
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_list))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.data_list) // self.batch_size
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self._generate_batch(batch_indexes)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _generate_batch(self, batch_indexes):
        X = np.zeros((self.batch_size, self.num_frames, *self.input_shape), dtype=np.float32)
        y = np.zeros(self.batch_size, dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            video_path, label = self.data_list[idx]
            frames = self._extract_frames(video_path)
            
            if self.augment and self.augmentation_pipeline:
                frames = self._augment_frames(frames)
            
            frames = frames.astype(np.float32) / 255.0
            X[i] = frames
            y[i] = label
        
        return X, y
    
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.input_shape[:2])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def _augment_frames(self, frames):
        augmented = []
        for frame in frames:
            aug_frame = self.augmentation_pipeline(image=frame)['image']
            augmented.append(aug_frame)
        return np.array(augmented)


def main():
    """Example usage of CCTV Deepfake Detector"""
    
    # Initialize detector
    detector = CCTVDeepfakeDetector(num_frames=40, input_shape=(256, 256, 3))
    
    # Build model
    model = detector.build_model(freeze_base_layers=True)
    model.summary()
    
    # Prepare data
    train_data, val_data, test_data = detector.prepare_training_data(
        real_videos_dir='data/real_videos',
        fake_videos_dir='data/fake_videos'
    )
    
    # Train model
    history = detector.train(
        train_data=train_data,
        val_data=val_data,
        epochs=25,
        batch_size=8
    )
    
    # Fine-tune
    history_ft = detector.fine_tune(
        train_data=train_data,
        val_data=val_data,
        epochs=10,
        batch_size=8
    )
    
    # Evaluate
    metrics = detector.evaluate(test_data=test_data, output_dir='evaluation_results')
    
    # Save model
    detector.model.save('checkpoints/final_model.h5')
    logger.info("Model saved successfully!")
    
    # Example prediction
    confidence, label = detector.predict_video('test_video.mp4')
    print(f"\nPrediction: {label} (Confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
