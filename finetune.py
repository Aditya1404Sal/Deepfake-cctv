# Fine-tune Script - Train EfficientNet on augmented deepfake dataset
# Unfreeze backbone layers and adapt to surveillance deepfakes

import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path.home() / 'Desktop' / 'finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATASET
# ============================================================================

class SurveillanceDeepfakeDataset(Dataset):
    """Custom dataset for surveillance deepfake detection."""
    
    def __init__(self, real_dir: str, fake_dir: str, transform=None):
        self.real_images = sorted(Path(real_dir).expanduser().glob('*.jpg')) + \
                          sorted(Path(real_dir).expanduser().glob('*.png'))
        self.fake_images = sorted(Path(fake_dir).expanduser().glob('*.jpg')) + \
                          sorted(Path(fake_dir).expanduser().glob('*.png'))
        self.transform = transform
        
        logger.info(f"Real images: {len(self.real_images)}")
        logger.info(f"Fake images: {len(self.fake_images)}")
    
    def __len__(self):
        return len(self.real_images) + len(self.fake_images)
    
    def __getitem__(self, idx):
        if idx < len(self.real_images):
            img_path = self.real_images[idx]
            label = 0  # Real
        else:
            img_path = self.fake_images[idx - len(self.real_images)]
            label = 1  # Fake
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# ============================================================================
# FINE-TUNER
# ============================================================================

class EfficientNetFineTuner:
    """Fine-tune pre-trained EfficientNet on surveillance deepfake dataset."""
    
    def __init__(self, checkpoint_path: str = None, device: str = 'cuda', learning_rate: float = 0.00005):
        """
        Initialize fine-tuner.
        
        Args:
            checkpoint_path: Path to pre-trained checkpoint (optional)
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for fine-tuning (typically lower than initial training)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained EfficientNet-B0
        logger.info("Loading pre-trained EfficientNet-B0...")
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Modify classification head for binary classification
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2)  # Binary: Real or Fake
        )
        
        self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Learning rate: {learning_rate}")
    
    def unfreeze_backbone(self, num_frozen_layers: int = 2):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            num_frozen_layers: How many layers from start to keep frozen (0 = unfreeze all)
        """
        logger.info(f"\nUnfreezing backbone (keeping first {num_frozen_layers} layers frozen)...")
        
        layers = list(self.model.features.children())
        logger.info(f"Total backbone layers: {len(layers)}")
        
        for i, layer in enumerate(layers):
            if i < num_frozen_layers:
                # Keep frozen
                for param in layer.parameters():
                    param.requires_grad = False
                logger.info(f"  Layer {i}: FROZEN")
            else:
                # Unfreeze
                for param in layer.parameters():
                    param.requires_grad = True
                if i < num_frozen_layers + 2:  # Log first unfrozen layers
                    logger.info(f"  Layer {i}: UNFROZEN")
        
        # Always unfreeze classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        logger.info(f"  Classifier: UNFROZEN")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
        logger.info(f"Percentage trainable: {100*trainable_params/total_params:.1f}%")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 15, save_dir: str = '~/Desktop/checkpoints'):
        """
        Full training loop with validation.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs: Number of epochs
            save_dir: Where to save checkpoints
        """
        save_dir = Path(save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "="*70)
        logger.info("STARTING FINE-TUNING")
        logger.info("="*70)
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 70)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = save_dir / f'best_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"✓ Best model saved: {checkpoint_path.name}")
            
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"✓ Checkpoint saved: {checkpoint_path.name}")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"Checkpoints saved to: {save_dir}")
        logger.info("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SURVEILLANCE DEEPFAKE DETECTION - FINE-TUNING")
    print("="*70 + "\n")
    
    # Configuration
    real_dir = '~/Desktop/augmented_frames/real'
    fake_dir = '~/Desktop/augmented_frames/fake'
    checkpoint_dir = '~/Desktop/checkpoints'
    batch_size = 16
    num_epochs = 15
    learning_rate = 0.00005
    frozen_layers = 2
    
    logger.info(f"Real images: {real_dir}")
    logger.info(f"Fake images: {fake_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Frozen layers: {frozen_layers}")
    
    # Create dataset
    logger.info("\n" + "="*70)
    logger.info("LOADING DATASET")
    logger.info("="*70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SurveillanceDeepfakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        transform=transform
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training samples: {train_size:,}")
    logger.info(f"Validation samples: {val_size:,}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize fine-tuner
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING MODEL")
    logger.info("="*70)
    
    finetuner = EfficientNetFineTuner(
        checkpoint_path=None,  # Set to path if you have a pre-trained model
        device='cuda',
        learning_rate=learning_rate
    )
    
    # Unfreeze backbone
    finetuner.unfreeze_backbone(num_frozen_layers=frozen_layers)
    
    # Train
    finetuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=checkpoint_dir
    )
    
    logger.info("Fine-tuning complete!")
