import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_DIR = "training_data/"
OUTPUT_DIR = "final_images/"
LABELS_FILE = "plots.csv"
LOG_FILE = "plot_classification_log.json"
MODEL_SAVE_PATH = "plot_classifier_model.pth"

# --- Hyperparameters ---
CONFIDENCE_THRESHOLD = 0.70
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 16
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2

class PlotDataset(Dataset):
    """Custom dataset for loading labeled images."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            blank = torch.zeros(3, IMG_SIZE, IMG_SIZE)  # Return a blank image and label 0 as fallback
            return blank, 0


class PlotBinaryClassifier:
    """
    Binary classifier trained on labeled data to detect plots vs diagrams.
    Uses transfer learning with EfficientNet-B0.
    """
    
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device.upper()}")
        
        # Image preprocessing - with augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def build_model(self):
        """Build and initialize the model."""
        print("Building EfficientNet-B0 model...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze early layers, fine-tune later layers
        for param in model.features[:5].parameters():
            param.requires_grad = False
        
        # Replace classifier for binary classification
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)  # Binary: [diagram, plot]
        )
        
        self.model = model.to(self.device)
        print("Model built successfully!")
        return self.model
    
    def train_model(self, train_loader, val_loader):
        """Train the model on labeled data."""
        if self.model is None:
            self.build_model()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        print(f"\nStarting training for {NUM_EPOCHS} epochs...")
        print("=" * 60)
        
        for epoch in range(NUM_EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update LR
            scheduler.step(avg_val_loss)
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"  ✓ Model saved (best validation loss)")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Load model from local if already present
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("\n" + "=" * 60)
        print("Training complete! Best model loaded.")
        
        # Print final validation metrics
        print("\nFinal Validation Metrics:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=['Diagram', 'Plot'],
                                   digits=3))
        
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print("                Predicted")
        print("              Diagram  Plot")
        print(f"Actual Diagram   {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"       Plot      {cm[1][0]:4d}    {cm[1][1]:4d}")
        
        return self.training_history
    
    def classify_image(self, image_path):
        """
        Classify a single image as plot or diagram.
        
        Returns:
            dict: {
                'is_plot': bool,
                'confidence': float,
                'plot_probability': float,
                'diagram_probability': float,
                'decision_reason': str
            }
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded!")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
            
            # Classify
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                diagram_prob = probabilities[0][0].item()
                plot_prob = probabilities[0][1].item()
            
            # Decision logic with confidence threshold
            confidence = max(diagram_prob, plot_prob)
            is_plot = plot_prob > CONFIDENCE_THRESHOLD
            
            if is_plot:
                decision_reason = f"Classified as PLOT with {plot_prob:.1%} confidence"
            else:
                decision_reason = f"Classified as DIAGRAM with {diagram_prob:.1%} confidence"
            
            return {
                'is_plot': is_plot,
                'confidence': confidence,
                'plot_probability': plot_prob,
                'diagram_probability': diagram_prob,
                'decision_reason': decision_reason
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                'is_plot': False,
                'confidence': 0.0,
                'plot_probability': 0.0,
                'diagram_probability': 0.0,
                'decision_reason': f'Error: {str(e)}'
            }
    
    def classify_directory(self, input_dir, output_dir, labeled_files=None):
        """
        Classify all images in directory and copy diagrams to output.
        Skips images that were used for training.
        
        Args:
            labeled_files: Set of filenames used for training (to skip)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg'}
        all_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        # Filter out labeled files if provided
        if labeled_files:
            image_files = [f for f in all_files if f not in labeled_files]
            print(f"\nFound {len(all_files)} total images")
            print(f"Skipping {len(labeled_files)} labeled images used for training")
            print(f"Classifying {len(image_files)} unlabeled images")
        else:
            image_files = all_files
            print(f"\nFound {len(image_files)} images to classify")
        
        if not image_files:
            print("No images to classify!")
            return {}
        
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print("=" * 60)
        
        results = []
        kept_count = 0
        plot_count = 0
        
        for filename in tqdm(image_files, desc="Classifying images"):
            input_path = os.path.join(input_dir, filename)
            
            # Classify
            result = self.classify_image(input_path)
            result['filename'] = filename
            results.append(result)
            
            # Keep or reject
            if not result['is_plot']:
                output_path = os.path.join(output_dir, filename)
                shutil.copy2(input_path, output_path)
                kept_count += 1
            else:
                plot_count += 1
                print(f"\n✗ REJECTED: {filename}")
                print(f"  {result['decision_reason']}")
        
        # Save log
        with open(LOG_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        stats = {
            'total_classified': len(image_files),
            'diagrams_kept': kept_count,
            'plots_filtered': plot_count,
            'filter_rate': (plot_count / len(image_files) * 100) if image_files else 0
        }
        
        return stats


def load_labeled_data(labels_file, input_dir):
    """Load labeled data from CSV file."""
    print(f"Loading labels from {labels_file}...")
    
    df = pd.read_csv(labels_file, header=None, names=['filename', 'isPlot'])
    
    # Build full paths and filter existing files
    image_paths = []
    labels = []
    missing_files = []
    
    for _, row in df.iterrows():
        filepath = os.path.join(input_dir, row['filename'])
        if os.path.exists(filepath):
            image_paths.append(filepath)
            labels.append(int(row['isPlot']))
        else:
            missing_files.append(row['filename'])
    
    if missing_files:
        print(f"WARNING: {len(missing_files)} labeled files not found in {input_dir}")
    
    print(f"Loaded {len(image_paths)} labeled images")
    print(f"  Diagrams (0): {labels.count(0)}")
    print(f"  Plots (1): {labels.count(1)}")
    
    return image_paths, labels, set(df['filename'].tolist())


def main():
    print("=" * 60)
    print("Binary Plot Classifier - Training Mode")
    print("Fine-tuning on labeled data")
    print("=" * 60)
    
    # Check for required files
    if not os.path.exists(LABELS_FILE):
        print(f"\nERROR: Labels file '{LABELS_FILE}' not found!")
        return
    
    if not os.path.isdir(INPUT_DIR):
        print(f"\nERROR: Input directory '{INPUT_DIR}' not found!")
        return
    
    # Load labeled data
    image_paths, labels, labeled_filenames = load_labeled_data(LABELS_FILE, INPUT_DIR)
    
    if len(image_paths) < 10:
        print("\nERROR: Need at least 10 labeled examples to train!")
        return
    
    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=VALIDATION_SPLIT,
        stratify=labels,
        random_state=42
    )
    
    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    classifier = PlotBinaryClassifier()
    
    # Create datasets and dataloaders
    train_dataset = PlotDataset(train_paths, train_labels, classifier.train_transform)
    val_dataset = PlotDataset(val_paths, val_labels, classifier.val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Train model
    history = classifier.train_model(train_loader, val_loader)
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Classify remaining unlabeled images
    print("\n" + "=" * 60)
    print("Classifying unlabeled images...")
    print("=" * 60)
    
    stats = classifier.classify_directory(INPUT_DIR, OUTPUT_DIR, labeled_filenames)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"Unlabeled images classified:  {stats['total_classified']}")
    print(f"Diagrams kept:                {stats['diagrams_kept']}")
    print(f"Plots filtered:               {stats['plots_filtered']}")
    print(f"Filter rate:                  {stats['filter_rate']:.1f}%")
    print(f"\nKept diagrams saved to:       {OUTPUT_DIR}")
    print(f"Classification log:           {LOG_FILE}")
    print(f"Trained model saved:          {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()