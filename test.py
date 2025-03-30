import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --------------------------
# 1. Dataset Curation Module
# --------------------------

#   -------------------------------
#   DATA SET NAHI MIL RAHA !!!!!!!!!!!!!!
#
class BiomedicalDatasetCurator:
    """Class to curate and organize biomedical datasets for pre-training"""
    
    def __init__(self):
        self.datasets = []
        
    def add_dataset(self, name, description, source, data_type, size, url):
        """Add a dataset to the collection"""
        self.datasets.append({
            'name': name,
            'description': description,
            'source': source,
            'data_type': data_type,
            'size': size,
            'url': url
        })
    
    def get_curated_datasets(self):
        """Return curated datasets as a DataFrame"""
        return pd.DataFrame(self.datasets)
    
    def save_to_csv(self, filename='biomedical_datasets.csv'):
        """Save curated datasets to CSV"""
        df = self.get_curated_datasets()
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} datasets to {filename}")

# Example usage of the curator
def curate_biomedical_datasets():
    curator = BiomedicalDatasetCurator()
    
    # Add some example biomedical datasets (in a real scenario, this would be more comprehensive)
    curator.add_dataset(
        name="TCGA Pan-Cancer Atlas",
        description="Comprehensive multi-omics data for 33 cancer types",
        source="NCI Genomic Data Commons",
        data_type="RNA-seq, Clinical, Imaging",
        size="2.5TB",
        url="https://portal.gdc.cancer.gov/"
    )
    
    curator.add_dataset(
        name="Human Protein Atlas",
        description="Tissue and cell expression profiles for human proteins",
        source="Protein Atlas",
        data_type="Immunohistochemistry, RNA-seq",
        size="500GB",
        url="https://www.proteinatlas.org/"
    )
    
    curator.add_dataset(
        name="Single Cell RNA-seq Benchmarking",
        description="Benchmark datasets for single-cell RNA-seq analysis",
        source="Broad Institute",
        data_type="scRNA-seq",
        size="200GB",
        url="https://singlecell.broadinstitute.org/single_cell"
    )
    
    # Save the curated datasets
    curator.save_to_csv()
    return curator.get_curated_datasets()

# -------------------------------
# 2. Space Biology Dataset Module
# -------------------------------

class SpaceBiologyDatasetFinder:
    """Class to identify and organize space biology datasets"""
    
    def __init__(self):
        self.datasets = []
        
    def add_dataset(self, name, description, source, organism, data_type, url):
        """Add a space biology dataset to the collection"""
        self.datasets.append({
            'name': name,
            'description': description,
            'source': source,
            'organism': organism,
            'data_type': data_type,
            'url': url
        })
    
    def get_space_datasets(self):
        """Return space biology datasets as a DataFrame"""
        return pd.DataFrame(self.datasets)
    
    def save_to_csv(self, filename='space_biology_datasets.csv'):
        """Save space biology datasets to CSV"""
        df = self.get_space_datasets()
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} datasets to {filename}")

# Example usage of the space dataset finder
def identify_space_biology_datasets():
    finder = SpaceBiologyDatasetFinder()
    
    # Add some example space biology datasets from NASA OSDR
    finder.add_dataset(
        name="NASA Twins Study",
        description="Multi-omics analysis of twin astronauts (one in space, one on Earth)",
        source="NASA OSDR",
        organism="Human",
        data_type="Genomics, Proteomics, Metabolomics",
        url="https://osdr.nasa.gov/bio/repo/data/studies/OSD-425"
    )
    
    finder.add_dataset(
        name="Rodent Research-1",
        description="Mouse physiological response to spaceflight",
        source="NASA OSDR",
        organism="Mouse",
        data_type="RNA-seq, Histopathology",
        url="https://osdr.nasa.gov/bio/repo/data/studies/OSD-566"
    )
    
    finder.add_dataset(
        name="Plant Growth in Space",
        description="Arabidopsis thaliana growth under microgravity",
        source="NASA OSDR",
        organism="Arabidopsis",
        data_type="Imaging, Transcriptomics",
        url="https://osdr.nasa.gov/bio/repo/data/studies/OSD-123"
    )
    
    # Save the space biology datasets
    finder.save_to_csv()
    return finder.get_space_datasets()

# --------------------------
# 3. Transfer Learning Module
# --------------------------

# Custom dataset class for space biology data
class SpaceBiologyDataset(Dataset):
    """Custom PyTorch dataset for space biology data"""
    
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# Simple CNN model for demonstration
class SimpleBioModel(nn.Module):
    """Simple CNN model for biological data"""
    
    def __init__(self, num_classes):
        super(SimpleBioModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """Train a PyTorch model with transfer learning"""
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def transfer_learning_pipeline():
    """Demonstration of transfer learning pipeline for space biology"""
    
    print("=== Starting Transfer Learning Pipeline ===")
    
    # 1. Load a pre-trained model (ResNet in this case)
    print("Loading pre-trained model...")
    model = models.resnet18(pretrained=True)
    
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for our specific task
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification for demonstration
    
    # 2. Prepare synthetic space biology data (in a real scenario, use actual data)
    print("Preparing synthetic space biology data...")
    
    # Generate synthetic data (replace with actual space biology data loading)
    num_samples = 200
    img_size = 64
    X_train = np.random.rand(num_samples, 3, img_size, img_size).astype(np.float32)
    y_train = np.random.randint(0, 2, size=num_samples)
    X_val = np.random.rand(num_samples//2, 3, img_size, img_size).astype(np.float32)
    y_val = np.random.randint(0, 2, size=num_samples//2)
    
    # Create datasets and dataloaders
    train_dataset = SpaceBiologyDataset(X_train, y_train)
    val_dataset = SpaceBiologyDataset(X_val, y_val)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False)
    }
    
    # 3. Train the model
    print("Fine-tuning model on space biology data...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)
    
    print("Transfer learning pipeline completed!")
    return model

# --------------------------
# Main Execution
# --------------------------

def main():
    # 1. Dataset curation
    print("=== Biomedical Dataset Curation ===")
    biomedical_datasets = curate_biomedical_datasets()
    print(biomedical_datasets.head())
    
    # 2. Space biology dataset identification
    print("\n=== Space Biology Dataset Identification ===")
    space_datasets = identify_space_biology_datasets()
    print(space_datasets.head())
    
    # 3. Transfer learning demonstration
    print("\n=== Transfer Learning Demonstration ===")
    model = transfer_learning_pipeline()
    
    # Save the model for the Model Zoo
    torch.save(model.state_dict(), 'space_bio_model.pth')
    print("Model saved to space_bio_model.pth")

if __name__ == "__main__":
    main()