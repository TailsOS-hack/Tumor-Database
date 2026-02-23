import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
import argparse

# Add the current directory to path to allow imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data
from src.gatekeeper_model import GatekeeperClassifier

def train_model(num_epochs=15, batch_size=32, learning_rate=0.05, smoke_test=False):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader = load_data(batch_size=batch_size)

    # Initialize Model
    model = GatekeeperClassifier(freeze_base=True)
    model = model.to(device)

    # Loss and Optimizer
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.base_model.fc.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    save_path = os.path.join('models', 'gatekeeper_classifier.pt')
    os.makedirs('models', exist_ok=True)

    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        
        for i, (images, labels) in enumerate(loop):
            if smoke_test and i >= 10:
                print("Smoke test limit reached. Breaking epoch.")
                break
                
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1) # Match shape (Batch, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate training accuracy for this batch
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        # If smoke test, adjust denominator
        denom = 10 if smoke_test else len(train_loader)
        avg_train_loss = running_loss / denom if denom > 0 else 0
        train_acc = 100 * correct / total if total > 0 else 0

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        # For sensitivity/recall tracking (Class 0 = Tumor, Class 1 = Dementia)
        # Confusion Matrix elements
        tp = 0 # Predicted 1 (Dementia), Actual 1
        tn = 0 # Predicted 0 (Tumor), Actual 0
        fp = 0 # Predicted 1, Actual 0
        fn = 0 # Predicted 0, Actual 1
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if smoke_test and i >= 5:
                    break
                    
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update confusion matrix stats
                tp += ((predicted == 1) & (labels == 1)).sum().item()
                tn += ((predicted == 0) & (labels == 0)).sum().item()
                fp += ((predicted == 1) & (labels == 0)).sum().item()
                fn += ((predicted == 0) & (labels == 1)).sum().item()

        val_denom = 5 if smoke_test else len(val_loader)
        avg_val_loss = val_loss / val_denom if val_denom > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Calculate specialized metrics
        tumor_recall = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
        dementia_recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0

        print(f"Epoch [{epoch+1}/{num_epochs}] completed.")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Tumor Recall (Specificity): {tumor_recall:.2f}% | Dementia Recall (Sensitivity): {dementia_recall:.2f}%")

        # Save best model
        if val_acc >= best_val_acc or smoke_test:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        print("-" * 30)

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test')
    args = parser.parse_args()
    
    train_model(num_epochs=args.epochs, smoke_test=args.smoke_test)
