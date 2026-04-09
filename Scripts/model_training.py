# ============================================================
# FILE: model_training_v2.py
# PURPOSE: Fixed Overfitting Version
# CHANGES:
#   1. Increased Dropout (0.4 → 0.5, 0.3 → 0.4)
#   2. Lower Learning Rate (0.001 → 0.0003)
#   3. More layers unfrozen for better fine-tuning
#   4. Added Label Smoothing to loss function
#   5. Added Mixup Augmentation
#   6. Stronger image augmentation in transforms
#   7. Higher weight decay (0.01 → 0.05)
#   8. Gradient Clipping added
# ============================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

DATA_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\data\spectrograms"

MODEL_SAVE_PATH = r"C:\Users\srira\OneDrive\Desktop\Anonymous Files\Cough Type classification Project\Cough Type classification Project\models"

# ============================================================
# HYPERPARAMETERS — FIXED FOR OVERFITTING
# ============================================================

IMAGE_SIZE    = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 0.0003    # ✅ FIX 1: Lowered from 0.001
NUM_CLASSES   = 3
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
PATIENCE      = 15        # ✅ More patience for stable training

# ============================================================
# ✅ FIX 2: STRONGER DATA AUGMENTATION IN TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),        # ✅ NEW
    transforms.RandomRotation(degrees=15),         # ✅ Increased
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,                            # ✅ NEW
        hue=0.1                                    # ✅ NEW
    ),
    transforms.RandomGrayscale(p=0.1),             # ✅ NEW
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)                       # ✅ NEW - slight shift
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.2)                # ✅ NEW - random erase
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# ✅ FIX 3: MIXUP AUGMENTATION FUNCTION
# Blends two images and their labels to reduce overfitting
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup: blend two samples together"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    # Verify GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*55)
    print("       🖥️  HARDWARE CONFIGURATION")
    print("="*55)
    print(f"  Device     : {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name   : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*55)

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # -----------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------
    print("\n📂 Loading Dataset...")
    full_dataset = datasets.ImageFolder(root=DATA_PATH)
    class_names  = full_dataset.classes
    total_size   = len(full_dataset)

    print(f"   Classes Found : {class_names}")
    print(f"   Total Images  : {total_size}")

    # Split sizes
    train_size = int(TRAIN_SPLIT * total_size)
    val_size   = int(VAL_SPLIT   * total_size)
    test_size  = total_size - train_size - val_size

    print(f"\n   📊 Dataset Split:")
    print(f"      🟢 Train : {train_size}")
    print(f"      🟡 Val   : {val_size}")
    print(f"      🔴 Test  : {test_size}")

    from torch.utils.data import random_split

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform   = val_test_transforms
    test_dataset.dataset.transform  = val_test_transforms

    # -----------------------------------------------
    # HANDLE CLASS IMBALANCE
    # -----------------------------------------------
    class_counts = np.array([
        sum(1 for _, label in full_dataset if label == i)
        for i in range(NUM_CLASSES)
    ])

    print(f"\n   📊 Class Distribution:")
    for i, cls in enumerate(class_names):
        print(f"      {cls:10s} : {class_counts[i]}")

    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[full_dataset[i][1]]
                      for i in train_dataset.indices]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\n✅ Data Loaders Created!")

    # -----------------------------------------------
    # ✅ FIX 4: BUILD EFFICIENTNET-B3 WITH MORE DROPOUT
    # -----------------------------------------------
    print("\n🏗️  Building EfficientNet-B3 Model (Anti-Overfitting)...")

    model = models.efficientnet_b3(pretrained=True)

    # Freeze ALL base layers first
    for param in model.parameters():
        param.requires_grad = False

    # ✅ FIX 5: Replace classifier with higher dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),                          # ✅ Increased from 0.4
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),                         # ✅ NEW: BatchNorm
        nn.Dropout(p=0.4),                           # ✅ Increased from 0.3
        nn.Linear(512, 256),                         # ✅ NEW: Extra layer
        nn.ReLU(),
        nn.Dropout(p=0.3),                           # ✅ NEW
        nn.Linear(256, NUM_CLASSES)
    )

    # ✅ FIX 6: Unfreeze more layers for better fine-tuning
    for param in model.features[-5:].parameters():  # ✅ More layers unfrozen
        param.requires_grad = True

    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    print(f"✅ Model Built!")
    print(f"   Total Parameters     : {total_params:,}")
    print(f"   Trainable Parameters : {trainable_params:,}")

    # -----------------------------------------------
    # ✅ FIX 7: LABEL SMOOTHING LOSS
    # Prevents model from being overconfident
    # -----------------------------------------------
    weights   = torch.FloatTensor(
        class_weights / class_weights.sum()
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1    # ✅ NEW: Label smoothing
    )

    # ✅ FIX 8: Higher weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.05      # ✅ Increased from 0.01
    )

    # Cosine Annealing scheduler (smoother than ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6           # ✅ Minimum learning rate
    )

    print("\n✅ Loss, Optimizer & Scheduler Defined!")
    print("   Label Smoothing  : 0.1")
    print("   Weight Decay     : 0.05")
    print("   LR Scheduler     : CosineAnnealing")

    # -----------------------------------------------
    # TRAINING LOOP WITH MIXUP
    # -----------------------------------------------
    print("\n🚀 Starting Training (Anti-Overfitting Mode)...\n")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   []
    }

    best_val_acc     = 0.0
    patience_counter = 0
    best_model_path  = os.path.join(MODEL_SAVE_PATH, 'best_model_v2.pth')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Training phase with Mixup
        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in tqdm(train_loader,
                                    desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
                                    leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # ✅ Apply Mixup augmentation
            images, labels_a, labels_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            outputs = model(images)

            # ✅ Mixup loss
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()

            # ✅ FIX 9: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels_a.size(0)
            train_correct += (
                lam * predicted.eq(labels_a).sum().item()
                + (1 - lam) * predicted.eq(labels_b).sum().item()
            )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                        desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]",
                                        leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = 100.0 * val_correct / val_total

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Train Acc: {avg_train_acc:.2f}% "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Acc: {avg_val_acc:.2f}% "
              f"| LR: {current_lr:.6f} "
              f"| Time: {elapsed:.1f}s")

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc':         best_val_acc,
                'class_names':     class_names
            }, best_model_path)
            print(f"   💾 Best Model Saved! Val Acc: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early Stopping at epoch {epoch+1}")
            break

    # -----------------------------------------------
    # PLOT TRAINING HISTORY
    # -----------------------------------------------
    print("\n📊 Plotting Training History...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'],   label='Val Loss',   color='red')
    ax1.set_title('Training & Validation Loss (v2)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'],   label='Val Accuracy',   color='red')
    ax2.set_title('Training & Validation Accuracy (v2)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_SAVE_PATH, 'training_history_v2.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"✅ Training plot saved!")

    # -----------------------------------------------
    # EVALUATE ON TEST SET
    # -----------------------------------------------
    print("\n🧪 Evaluating on Test Set...")

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images  = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # -----------------------------------------------
    # FINAL RESULTS & CONFUSION MATRIX
    # -----------------------------------------------
    print("\n" + "="*55)
    print("          📊 FINAL TEST RESULTS")
    print("="*55)
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names
    ))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Cough Classification v2')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_SAVE_PATH, 'confusion_matrix_v2.png')
    plt.savefig(cm_path)
    plt.show()

    # Final Summary
    print("\n" + "="*55)
    print("       ✅ TRAINING V2 COMPLETE!")
    print("="*55)
    print(f"  Best Validation Accuracy : {best_val_acc:.2f}%")
    print(f"  Best Model Saved At      : {best_model_path}")
    print(f"  Training Plot            : {plot_path}")
    print(f"  Confusion Matrix         : {cm_path}")
    print("="*55)
    print("\n🎯 Next Step: Inference & Deployment!")