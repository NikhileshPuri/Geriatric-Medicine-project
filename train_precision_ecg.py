import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------- Dataset & Model --------
class InMemoryECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('int64')
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx]
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class Tiny1DCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.35),  # increased dropout for precision
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# -------- Helpers --------
def load_and_merge(data_csv, label_csv):
    print("Loading ECG data CSV...")
    df_data = pd.read_csv(data_csv)
    print("Loading labels CSV...")
    df_label = pd.read_csv(label_csv)
    if df_label.shape[1] == 1:
        df_label.columns = ['label']
    elif df_label.shape[1] > 1:
        df_label = df_label.iloc[:, 0:1]
        df_label.columns = ['label']
    print("Merging CSVs in memory...")
    df_full = pd.concat([df_label, df_data], axis=1)
    X = df_full.iloc[:, 1:].fillna(0).astype(float).values.astype(np.float32)
    raw_labels = df_full['label'].astype(str).str.strip().values
    y = np.array([1 if v=='A' else 0 for v in raw_labels], dtype=np.int64)
    print("Label counts (binary):", pd.Series(y).value_counts().to_dict())
    return X, y

def create_weighted_sampler(y_train, oversample_factor=3):
    # sample only minority class multiple times
    class_counts = np.bincount(y_train)
    weights = np.ones_like(y_train, dtype=np.float32)
    # AF indices = 1
    af_indices = np.where(y_train==1)[0]
    weights[af_indices] = oversample_factor
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler

def train_loop(model, loader, loss_fn, opt, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_all(model, loader, device, af_threshold=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()  # AF probability
            if af_threshold is None:
                p = out.argmax(dim=1).cpu().numpy()
            else:
                p = (probs > af_threshold).astype(int)
            preds.extend(p)
            trues.extend(yb.numpy())
    return np.array(trues), np.array(preds)

# -------- Main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--out', type=str, default='out')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--af_threshold', type=float, default=None, help='Optional AF probability threshold (0–1)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y = load_and_merge(args.data_csv, args.label_csv)

    # stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=43, stratify=y_temp)

    train_ds = InMemoryECGDataset(X_train, y_train)
    val_ds = InMemoryECGDataset(X_val, y_val)
    test_ds = InMemoryECGDataset(X_test, y_test)

    sampler = create_weighted_sampler(y_train, oversample_factor=3)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch*2, shuffle=False, num_workers=0)

    device = args.device
    model = Tiny1DCNN(n_classes=2).to(device)

    # weighted loss with reduced AF weight
    counts = np.bincount(y_train)
    class_weights = torch.tensor([1.0, 3.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for ep in range(args.epochs):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        yv, pv = eval_all(model, val_loader, device, af_threshold=args.af_threshold)
        val_acc = accuracy_score(yv, pv)
        print(f"Epoch {ep+1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out,'model.pt'))

    # final test
    model.load_state_dict(torch.load(os.path.join(args.out,'model.pt'), map_location=device))
    yt, pt = eval_all(model, test_loader, device, af_threshold=args.af_threshold)
    acc = accuracy_score(yt, pt)
    rep = classification_report(yt, pt, target_names=['NotAF','AF'], output_dict=True)
    cm = confusion_matrix(yt, pt)
    tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
    sensitivity = tp/(tp+fn) if (tp+fn)>0 else 0.0
    specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0

    rows = []
    for cls in ['NotAF','AF']:
        rows.append({'class':cls,'precision':rep[cls]['precision'],
                     'recall':rep[cls]['recall'],'f1':rep[cls]['f1-score'],
                     'support':rep[cls]['support']})
    rows.append({'class':'overall_accuracy','precision':rep['accuracy'],'recall':'','f1':'','support':len(yt)})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(args.out,'metrics.csv'), index=False)

    print("\n=== Test results ===")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:\n", cm)
    print(f"Sensitivity (recall for AF): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Saved model ->", os.path.join(args.out,'model.pt'))
    print("Saved metrics ->", os.path.join(args.out,'metrics.csv'))
    print("Tip: You can adjust --af_threshold (e.g., 0.6-0.7) to further increase AF precision.")

if __name__=='__main__':
    main()