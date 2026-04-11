import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
from model_defs.mam_aggregator import MAM_Classifier
from model_defs.comparative_model import Comparative_MIL_Model

FEAT_DIR = "/content/drive/MyDrive/CSC3094_Project/features/pt_files"
CSV_PATH = "/content/drive/MyDrive/CSC3094_Project/bcr_labels.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
LR = 1e-4

from model_defs.comparative_model import PRAD_BCR_Dataset # 使用之前写的 Dataset 类

def train_one_model(model_type='MAM'):
    print(f"\n--- Starting Training: {model_type} ---")
    
    if model_type == 'MAM':
        model = MAM_Classifier().to(DEVICE)
    else:
        model = Comparative_MIL_Model(method='CAMIL').to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    dataset = PRAD_BCR_Dataset(CSV_PATH, FEAT_DIR)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_db, test_db = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_db, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=1, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for features, label in train_loader:
            features, label = features.squeeze(0).to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        metrics = evaluate(model, test_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {train_loss/len(train_loader):.4f} | AUC: {metrics['auc']:.4f} | Acc: {metrics['acc']:.4f}")

    return metrics

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for features, label in loader:
            features = features.squeeze(0).to(DEVICE)
            logits = model(features)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            
            all_labels.append(label.item())
            all_probs.append(prob.item())
            all_preds.append(pred.item())
            
    return {
        'acc': accuracy_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds)
    }

if __name__ == '__main__':
    mam_results = train_one_model('MAM')

    camil_results = train_one_model('CAMIL')
    
    results_df = pd.DataFrame([mam_results, camil_results], index=['MAM (Baseline)', 'CAMIL (Experimental)'])
    print("\n" + "="*30)
    print("FINAL COMPARATIVE RESULTS")
    print("="*30)
    print(results_df)