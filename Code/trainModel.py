import torch
from torch import nn
from tqdm import trange
from sklearn.metrics import roc_auc_score
from copy import deepcopy

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_roc_auc(model,val_loader):
    model.eval()
    y_true=[]
    y_score=[]
    
    with torch.no_grad():
        for images,mols,labels in val_loader:
            images=images.to(device)
            mols = mols.to(device)
            labels=labels.to(device)
            logits = model(images,mols)
            score = torch.nn.functional.softmax(logits,dim=1)[:,1]
            y_true.extend(labels.cpu().numpy())
            y_score.extend(score.cpu().numpy())
    return roc_auc_score(y_true,y_score)


def trainModel(model:nn.Module, train_loader,optimiser=None,device=torch.device("cpu"),num_epochs=10,val_loader=None,patience=5):
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_roc_auc = 0
    stale_epochs = 0
    best_model_state_dict = None
    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, mol, labels in train_loader:
            images = images.to(device)
            mol = mol.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = model(images,mol)
            loss = loss_fn(logits,labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} Done. Avg Loss: {running_loss / len(train_loader):.4f}")
        
        if val_loader is not None:
            roc_auc = evaluate_roc_auc(model,val_loader)
            
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                stale_epochs=0
                best_model_state_dict=deepcopy(model.state_dict())
            else:
                stale_epochs+=1
                if stale_epochs>=patience:
                    print(f"Early stopping triggered at epoch {epoch} (best AUC: {best_roc_auc:.4f}).")
                    break
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
            
    return model,best_roc_auc