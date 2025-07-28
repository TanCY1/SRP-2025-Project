import torch
from torch import nn
from tqdm import trange

device = "cuda" if torch.cuda.is_available else "cpu"

def trainModel(model:nn.Module, train_loader,optimiser=None,device=torch.device("cpu"),num_epochs=10,):
    
    model = model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
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
    return model