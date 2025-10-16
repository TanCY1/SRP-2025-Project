from torch import nn
import torch
from typing import Optional,Literal
from collections.abc import Callable
from torch.utils.data import DataLoader
from copy import deepcopy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            
def trainModel(model:nn.Module,
               train_loader:DataLoader,
               combine_timepoints,
               out_features,
               loss_fn:Optional[Callable]=None,
               optimiser:Optional[torch.optim.Optimizer]=None,
               num_epochs:int=20,
               val_loader:Optional[DataLoader]=None,
               score_fn:Optional[Callable[[nn.Module,DataLoader,bool,Literal[1,2],dict],float]]=None,
               score_name:str="Score",
               patience:Optional[int]=None,
               trainKwargs:dict={},
               testKwargs:dict={}) -> tuple[nn.Module, float]:
    
    if optimiser is None:
        raise ValueError("Optimiser must be provided")
    if loss_fn is None:
        raise ValueError("Loss function must be provided")
    if val_loader is not None:
        if score_fn is None:
            raise ValueError("Score function must be provided")
        if patience is None:
            raise ValueError("Patience must be provided")

    bestScore = 0
    staleEpochs = 0
    best_model_state_dict = None
    
    
    for epoch in range(num_epochs):
        model.train()
        losses = 0.
        for T0_volumes, T3_volumes, mols, labels in train_loader:
            T0_volumes = T0_volumes.to(device)
            T3_volumes = T3_volumes.to(device)
            mols = mols.to(device)
            labels:torch.Tensor = labels.to(device)
            optimiser.zero_grad()
            if combine_timepoints:
                volumes = torch.cat((T0_volumes,T3_volumes),dim=1)
                logits = model(volumes,mols,**trainKwargs)
            else:
                labels = labels.to(dtype=torch.float32)
                logits = model(T0_volumes,T3_volumes,mols,**trainKwargs)
            
            if out_features==1:
                logits = logits.squeeze(1)
            loss:torch.Tensor = loss_fn(logits,labels)
            loss.backward()
            optimiser.step()
            losses+=loss.item()
        avg_loss = losses/len(train_loader)
        print(f"Epoch {epoch} Done. Average Loss={avg_loss:.4f}")
        if val_loader is not None:
            assert score_fn is not None
            assert patience is not None
            score = score_fn(model,val_loader,combine_timepoints,out_features,testKwargs)
            print(f"{score_name}={score}")
            if score > bestScore:
                bestScore = score
                staleEpochs=0
                best_model_state_dict = deepcopy(model.state_dict())
            else:
                staleEpochs+=1
                if staleEpochs>=patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best {score_name}={bestScore}")
                    break
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    return model, bestScore