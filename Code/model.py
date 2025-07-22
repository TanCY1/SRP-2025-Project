#import lightning as L
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset

def centreCrop3D(tensor:Tensor,target_shape):
    b,c,x,y,z = tensor.shape
    tx,ty,tz = target_shape
    sx = (x-tx)//2
    sy = (y-ty)//2
    sz = (z-tz)//2
    return tensor[:,:,sx:sx+tx,sy:sy+ty,sz:sz+tz]

class CMCUnit(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.maxPoolingPath = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )

    def forward(self,x):
        x_pool = self.maxPoolingPath(x)
        # print(x_pool.shape)
        x_crop = centreCrop3D(x,x_pool.shape[-3:])
        # print(x_crop.shape)
        return torch.cat((x_pool,x_crop),dim=1)

class FeatureExtractionUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.CMCs = nn.Sequential(
            CMCUnit(1), 
            CMCUnit(2), 
            CMCUnit(4), 
            CMCUnit(8), 
            CMCUnit(16)
        )
    def forward(self,x):
        return self.CMCs(x)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.FEUs = nn.ModuleList([FeatureExtractionUnit() for _ in range(3)])
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(24579,512)
        self.fc2 = nn.Linear(512,2)
    def forward(self,images,mol):
        channels = torch.split(images,1,dim=1)
        x = [feu(ch) for ch,feu in zip(channels, self.FEUs)]
        x = torch.cat(x,dim=1)
        assert x.is_contiguous()
        x = x.view(x.size(0),-1)
        x = torch.cat([x,mol],dim=1) #shape of (B,24579)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''       
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model()
        self.loss_fn = nn.CrossEntropyLoss()
    def training_step(self, batch, batch_idx):
        images, mol, labels = batch
        logits = self.model(images, mol)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True,on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch, batch_idx,):
        images, mol, labels = batch
        logits = self.model(images, mol)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    def on_test_epoch_start(self):
        self.preds = list()
        self.labels = list()
    
    def test_step(self, batch, batch_idx):
        images, mol, labels = batch
        logits = self.model(images, mol)
        loss = self.loss_fn(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.preds.append(logits.argmax(dim=1))
        self.labels.append(labels)
        return loss
    def on_test_epoch_end(self):
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)
        acc = (preds == labels).float().mean()
        self.log("acc", acc, )
'''    



