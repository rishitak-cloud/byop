import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet_parts import unet
from duts_dataset import dataset, ApplyTransform, train_transform, val_transform
import torch.nn.functional as F

class Dice(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        inte = (pred * target).sum()
        dice = 2 * (inte + self.smooth)/(pred.sum() + target.sum() + self.smooth)
        return bce+1-dice

if __name__=="__main__":
    rate = 3e-4
    batch_size = 32
    epochs = 2
    datapath = "/Users/rishitakandpal/Downloads/duts_dataset/"
    savepath = "/Users/rishitakandpal/Documents/byop/byop/trained/"
    device = torch.device("mps")
    base_dataset = dataset(datapath)
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(base_dataset, [0.8,0.2], generator=generator)

    train_dataset = ApplyTransform(train_subset, transform=train_transform())
    val_dataset = ApplyTransform(val_subset, transform=val_transform())

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    model = unet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=rate)
    criterion = Dice(1e-6)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), savepath)

