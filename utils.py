import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets
from einops.einops import repeat
import math
from torch.optim.lr_scheduler import LambdaLR

class Sampler(nn.Module):
    
    def __init__(self, total_tokens, mask_per):
        super(Sampler, self).__init__()

        self.perm = torch.empty(total_tokens, dtype=torch.int64)
        self.retain = (total_tokens * mask_per) // 100
        self.total_tokens = total_tokens
        
    def shuffle(self, x, new_perm=True):
        
        self.perm = torch.randperm(self.total_tokens).to('cuda')
        x = torch.gather(x, 1, repeat(self.perm, 't -> b t c', c=x.size(2), b=x.size(0)))
        y, z = torch.split_with_sizes(x, [self.retain, self.total_tokens - self.retain], dim=1)
        return y, z
        
    def reshuffle(self, y, z):
        
        rev = torch.argsort(self.perm).to('cuda')
        x = torch.cat((y, z), dim=1)
        x = torch.gather(x, 1, repeat(rev, 't -> b t c', c=x.size(2), b=x.size(0)))
        return x

train_transforms = transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
])

train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
test = datasets.CIFAR10('./data', train=False, download=True, transform=train_transforms)

tdlr = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
vdlr = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

max_lr = 1e-5
min_lr = 0.1 * max_lr
epochs = 100
epoch_steps = len(tdlr)
gradient_step = 4
max_steps = (epoch_steps * epochs) / gradient_step
warmup_steps = (max_steps * 20) // 100
loss_fn = torch.nn.MSELoss()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train(
    model,
    optimizer,
    sch,
    loss_fn,
    epoch,
    tdlr
):
    
    model.train()
    device = 'cuda'
    pbar = tqdm(tdlr)
    optimizer.zero_grad()
    
    loss_accum = 0
    for step, (image, label) in enumerate(pbar):
        
        image = image.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            re_image = model(image)
            loss = loss_fn(image, re_image)
            loss_accum += loss.detach()
            loss.backward()
        
        if (epoch * epoch_steps + step + 1) % gradient_step == 0:
            optimizer.step()
            sch.step()
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            
            pbar.set_description(f'step{step + epoch * epoch_steps} loss : {loss.item()}  lr : {lr}')
            loss_accum = 0
            
def test(
    model,
    loss_fn,
    epoch,
    vdlr
):
    
    model.eval()
    pbar = tqdm(vdlr)
    loss = 0
    for step, (image, label) in enumerate(pbar):
        
        image = image.to('cuda')
        with torch.no_grad():
            
            re_image = model(image)
            loss += loss_fn(image, re_image)
            
            if (epoch * epoch_steps + step + 1) % gradient_step == 0:        
                pbar.set_description(f'Loss at step: {loss.item()}')
                loss = 0