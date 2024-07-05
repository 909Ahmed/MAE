import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets

class Sampler(nn.Module):
    
    def __init__(self, total_tokens, mask_per):
        super(Sampler, self).__init__()

        self.perm = torch.empty(total_tokens, dtype=torch.int64)
        self.reperm = torch.empty(total_tokens, dtype=torch.int64)
        self.retain = (total_tokens * mask_per) // 100
        self.total_tokens = total_tokens
        
    def shuffle(self, x, new_perm=True):
        
        if new_perm:
            self.perm = torch.randperm(self.total_tokens)
        x = x[:,self.perm]
        x = x[:, ]
        y, z = torch.split_with_sizes(x, [self.retain, self.total_tokens - self.retain], dim=1)
        return y, z
        
    def reshuffle(self, y, z):
        
        x = torch.cat((y, z), dim=1)
        self.reperm[self.perm] = torch.arange(self.reperm.size(0))
        x =  x[:,self.reperm]
        return x

train_transforms = transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
])

train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
test = datasets.CIFAR10('./data', train=False, download=True, transform=train_transforms)

tdlr = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
vdlr = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

max_lr = 3 * 1e-3
min_lr = 0.1 * max_lr
epochs = 100
epoch_steps = len(tdlr)
gradient_step = 4         #check this shit
max_steps = (epoch_steps * epochs) / gradient_step
warmup_steps = (max_steps * 20) // 100
loss_fn = torch.nn.MSELoss()

def get_lr(steps):
    
    if steps < warmup_steps:
        return ((steps + 1) / warmup_steps) * max_lr
    else:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(((steps - warmup_steps) / (max_steps - warmup_steps)) * math.pi))


def train(
    model,
    optimizer,
    loss_fn,
    epoch,
    gradient_step,
    tdlr
):
    
    model.train()
    device = 'cuda'
    pbar = tqdm(tdlr)
    optimizer.zero_grad()
    
    for step, (image, label) in enumerate(pbar):
        
        image = image.to(device)
        loss_accum = 0
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            re_image = model(image)
            loss = loss_fn(image, re_image)
            loss = loss / gradient_step
            loss_accum += loss.detach()
            loss.backward()
        
        if (step + 1) % gradient_step == 0:
            norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

            lr = get_lr(step + epoch * epoch_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            torch.cuda.synchronize()
            pbar.set_description(f'Loss at step: {loss.item()}')


def test(
    model,
    loss_fn,
    vdlr
):
    
    model.eval()
    pbar = tqdm(vdlr)
    for step, (image, label) in enumerate(pbar):
        
        image = image1.to(device)
        with torch.no_grad():
            
            re_image, loss = model(image)
            pbar.set_description(f'Loss at step: {loss.item()}')