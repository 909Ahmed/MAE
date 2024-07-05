import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import repeat, rearrange
from utils import Sampler

class Patch_Layer(nn.Module):

    def __init__(self, p1, p2, c, num_patches, emd_dim) -> None:
        super(Patch_Layer, self).__init__()

        patch_dim = p1 * p2 * c
        self.p1 = p1
        self.p2 = p2
        self.pos_emd = nn.Parameter(torch.randn(1, num_patches, emd_dim), requires_grad=True)
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emd_dim)
        )

    def forward(self, x):

        x = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                      ph=self.p1,
                      pw=self.p2)

        x = self.patch_embed(x)        
        x += self.pos_emd
 
        return x

class MultiHead(nn.Module):

    def __init__(self, emd_dim, d_model, head):
        super(MultiHead, self).__init__()

        self.d_model = d_model
        self.head = head
        self.qmat = nn.Linear(emd_dim, d_model)
        self.kmat = nn.Linear(emd_dim, d_model)
        self.vmat = nn.Linear(emd_dim, d_model)
        self.omat = nn.Linear(d_model, emd_dim)

    def make_heads(self, x):
        return x.view(x.size()[0], x.size()[1], self.head, self.d_model // self.head).transpose(1, 2)

    def forward(self, x):

        q, k, v = self.qmat(x), self.kmat(x), self.vmat(x)
        q, k, v = self.make_heads(q), self.make_heads(k), self.make_heads(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(x.size()[0], -1, self.d_model)

        return self.omat(x)

class MLP(nn.Module):
    
    def __init__(self, emd_dim):
        super(MLP, self).__init__()
        self.emd_dim = emd_dim
        self.ff = nn.Sequential(
            nn.Linear(self.emd_dim, self.emd_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.emd_dim * 2, self.emd_dim)
        )
        
    def forward(self, input_tensor): 
        return self.ff(input_tensor)

class BLOCK(nn.Module):

    def __init__(self, emd_dim, d_model, heads):
        super(BLOCK, self).__init__()

        self.norm1 = nn.LayerNorm(emd_dim)
        self.multihead = MultiHead(emd_dim, d_model, heads)
        self.norm2 = nn.LayerNorm(emd_dim)
        self.ff = MLP(emd_dim)

    def forward(self, x):
        
        x = x + self.multihead(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        
        return x

class Encoder(nn.Module):

    def __init__(self, 
                 
                sampler,
                num_encoders=12,
                emd_dim=512,
                d_model=512,
                heads=4,
                p1=16,                
                p2=16,
                image_shape=(3, 224, 224)

                ):
        super(Encoder, self).__init__()
    
        num_patches = (image_shape[1] // p1) * (image_shape[2] // p2)
        self.num_encoders = num_encoders
        c = image_shape[0]
        self.norm = nn.LayerNorm(emd_dim)
        self.pos_emd = Patch_Layer(p1, p2, c, num_patches, emd_dim)
        self.encoders = nn.ModuleList([BLOCK(emd_dim, d_model, heads) for _ in range(num_encoders)])
        self.sampler = sampler
        
    def forward(self, x):
        
        x = self.pos_emd(x)
        x, y = self.sampler.shuffle(x)
        for i in range(self.num_encoders):
            x = self.encoders[i](x)

        x = self.norm(x)

        return x, y

class Decoder(nn.Module):
    
    def __init__(self, 
                sampler,
                num_patches,
                num_decoders=4,
                emd_dim=512,
                d_model=512,
                heads=4,
                p1=16,
                p2=16,
        
        ):
        super(Decoder, self).__init__()
        
        self.num_patches = num_patches
        self.sampler = sampler
        self.num_decoders = num_decoders
        self.p1 = p1
        self.p2 = p2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emd_dim))
        self.pos_emd = nn.Parameter(torch.randn(1, num_patches, emd_dim), requires_grad=True)
        self.decoders = nn.ModuleList([BLOCK(emd_dim, d_model, heads) for _ in range(num_decoders)])
        c = 3 #fix this shit
        self.image_dim = nn.Sequential(    
            nn.LayerNorm(emd_dim),
            nn.Linear(emd_dim, p1 * p2 * c)
        )

    def forward(self, x):
        
        masks = self.mask_token.repeat(x.size(0), self.num_patches - x.size(1), 1)
        x = self.sampler.reshuffle(x, masks)
        x += self.pos_emd
        for i in range(self.num_decoders):
            x = self.decoders[i](x)

        x = self.image_dim(x)

        img = rearrange(x, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                      ph=self.p1,
                      pw=self.p2,
                      nh = 224 // self.p1,
                      nw = 224 // self.p2
                    )

        return img

class MAE(nn.Module):
    
    def __init__(self):
        super(MAE, self).__init__()
        #hardest coder in the room lols
        self.sam = Sampler(196, 25)
        self.encoder = Encoder(self.sam)
        self.decoder = Decoder(self.sam, 196)
        
    def forward(self, x):
        
        return self.decoder(self.encoder(x)[0])