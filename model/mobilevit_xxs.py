import torch
from torch import nn
from einops import rearrange

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_1x1_bn_relu(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def norm(dim):
  return nn.LayerNorm(dim)

def conv_1x1(inp, oup, mode = 1):
    if mode == 0:
      return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
    )
    else:
      return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.SiLU()
    )

def conv_Depth(dim, output, mode):
  if mode == 1:
    return nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1, groups=dim, bias=False),
        nn.BatchNorm2d(dim),
        nn.ReLU(),
        nn.Conv2d(dim, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output))
  else:
    return nn.Sequential(nn.Conv2d(dim, dim, 3, 2, 1, groups=dim, bias=False),
        nn.BatchNorm2d(dim),
        nn.Conv2d(dim, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        nn.ReLU())

    
def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        if dim == 80 or dim == 96:
          layers = [nn.Linear(dim, hidden_dim // 2),
              nn.SiLU(),
              nn.Dropout(dropout),
              nn.Linear(hidden_dim // 2 , dim)]

          self.net = nn.Sequential(*layers)

        else:
          layers = [nn.Linear(dim, hidden_dim),
              nn.SiLU(),
              nn.Dropout(dropout),
              nn.Linear(hidden_dim, dim)]
          
          self.net = nn.Sequential(*layers)
        
      
    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * self.heads
        project_out = not (heads == 1 and dim_head == dim)

        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.heads = heads
        self.attend = nn.Softmax(dim = -1)

        if dim == 80:
          self.to_qkv = nn.Linear(dim, 240, bias = True)

        elif dim == 96:
          self.to_qkv = nn.Linear(dim, 288, bias = True)
        else:
          self.to_qkv = nn.Linear(dim, inner_dim * 6, bias = True)

        if dim == 80 or dim == 96:
          self.to_out = nn.Sequential(
              nn.Linear(dim, dim),
              nn.Dropout(dropout)
          ) if project_out else nn.Identity()  
        
        else:
          self.to_out = nn.Sequential(
              nn.Linear(inner_dim *2, dim),
              nn.Dropout(dropout)
          ) if project_out else nn.Identity()

    def forward(self, x): # fixed

        bt, b_sz, n_patches, in_channels = x.shape

        qkv = (self.to_qkv(x).reshape(bt, b_sz, n_patches, 3, self.heads, -1))
        qkv = qkv.transpose(2, 4)

        # [B x h x 3 x N x C] --> [B x h x N x C] x 3
        query, key, value = qkv[:, :, :,0], qkv[:, :, :, 1], qkv[:, :, :, 2]

        query = query * self.scale
        key = key.transpose(3, 4)
        # QK^T
        # [B x h x N x c] x [B x h x c x N] --> [B x h x N x N]
        attn = torch.matmul(query, key)
        attn = self.attend(attn)
        # weighted sum
        out = torch.matmul(attn, value)

        out = out.transpose(2, 3).reshape(bt, b_sz, n_patches, -1)

        out = self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for k in range(depth):
            if (k+1) % depth != 0:
              self.layers.append(nn.ModuleList([
                  PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                  PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
              ]))
            else:
              self.layers.append(nn.ModuleList([
                  PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                  PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                  norm(dim)
              ]))


    def forward(self, x): 
        for layer in self.layers:
          if len(layer) == 2:
            x = layer[0](x) + x
            x = layer[1](x) + x
          else:
            x = layer[0](x) + x
            x = layer[1](x) + x
            x = layer[2](x)  
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1(channel, dim, 0)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        x = self.transformer(x)

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        # Fusion
        x = self.conv3(x)

        x = torch.cat((y, x), 1) # fixed
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)
        self.conv_Depth_1 = conv_Depth(80, 256, 0)
        self.conv_Depth_2 = conv_Depth(256, 128, 0)
        self.conv_Depth_3 = conv_Depth(128, 128, 0)

        self.pool = nn.AvgPool2d(2,1)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

    def forward(self, x):

        x = self.conv1(x)

        x = self.mv2[0](x)

        x = self.mv2[1](x)

        x = self.mv2[2](x)

        x = self.mv2[3](x)      # Repeat
        
        x = self.mv2[4](x)

        x = self.mvit[0](x)

        x = self.mv2[5](x)

        first_head_in = self.mvit[1](x)

        x = self.mv2[6](first_head_in)

        second_head_in = self.mvit[2](x)

        third_head_in = self.conv_Depth_1(second_head_in)

        fourth_head_in = self.conv_Depth_2(third_head_in)
        
        fifth_head_in = self.conv_Depth_3(fourth_head_in)

        last_head = self.pool(fifth_head_in).view(-1, fifth_head_in.shape[1])

        return first_head_in, second_head_in, third_head_in, fourth_head_in, fifth_head_in, last_head

class MobileDetector(nn.Module):
  def __init__(self):
    super().__init__()
    self.dims = [64, 80, 96]
    self.channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    self.heads = MobileViT((320, 320),self.dims, self.channels, expansion=2)

    # Projections:
    self.project_1 = conv_1x1_bn_relu(64, 512)
    self.project_2 = conv_1x1_bn_relu(80, 256)

    # First outputs
    self.output_1_prev =  nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False),
                          nn.BatchNorm2d(512))
    self.output_1 = nn.Conv2d(512, 4*6, 1, 1, 0, bias=True)

    # Second output 
    self.output_2_prev =  nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
                          nn.BatchNorm2d(256))
    self.output_2 = nn.Conv2d(256, 4*6, 1, 1, 0, bias=True)
    
    # Third output
    self.output_3_prev =  nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False),
                          nn.BatchNorm2d(256))
    self.output_3 = nn.Conv2d(256, 4*6, 1, 1, 0, bias=True)
    
    # Fourth output
    self.output_4_prev =  nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
                          nn.BatchNorm2d(128))
    self.output_4 = nn.Conv2d(128, 4*6, 1, 1, 0, bias=True)

    # Fifth output
    self.output_5_prev =  nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
                          nn.BatchNorm2d(128))
    self.output_5 = nn.Conv2d(128, 4*6, 1, 1, 0, bias=True)

    # 1 x 1 Conv last
    self.second_last = nn.Sequential(
        nn.Conv2d(128, 64, 1, 1, 0, bias=False),
        nn.ReLU())

    self.last = nn.Conv2d(64, 4*4, 1, 1, 0, bias=True)


  def forward(self, x):

    outs = self.heads.forward(x)    
    # 1st SSD Head
    proj_out = self.project_1(outs[0])
    proj_out_k = self.output_1_prev(proj_out)
    out_1 = self.output_1(proj_out_k)
    
    # 2nd SSD Head
    proj_out_1 = self.project_2(outs[1])
    proj_out_1_k = self.output_2_prev(proj_out_1)
    out_2 = self.output_2(proj_out_1_k)

    # 3rd SSD Head
    prev_1 = self.output_3_prev(outs[2])
    out_3 = self.output_3(prev_1)

    # 4th SSD Head
    prev_2 = self.output_4_prev(outs[3])
    out_4 = self.output_4(prev_2)

    # 5th SSD Head
    prev_3 = self.output_5_prev(outs[4])
    out_5 = self.output_5(prev_3)
    
    # 6th SSD Head
    sec = self.second_last(outs[5].unsqueeze(2).reshape(outs[5].shape[0], 128, 1, 1))
    las = self.last(sec)

    return [out_1, out_2, out_3, out_4, out_5, las], [proj_out_k, proj_out_1_k, prev_1, prev_2, prev_3, sec]


class MobileDetector_Tr(nn.Module):

  def __init__(self, classes=81):
    super().__init__()
    self.model = MobileDetector()
    self.output_1_class =  nn.Conv2d(512, (classes)*6, 1, 1, 0, bias=True)
    self.output_2_class =  nn.Conv2d(256, (classes)*6, 1, 1, 0, bias=True)
    self.output_3_class =  nn.Conv2d(256, (classes)*6, 1, 1, 0, bias=True)
    self.output_4_class =  nn.Conv2d(128, (classes)*6, 1, 1, 0, bias=True)
    self.output_5_class =  nn.Conv2d(128, (classes)*6, 1, 1, 0, bias=True)
    self.last_class = nn.Conv2d(64, classes*4, 1, 1, 0, bias=True)
    self.classes = classes


  def forward(self, x):  

    bt = x.shape[0]
    box_vals, class_vals = self.model.forward(x)
    out_1_class = self.output_1_class(class_vals[0])
    out_2_class = self.output_2_class(class_vals[1])
    out_3_class = self.output_3_class(class_vals[2])
    out_4_class = self.output_4_class(class_vals[3])
    out_5_class = self.output_5_class(class_vals[4])
    las_class = self.last_class(class_vals[5])

    box_locations_1 = box_vals[0].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_1 = out_1_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)

    box_locations_2 = box_vals[1].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_2 = out_2_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)

    box_locations_3 = box_vals[2].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_3 = out_3_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)

    box_locations_4 = box_vals[3].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_4 = out_4_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)

    box_locations_5 = box_vals[4].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_5 = out_5_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)

    box_locations_6 = box_vals[5].permute(0,2,3,1).contiguous().view(1, bt, -1, 4)
    box_classes_6 = las_class.permute(0,2,3,1).contiguous().view(1, bt, -1, self.classes)


    box_locations = torch.cat((box_locations_1, box_locations_2,
                                box_locations_3, box_locations_4, 
                                box_locations_5, box_locations_6), dim=2)
    box_classes = torch.cat((box_classes_1, box_classes_2,
                              box_classes_3, box_classes_4, 
                              box_classes_5, box_classes_6), dim=2)

    return box_locations, box_classes
