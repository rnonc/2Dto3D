import torch.nn as nn
import torch
from camera import Camera_v2,transform_cam
from utils import coordRay
import numpy as np
from transformers import ViTConfig, ViTModel
cam = Camera_v2(35,35,32,32)

def mask_gen(dimension):
    return torch.triu(torch.full((dimension, dimension), float('-inf')), diagonal=1)

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

class Residual_fc(nn.Module):
    def __init__(self,input_dim=64,output_dim=64,nb_residual_blocks=4,hidden_state=64,dropout_rate=0):
        super(Residual_fc,self).__init__()
        self.fc_input = nn.Linear(input_dim,hidden_state)

        self.residual_blocks = nn.ModuleList()
        for l in range(nb_residual_blocks):
            layer = []
            layer.append(nn.Linear(hidden_state,hidden_state))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout_rate))

            self.residual_blocks.append(nn.Sequential(*layer))

        self.fc_output = nn.Linear(hidden_state,output_dim)


    def forward(self,x):
        x = self.fc_input(x).relu()
        for block in self.residual_blocks:
            x = x+block(x)
        x = self.fc_output(x)
        return x

class ImagePatchEncoder(nn.Module):
    def __init__(self,last_channel=64,resolution=(512,512)) -> None:
        super().__init__()
        self.last_channel = last_channel
        #self.pre_conv1 = nn.Conv2d(4,last_channel//8,5,stride=2,padding=2,bias=False)
        #self.pre_conv2 = nn.Conv2d(last_channel//8,last_channel//4,5,stride=2,padding=2,bias=False)
        self.post_conv = nn.Conv2d(4,last_channel//8,5,stride=2,padding=1,bias=False)
        self.post_conv0 = nn.Conv2d(last_channel//8,last_channel//4,3,padding=1,bias=False)
        self.post_conv1 = nn.Conv2d(last_channel//4,last_channel//2,3,bias=False)
        self.post_conv2 = nn.Conv2d(last_channel//2,last_channel//2,5,stride=2,bias=False)
        self.post_conv3 = nn.Conv2d(last_channel//2,last_channel,5,bias=False)
        self.grid = build_grid((resolution[0]//32,resolution[1]//32))#cam.create(resolution[0]//(4*16),resolution[1]//(4*16))
        self.softPosEmbbeding = nn.Linear(4,last_channel)
    
    def forward(self,x):
        #x =self.pre_conv1(x).relu()
        #x =self.pre_conv2(x).relu()
        x = x.unfold(2,32 ,32).unfold(3,32,32).permute(0,2,3,1,4,5)
        b,p_w,p_h,c,w,h = x.shape
        x= x.reshape(-1,c,w,h)

        x = self.post_conv(x).relu()
        x = self.post_conv0(x).relu()
        x = self.post_conv1(x).relu()
        x = self.post_conv2(x).relu()
        x = self.post_conv3(x).squeeze(-1).squeeze(-1)

        x = x.reshape(b,p_w,p_h,self.last_channel)
        x = x + self.softPosEmbbeding(self.grid.to(x.device))

        return x

class ViTEncoder(nn.Module):
    def __init__(self,hidden_size = 256,num_hidden_layers = 6,num_attention_heads = 4,
                                  intermediate_size = 1000,image_size = 512,
                                  patch_size = 32,num_channels = 4,encoder_stride=32) -> None:
        super().__init__()
        configuration = ViTConfig(hidden_size = hidden_size,num_hidden_layers = num_hidden_layers,num_attention_heads = num_attention_heads,
                                  intermediate_size = intermediate_size,image_size = image_size,
                                  patch_size = patch_size,num_channels = num_channels,encoder_stride=encoder_stride)
        self.model = ViTModel(configuration)
    def forward(self,x):
        return self.model(x)[0]

class ImageEncoder(nn.Module):
    def __init__(self,last_channel=64,resolution=(512,512)) -> None:
        super().__init__()
        self.last_channel = last_channel
        self.pre_conv1 = nn.Conv2d(4,last_channel//8,5,stride=2,padding=2,bias=False)
        self.pre_conv2 = nn.Conv2d(last_channel//8,last_channel//4,5,stride=2,padding=2,bias=False)

        self.post_conv = nn.Conv2d(last_channel//4,last_channel//2,5,stride=2,padding=1,bias=False)
        self.post_conv1 = nn.Conv2d(last_channel//2,last_channel//2,3,bias=False)
        self.post_conv2 = nn.Conv2d(last_channel//2,last_channel,5,stride=2,bias=False)
        self.post_conv3 = nn.Conv2d(last_channel,last_channel,5,bias=False)
        self.grid = build_grid((resolution[0]//128,resolution[1]//128))
        self.softPosEmbbeding = nn.Linear(4,last_channel)
    
    def forward(self,x):
        x =self.pre_conv1(x).relu()
        x =self.pre_conv2(x).relu()
        x = x.unfold(2,32 ,32).unfold(3,32,32).permute(0,2,3,1,4,5)
        b,p_w,p_h,c,w,h = x.shape
        x= x.reshape(-1,c,w,h)

        x = self.post_conv(x).relu()
        x = self.post_conv1(x).relu()
        x = self.post_conv2(x).relu()
        x = self.post_conv3(x).squeeze(-1).squeeze(-1)


        x = x.reshape(b,p_w,p_h,self.last_channel)
        x = x + self.softPosEmbbeding(self.grid.to(x.device))

        return x

class Encoding2Tokens(nn.Module):
    def __init__(self,d_encoder=64,num_transformer_layer=1,nb_object=20):
        super().__init__()
        self.d_encoder = d_encoder
        self.nb_object = nb_object
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        #self.TransformerSpatialDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.init_token = nn.Parameter(torch.randn(1,1,d_encoder))

    def gen_token(self,source,gen=20):
        mask = mask_gen(gen).to(source.device)
        tokens = self.init_token.repeat(source.shape[0],1,1)

        for i in range(gen):
            out = self.TransformerDecoder(tokens,source,tgt_mask=mask[:i+1,:i+1])
            tokens = torch.concat([tokens,out[:,-1:,:]],dim=-2)

        return out
    

    
    def forward(self,source):
        b = source.shape[0]
        source = source.reshape(b,-1,self.d_encoder)
        #source = self.TransformerEncoder(source) #(b,e,d_encoder)

        output = self.gen_token(source,self.nb_object)#(b,nb_object,d_encoder)
        return output

class Tokens2Decoding(nn.Module):
    def __init__(self, p_h=8,p_w=8,d_input=64,num_transformer_layer=1) -> None:
        super().__init__()
        self.rays = cam.create(p_h,p_w)
        self.positional_ff = nn.Linear(d_input,d_input+9)
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.softPosEmbbeding = nn.Linear(6,d_input)
        self.ffRes = Residual_fc(d_input,d_input)
        self.grid = build_grid((p_h,p_w))
    
    def forward(self,tokens,M_relative=None):
        positional_token, feature_tokens = tokens.split([1, tokens.shape[1]-1],dim=1)
        rest,spatial = self.positional_ff(positional_token).split([positional_token.shape[-1],9],dim=-1)
        
        if not M_relative is None:
            spatial = spatial - torch.concat([torch.zeros(spatial.shape[0],6).to(tokens.device),M_relative[...,-1]],-1).unsqueeze(1)
            M_relative = M_relative.to(tokens.device)
        rays = self.rays.unsqueeze(0).repeat(tokens.shape[0],1,1,1)
        cam = transform_cam(rays.to(tokens.device),M_relative)

        b,p_h,p_w,_ = cam.shape
        cam = coordRay(spatial,cam.reshape(b,-1,3)).squeeze(-2)
        #cam = self.grid.reshape(-1,4).to(tokens.device)
        cam = self.softPosEmbbeding(cam) + rest
        cam = self.ffRes(cam)

        feature_tokens = self.TransformerEncoder(feature_tokens)
        decoding = self.TransformerDecoder(cam,feature_tokens).reshape(b,p_h,p_w,-1)

        return decoding

class ImagePatchDecoder(nn.Module):
    def __init__(self,d_input=64,patch_size=2) -> None:
        super().__init__()
        self.d_input  = d_input
        self.patch_size = patch_size
        self.grid = build_grid((patch_size,patch_size)).squeeze(0)
        self.reduce = nn.Conv2d(d_input,64,1)
        self.softPosEmbbeding = nn.Linear(4,d_input)
        self.pre_conv0 = nn.Conv2d(64,32,3,padding=1)
        self.pre_conv_upscale0 = nn.ConvTranspose2d(32,32,3,stride=2,output_padding=1,padding=1)
        self.pre_conv1 = nn.Conv2d(32,16,3,padding=1)

        self.conv_upscale0 = nn.ConvTranspose2d(16,16,3,stride=2,output_padding=1,padding=1)
        self.conv0 = nn.Conv2d(16,16,3,padding=1)
        
        self.conv_upscale1 = nn.ConvTranspose2d(16,16,3,stride=2,output_padding=1,padding=1)
        self.conv1 = nn.Conv2d(16,4,3,padding=1)
        
        #self.conv_upscale2 = nn.ConvTranspose2d(d_input//4,d_input//8,3,stride=2,output_padding=1,padding=1)
        #self.conv2 = nn.Conv2d(d_input//8,d_input//16,3,padding=1)
        
        #self.conv_upscale3 = nn.ConvTranspose2d(d_input//8,d_input//16,3,stride=2,output_padding=1,padding=1)
        #self.conv3 = nn.Conv2d(d_input//16,4,3,padding=1)

    
    def forward(self,x):
        b,h,w,d = x.shape
        patch = self.softPosEmbbeding(self.grid.to(x.device)).permute(2,0,1)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  + patch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = x.reshape(-1,self.d_input,self.patch_size,self.patch_size)
        x = self.reduce(x).relu()
        x = self.pre_conv0(x).relu()
        x = self.pre_conv_upscale0(x)
        x = self.pre_conv1(x).relu()

        x = x.reshape((b,h,w)+x.shape[1:])
        x = x.permute(0,3,1,-2,2,-1)
        x = x.reshape(b,x.shape[1],2*h*self.patch_size,2*w*self.patch_size)

        x = self.conv_upscale0(x)
        x = self.conv0(x).relu()
        x = self.conv_upscale1(x)
        x = self.conv1(x).sigmoid()

        return x

class ImageDecoder(nn.Module):
    def __init__(self,d_input=64) -> None:
        super().__init__()
        self.d_input  = d_input

        self.conv_upscale0 = nn.ConvTranspose2d(d_input,d_input,3,stride=2,output_padding=1,padding=1)
        self.conv0 = nn.Conv2d(d_input,d_input//2,3,padding=1)
        self.conv_upscale1 = nn.ConvTranspose2d(d_input//2,d_input//2,3,stride=2,output_padding=1,padding=1)
        self.conv1 = nn.Conv2d(d_input//2,d_input//4,3,padding=1)
        self.conv_upscale2 = nn.ConvTranspose2d(d_input//4,d_input//4,3,stride=2,output_padding=1,padding=1)
        self.conv2 = nn.Conv2d(d_input//4,d_input//8,3,padding=1)
        self.conv_upscale3 = nn.ConvTranspose2d(d_input//8,d_input//8,3,stride=2,output_padding=1,padding=1)
        self.conv3 = nn.Conv2d(d_input//8,4,3,padding=1)
        

    
    def forward(self,x):
        x= x.permute(0,-1,1,2)

        x = self.conv_upscale0(x).relu()
        x = self.conv0(x).relu()
        x = self.conv_upscale1(x).relu()
        x = self.conv1(x).relu()
        x = self.conv_upscale2(x).relu()
        x = self.conv2(x).relu()
        x = self.conv_upscale3(x).relu()
        x = self.conv3(x)

        return x

class MultiViewModel(nn.Module):
    def __init__(self,dimension=256, resolution=(512,512), nb_object=20, p_h=32, p_w=32, num_transformer_layer=4) -> None:
        super().__init__()
        #self.encoder = ImageEncoder( dimension, resolution)
        self.encoder = ViTEncoder(dimension)
        self.transf = Encoding2Tokens(dimension, num_transformer_layer, nb_object)
        self.transfd = Tokens2Decoding(p_h,p_w,dimension,num_transformer_layer)
        self.decoder = ImageDecoder(dimension)

    def encoding(self,input):
        output  = self.encoder(input)
        output = self.transf(output)
        return output
    
    def decoding(self,encoded,M_relative=None):
        output = self.transfd(encoded,M_relative)
        output = self.decoder(output)

        return output
    
    def forward(self,input,M_relative=None):
        enc = self.encoding(input)
        output = self.decoding(enc,M_relative)
        
        return output




if __name__ == "__main__":
    model = MultiViewModel()
    M_relative = torch.concat([torch.eye(3),torch.zeros(3).unsqueeze(1)],-1).unsqueeze(0).repeat(2,1,1)
    input = torch.rand(2,4,512,512)

    #print(model(input[0].unsqueeze(0),M_relative[0].unsqueeze(0)))
    output = model(input,M_relative)
    print(model.transf.TransformerDecoder)
    print(output.shape)

    input = torch.rand(2,4,512,512)
    ViT = ViTEncoder()
    print(ViT)
    print(ViT(input)[0].shape)


