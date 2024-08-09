import torch.nn as nn
import torch, math
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
        self.TransformerSpatialDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_encoder, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.init_token = nn.Parameter(torch.randn(1,1,d_encoder))

    def gen_token(self,source,gen=20):
        mask = mask_gen(gen).to(source.device)
        tokens_pos = self.init_token.repeat(source.shape[0],1,1)
        tokens_pos = self.TransformerSpatialDecoder(tokens_pos,source)

        for i in range(gen):
            out = self.TransformerDecoder(tokens_pos,source,tgt_mask=mask[:i+1,:i+1])
            tokens_pos = torch.concat([tokens_pos,out[:,-1:,:]],dim=-2)

        return tokens_pos
    

    
    def forward(self,source):
        b = source.shape[0]
        source = source.reshape(b,-1,self.d_encoder)
        source = self.TransformerEncoder(source) #(b,e,d_encoder)

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
        #self.grid = build_grid((p_h,p_w))
    
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
        cam = self.softPosEmbbeding(cam) #+ rest
        

        feature_tokens = self.TransformerEncoder(feature_tokens)
        decoding = self.TransformerDecoder(cam,feature_tokens).reshape(b,p_h,p_w,-1)
        decoding = self.ffRes(decoding)
        #decoding  = cam.reshape(b,p_h,p_w,-1)
        return decoding

class Tokens2Decodingv2(nn.Module):
    def __init__(self, p_h=8,p_w=8,d_input=64,num_transformer_layer=1) -> None:
        super().__init__()
        self.rays = cam.create(p_h,p_w)
        self.positional_ff = nn.Linear(d_input,d_input+9)
        self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.TransformerEncoderDist = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.TransformerDecoderDist = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)
        self.softPosEmbbeding = nn.Linear(6,d_input)
        self.ffRes = Residual_fc(d_input,1)
        self.ff_pos = nn.Linear(3,d_input)
        self.ff_dir = nn.Linear(3,d_input)
        #self.grid = build_grid((p_h,p_w))
    
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
        dist_tokens = self.TransformerEncoderDist(feature_tokens)
        dist = self.softPosEmbbeding(cam)# + rest
        dist = self.TransformerDecoderDist(dist,dist_tokens)
        dist = self.ffRes(dist)

        position = torch.cross(cam[...,:3],cam[...,3:],-1) + dist*cam[...,3:]
        position = self.ff_pos(position)
        feature_tokens = self.TransformerEncoder(feature_tokens)
        decoding = self.TransformerDecoder(position,feature_tokens).reshape(b,p_h,p_w,-1)
        direction  = self.ff_dir(cam[...,3:]).reshape(b,p_h,p_w,-1)
        decoding = direction + decoding
        #decoding  = cam.reshape(b,p_h,p_w,-1)
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

class ImageDecoder128(nn.Module):
    def __init__(self,d_input=64) -> None:
        super().__init__()
        self.d_input  = d_input

        self.conv_upscale0 = nn.ConvTranspose2d(d_input,d_input,3,stride=2,output_padding=1,padding=1)
        self.conv0 = nn.Conv2d(d_input,d_input//4,3,padding=1)
        self.conv_upscale1 = nn.ConvTranspose2d(d_input//4,d_input//8,3,stride=2,output_padding=1,padding=1)
        self.conv1 = nn.Conv2d(d_input//8,4,3,padding=1)

    def forward(self,x):
        x= x.permute(0,-1,1,2)

        c = self.conv_upscale0(x).relu()
        c = self.conv0(c).relu()
        c = self.conv_upscale1(c).relu()
        c = self.conv1(c)

        return c

class MultiViewModel(nn.Module):
    def __init__(self,dimension=128, resolution=(128,128), nb_object=20, p_h=32, p_w=32, num_transformer_layer=2) -> None:
        super().__init__()
        self.encoder = ImageEncoder( dimension, resolution)
        #self.encoder = ViTEncoder(dimension)
        self.transf = Encoding2Tokens(dimension, num_transformer_layer, nb_object)
        self.transfd = Tokens2Decoding(p_h,p_w,dimension,num_transformer_layer)
        self.decoder = ImageDecoder128(dimension)

    def encoding(self,input):
        output  = self.encoder(input)
        output = self.transf(output)
        return output
    
    def decoding(self,encoded,M_relative=None):
        if len(M_relative.shape) == 4:
            output = []
            for i in range(M_relative.shape[1]):
                constraint =  self.transfd(encoded,M_relative[:,i])
                output.append(self.decoder(constraint))
            output = torch.stack(output,1)
        else:
            constraint =  self.transfd(encoded,M_relative)
            output = self.decoder(constraint)
        return output
    
    def forward(self,input,M_relative=None):
        enc = self.encoding(input)
        output = self.decoding(enc,M_relative)
        
        return output

class MultiViewDiffusionModel(nn.Module):
    def __init__(self,dimension=128, resolution=(128,128), nb_object=20, p_h=32, p_w=32, num_transformer_layer=2,max_t=10) -> None:
        super().__init__()
        self.encoder = ImageEncoder( dimension, resolution)
        self.transf = Encoding2Tokens(dimension, num_transformer_layer, nb_object)
        self.diffusion = Unet_block(dimension,num_transformer_layer,max_t)

    def forward(self,input,M_relative=None):
        self.encoder.eval()
        self.transf.eval()
        with torch.no_grad():
            input_enc  = self.encoder(input)
            enc = self.transf(input_enc)
            if not M_relative is None and len(M_relative.shape) == 4:
                output = []
                for i in range(M_relative.shape[1]):
                    output.append(self.diffusion.eval_image(enc,M_relative[:,i]))
                output = torch.stack(output,1)
            else:
                output = self.diffusion.eval_image(enc,M_relative)
            
            return output
    
    def train_loop(self,X,Y,M_relative=None):
        self.encoder.train()
        self.transf.train()
        input_enc  = self.encoder(X)
        enc = self.transf(input_enc)
        if not M_relative is None and len(M_relative.shape) == 4:
            predicted_noises = []
            noises = []
            for i in range(M_relative.shape[1]):
                predicted_noise,noise =  self.diffusion.train_predict_noise(Y[:,i],enc,M_relative[:,i])
                predicted_noises.append(predicted_noise)
                noises.append(noise)
            predicted_noises = torch.stack(predicted_noises,1)
            noises = torch.stack(noises,1)
            return predicted_noises,noises
        else:
            output = self.diffusion.train_predict_noise(Y,enc,M_relative)
        
        return output

    def partial_diffusion(self,X,Y,t,M_relative=None):
        self.encoder.eval()
        self.transf.eval()
        with torch.no_grad():
            input_enc  = self.encoder(X)
            enc = self.transf(input_enc)
            if not M_relative is None and len(M_relative.shape) == 4:
                output = []
                for i in range(M_relative.shape[1]):
                    output.append(self.diffusion.eval_init_image(Y[:,i],t,enc,M_relative[:,i]))
                output = torch.stack(output,1)
            else:
                output = self.diffusion.eval_init_image(Y,t,enc,M_relative)
            
            return output

class Unet(nn.Module):
    def __init__(self,d_input=64,num_transformer_layer=2,max_t=10) -> None:
        super().__init__()
        self.max_t = max_t
        self.alpha = 1-torch.arange(0,1,1/(max_t+1))[1:]
        self.alpha_bar = torch.Tensor([self.alpha[:i+1].prod() for i in range(len(self.alpha))])
        self.sigma = torch.concat([torch.zeros(1),torch.sqrt((1-self.alpha_bar[:-1])/(1-self.alpha_bar[1:])*(1-self.alpha[1:]))],dim=-1)

        self.time_embed = nn.Linear(d_input,d_input)

        self.enc_conv0 = nn.Conv2d(4,d_input//8,3,padding=1)

        self.enc_conv1 = nn.Conv2d(d_input//8,d_input//4,5,stride=2,padding=2,bias=False)

        self.enc_conv2 = nn.Conv2d(d_input//4,d_input//2,5,stride=2,padding=2,bias=False)

        self.enc_conv3 = nn.Conv2d(d_input//2,d_input,3,padding=1)

        self.positional_ff = nn.Linear(d_input,9)
        cam = Camera_v2(35,35,32,32)
        self.rays = cam.create(32,32)
        self.softPosEmbbeding =  nn.Linear(6,d_input)
        self.TransformerDecoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)

        self.dec_conv_upscale0 = nn.ConvTranspose2d(d_input,d_input//4,3,stride=2,output_padding=1,padding=1)
        self.dec_conv0 = nn.Conv2d(d_input//2,d_input//4,3,padding=1)

        self.dec_conv_upscale1 = nn.ConvTranspose2d(d_input//4,d_input//8,3,stride=2,output_padding=1,padding=1)
        self.dec_conv1 = nn.Conv2d(d_input//4,4,3,padding=1)

    def pos_encoder(self,tokens,M_relative=None):
        if M_relative is None:
            M_relative = torch.concat([torch.eye(3),torch.zeros(3)[:,None]],dim=-1)[None].repeat(tokens.shape[0],1,1).to(tokens.device)
        else:
            M_relative = M_relative.to(tokens.device)
        positional_token = tokens[:,:1]
        spatial = self.positional_ff(positional_token)
        spatial = spatial - torch.concat([torch.zeros(spatial.shape[0],6).to(tokens.device),M_relative[...,-1]],-1).unsqueeze(1)

        rays = self.rays.unsqueeze(0).repeat(tokens.shape[0],1,1,1)
        cam = transform_cam(rays.to(tokens.device),M_relative)

        b,p_w,p_h,_ = cam.shape
        cam = coordRay(spatial,cam.reshape(b,-1,3)).squeeze(-2)
        cam = self.softPosEmbbeding(cam)

        return cam.reshape(b,p_w,p_h,-1).permute(0,3,1,2)

    def forward(self,x,timesteps,tokens,pos_enc):

        x0 = self.enc_conv0(x)

        x1 = self.enc_conv1(x0)

        x2 = self.enc_conv2(x1)
        
        x3 =  self.enc_conv3(x2) 
        b,d,w,h = x3.shape

        timesteps = self.time_embed(timestep_embedding(timesteps,d,self.max_t).to(tokens.device))
        #pos_enc = self.pos_encoder(tokens,M_relative)
        x3 = x3 + pos_enc + timesteps[:,:,None,None]

        y3 = self.TransformerDecoder(x3.permute(0,2,3,1).reshape(b,-1,d),tokens[:,1:]).permute(0,2,1).reshape(x3.shape)

        y2 = self.dec_conv0(torch.concat([x1,self.dec_conv_upscale0(y3)],dim=1))

        y1 = self.dec_conv1(torch.concat([x0,self.dec_conv_upscale1(y2)],dim=1))

        return y1
    
    def train_predict_noise(self,x0,tokens,M_relative=None):
        self.train()
        x0 = 2*x0-1
        timesteps = torch.randint(0,self.max_t,(x0.shape[0],))
        pos_enc = self.pos_encoder(tokens,M_relative)
        noises = torch.randn_like(x0)
        alpha_bar = self.alpha_bar.gather(dim=0,index=timesteps).to(x0.device)
        x = x0*torch.sqrt(alpha_bar)[:,None,None,None] + noises*torch.sqrt(1-alpha_bar)[:,None,None,None]
        predicted_noises = self.forward(x,timesteps,tokens,pos_enc)
        return predicted_noises,noises
    
    def eval_image(self,tokens,M_relative=None):
        self.eval()
        with torch.no_grad():
            xt = torch.randn(tokens.shape[0],4,128,128).to(tokens.device)
            pos_enc = self.pos_encoder(tokens,M_relative)
            for i in range(self.max_t-1,-1,-1):
                z = torch.randn_like(xt) 
                xt = self.alpha[i]**(-1/2)*(xt - (1-self.alpha[i])/torch.sqrt(1-self.alpha_bar[i])*self.forward(xt,i+torch.zeros(tokens.shape[0]),tokens,pos_enc))+self.sigma[i]*z
        return xt

class Unet_block(nn.Module):
    def __init__(self,d_input=64,num_transformer_layer=2,max_t=10) -> None:
        super().__init__()

        self.max_t = max_t
        self.alpha = 1-torch.arange(0,1,1/(max_t+1))[1:]
        self.alpha_bar = torch.Tensor([self.alpha[:i+1].prod() for i in range(len(self.alpha))])
        self.sigma = torch.concat([torch.zeros(1),torch.sqrt((1-self.alpha_bar[:-1])/(1-self.alpha_bar[1:])*(1-self.alpha[1:]))],dim=-1)
        self.time_embed = nn.Linear(d_input,d_input)

        self.enc_block1 = encoding_block(d_input=4,dim=d_input//4)
        self.enc_block2 = encoding_block(d_input=d_input//4,dim=d_input//2)
        self.enc_last_conv = nn.Conv2d(d_input//2,d_input,3,padding=1)

        self.positional_ff = nn.Linear(d_input,9)
        cam = Camera_v2(35,35,32,32)
        self.rays = cam.create(32,32)
        self.softPosEmbbeding =  nn.Linear(6,d_input)

        self.TransformerSpatial = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_input, nhead=2,batch_first=True),num_layers=num_transformer_layer)

        self.dec_block2 = decoding_block(d_input=d_input,dim=d_input//2)
        self.dec_block1 = decoding_block(d_input=d_input//2,dim=d_input//4)
        self.dec_last_conv = nn.Conv2d(d_input//4,4,1)

    def pos_encoder(self,tokens,M_relative=None):
        if M_relative is None:
            M_relative = torch.concat([torch.eye(3),torch.zeros(3)[:,None]],dim=-1)[None].repeat(tokens.shape[0],1,1).to(tokens.device)
        else:
            M_relative = M_relative.to(tokens.device)
        positional_token = tokens[:,:1]
        spatial = self.positional_ff(positional_token)
        spatial = spatial - torch.concat([torch.zeros(spatial.shape[0],6).to(tokens.device),M_relative[...,-1]],-1).unsqueeze(1)
        rays = self.rays.unsqueeze(0).repeat(tokens.shape[0],1,1,1)
        cam = transform_cam(rays.to(tokens.device),M_relative)

        b,p_w,p_h,_ = cam.shape
        cam = coordRay(spatial,cam.reshape(b,-1,3)).squeeze(-2)
        cam = self.softPosEmbbeding(cam)

        return cam.reshape(b,p_w,p_h,-1).permute(0,3,1,2)

    def forward(self,x,timesteps,tokens,pos_enc):

        x1,res1 = self.enc_block1(x)
        x2,res2 = self.enc_block2(x1)
        x3 = self.enc_last_conv(x2)
        b,d,w,h = x3.shape

        timesteps = self.time_embed(timestep_embedding(timesteps,d,self.max_t).to(tokens.device))

        x3 = x3 + timesteps[:,:,None,None] + pos_enc
        y3 = self.TransformerSpatial(x3.permute(0,2,3,1).reshape(b,-1,d),tokens[:,1:]).permute(0,2,1).reshape(x3.shape)
        
        
        y2 = self.dec_block2(y3,res2)
        y1 = self.dec_block1(y2,res1)
        y = self.dec_last_conv(y1)

        return y
    
    def train_predict_noise(self,x0,tokens,M_relative=None):
        #rand_color = torch.rand((x0.shape[0],3)).to(x0.device)
        #x0[:,:3] += rand_color[:,:,None,None]*x0[:,-1:]
        x0 = 2*x0-1
        self.train()
        timesteps = torch.randint(0,self.max_t,(x0.shape[0],))
        pos_enc = self.pos_encoder(tokens,M_relative)
        noises = torch.randn_like(x0)
        alpha_bar = self.alpha_bar.gather(dim=0,index=timesteps).to(x0.device)
        x = x0*torch.sqrt(alpha_bar)[:,None,None,None] + noises*torch.sqrt(1-alpha_bar)[:,None,None,None]
        predicted_noises = self.forward(x,timesteps,tokens,pos_enc)
        
        return predicted_noises ,noises
    
    def eval_image(self,tokens,M_relative=None):
        self.eval()
        with torch.no_grad():
            xt = torch.randn(tokens.shape[0],4,128,128).to(tokens.device)
            pos_enc = self.pos_encoder(tokens,M_relative)
            for i in range(self.max_t-1,-1,-1):
                z = torch.randn_like(xt)
                xt = self.alpha[i]**(-1/2)*(xt - (1-self.alpha[i])/torch.sqrt(1-self.alpha_bar[i])*self.forward(xt,i+torch.zeros(tokens.shape[0]),tokens,pos_enc))+self.sigma[i]*z
        return (xt+1)/2
    
    def eval_init_image(self,x,t,tokens,M_relative=None):
        self.eval()
        x = 2*x-1
        with torch.no_grad():
            alpha_bar = self.alpha_bar.gather(dim=0,index=t-1+torch.zeros(x.shape[0],dtype=torch.int64)).to(x.device)
            xt = x*torch.sqrt(alpha_bar)[:,None,None,None] +torch.randn_like(x)*torch.sqrt(1-alpha_bar)[:,None,None,None]
            pos_enc = self.pos_encoder(tokens,M_relative)
            for i in range(t-1,-1,-1):
                z = torch.randn_like(xt)
                xt = self.alpha[i]**(-1/2)*(xt - (1-self.alpha[i])/torch.sqrt(1-self.alpha_bar[i])*self.forward(xt,i+torch.zeros(tokens.shape[0]),tokens,pos_enc))+self.sigma[i]*z
        return (xt+1)/2

class encoding_block(nn.Module):
    def __init__(self,d_input=32,dim=64) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(d_input,dim,3,padding=1)
        self.conv_2 = nn.Conv2d(dim,dim,3,padding=1)
        self.pool_1 = nn.MaxPool2d(2,stride=2)
    def forward(self,x):
        x = self.conv_1(x).relu()
        mid = self.conv_2(x).relu()
        y = self.pool_1(mid)
        return y, mid

class decoding_block(nn.Module):
    def __init__(self,d_input=64,dim=32) -> None:
        super().__init__()
        self.up_1 = nn.ConvTranspose2d(d_input,dim,3,stride=2,output_padding=1,padding=1)
        self.conv_1 = nn.Conv2d(dim,dim,3,padding=1)
        self.conv_2 = nn.Conv2d(dim*2,dim,3,padding=1)

    def forward(self,x,residual):
        x = self.up_1(x)
        x = self.conv_1(x).relu()
        mid = torch.concat([x,residual],dim=1)
        y = self.conv_2(mid).relu()
        return y

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


if __name__ == "__main__":
    model = MultiViewModel()
    M_relative = torch.concat([torch.eye(3),torch.zeros(3).unsqueeze(1)],-1).unsqueeze(0).unsqueeze(0).repeat(2,2,1,1)
    X = torch.rand(2,4,128,128)
    Y = torch.rand(2,2,4,128,128)
    diffusion = MultiViewDiffusionModel()
    print(diffusion(X,M_relative).shape)
    print(diffusion.train_loop(X,Y,M_relative)[0].shape,diffusion.train_loop(X,Y,M_relative)[1].shape)
    #print(model(input[0].unsqueeze(0),M_relative[0].unsqueeze(0)))
    output = model(X,M_relative)
    print(output.shape)
    #print(timestep_embedding(torch.IntTensor([3]),128,3))



