import torch
from PIL import Image
from torch.utils.data import Dataset
import glob,os
import torchvision
import random,json
from camera import relative_M

class CLEVR(Dataset):
    def __init__(self,path_image,resolution=(112,112)) -> None:
        super().__init__()
        self.list_path = glob.glob(path_image+"\*.png")
        self.resize = torchvision.transforms.Resize(resolution)
    def __len__(self):
        return len(self.list_path)
    
    def __getitem__(self, index):
        image = torchvision.io.read_image(self.list_path[index])
        image = image[:-1,:,80:-80]
        image = self.resize(image)

        return image/255

# work in progress
class GSO(Dataset):
    def __init__(self,path,nb_images=8,output_image=2) -> None:
        super().__init__()
        self.path_dataset = path
        self.object_pathes = os.listdir(path)
        self.nb_images = nb_images
        self.output_image = output_image
    
    def __len__(self):
        return len(self.object_pathes)
    
    def __getitem__(self, index):
        output = {}
        path = self.path_dataset + '/' + self.object_pathes[index]
        inds = [i for i in range(self.nb_images)]
        id_x = random.sample(inds,1)[0]
        ids_y = random.sample(inds, self.output_image)
        
        output['X'] = load_rgba(path+'/'+str(id_x)+'.png').permute(2,0,1)/255
        output['Y'] = torch.stack([load_rgba(path+'/'+str(i)+'.png').permute(2,0,1)/255 for i in ids_y],0)
        f = open(path+'/' + 'metadata.json') 
        metadata = json.load(f) 

        x_cam_world_matrix = torch.Tensor(metadata['cam_world_matrix'][id_x])[:3]
        y_cam_world_matrix = [torch.Tensor(metadata['cam_world_matrix'][i])[:3] for i in ids_y]

        output['relative_M']  = torch.stack([relative_M(x_cam_world_matrix,i) for i in y_cam_world_matrix],0)

        
        return output
import numpy as np
def load_rgba(path):
    img = Image.open(path)
    pix = np.array(img)
    return torch.Tensor(pix)

if __name__=="__main__":
    data = CLEVR(r"C:\Users\Admin\Documents\Dataset\CLEVR_v1.0\CLEVR_v1.0\images\train")
    data = GSO(r"C:\Users\Admin\Documents\Dataset\GSO_image")
    output = data.__getitem__(0)
    print(output)
    