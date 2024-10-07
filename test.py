#CONFIGURACIÓN
from Metricas import get_DVH
import torch
import argparse
import os.path
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset, random_split
import json
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import statistics
from torcheval.metrics import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim

#definimos variables globales 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 128
CHANNELS_IMG = 3
dim_3D = [128,128,128]
dim = [dim_3D[0]*2+1, dim_3D[1], dim_3D[2]]

#Pedimos los parametros
parser = argparse.ArgumentParser(description='Parametros para entrenamiento')
parser.add_argument('--LEARNING_RATE', metavar='LEARNING_RATE', type=float, default=2e-4, help='LEARNING_RATE')
parser.add_argument('--BATCH_SIZE', metavar='BATCH_SIZE', type=int, default=2, help='BATCH_SIZE')
parser.add_argument('--L1_LAMBDA', metavar='L1_LAMBDA', type=int, default=135, help='L1_LAMBDA')
parser.add_argument('--NUM_EPOCHS', metavar='NUM_EPOCHS', type=int, default=212, help='NUM_EPOCHS')
parser.add_argument('--BETA1', metavar='BETA1', type=float, default=0.5, help='BETA1')
parser.add_argument('--BETA2', metavar='BETA2', type=float, default=0.999, help='BETA2')
parser.add_argument('--LOAD_MODEL', metavar='LOAD_MODEL', type=bool, default=0, help='LOAD_MODEL')
parser.add_argument('--SAVE_MODEL', metavar='SAVE_MODEL', type=bool, default=1, help='SAVE_MODEL')
parser.add_argument('--MODO', metavar='MODO', type=str, default="Flip", help='MODO DEL DATA AUG "FandP" , "Flip" , "Crop"')
parser.add_argument('--FOLDER_OUT', metavar='FOLDER_OUT', type=str, help='Directorio de salida')
parser.add_argument('--TEST_DIR', metavar='TEST_DIR', type=str, help='Directorio del conjunto de test')
parser.add_argument('--MODELO_DIR', metavar='MODELO_DIR', type=str, help='Directorio del modelo entrenado a cargar')
args = parser.parse_args()

LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
L1_LAMBDA = args.L1_LAMBDA
NUM_EPOCHS = args.NUM_EPOCHS
BETA1 = args.BETA1
BETA2 = args.BETA2
LOAD_MODEL = args.LOAD_MODEL
SAVE_MODEL = args.SAVE_MODEL
MODO = args.MODO
TRAIN_DIR = args.TRAIN_DIR
TEST_DIR = args.TEST_DIR
FOLDER_OUT = args.FOLDER_OUT
MODELO_DIR = args.MODELO_DIR

BETAS = (BETA1,BETA2)


#TRANSFORMACIONES
   
# definjcion de las transformaciones para ct , dosis y data augmentation 

def get_transform_dosis(dim_3D):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda dose_val : transformacion_valores_tensores_dosis(dose_val)))
    transform_list.append(transforms.Lambda(lambda dose_val : normalized(dose_val,0.5,0.5)))
    transform_list.append(transforms.Lambda(lambda dose_val : trilinear_interpolation(dose_val, dim_3D))) # que la medida sea parametro
    return transforms.Compose(transform_list)
    
# compose compone varias funciones juntas, admite una lista de transformaciones 

def transformacion_valores_tensores_dosis(dose_val):
    dose_val = dose_val.numpy()
    if dose_val.dtype == np.uint8 or dose_val.dtype == np.uint16:
        dose_val= dose_val / 256
    dose_val = torch.from_numpy(dose_val).float()

    return dose_val

def normalized(img, mean, std):
    img = img.unsqueeze(0)
    max_img = img.max()
    img = img/max_img
    img = img.sub_(mean).div_(std)
    return img

def get_transform_ct(dim_3D):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda ct_image : transformacion_valores_tensores_ct(ct_image)))
    transform_list.append(transforms.Lambda(lambda ct_image : normalize3d(ct_image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))))
    transform_list.append(transforms.Lambda(lambda ct_image : trilinear_interpolation(ct_image, dim_3D)))

    return transforms.Compose(transform_list)
    

def transformacion_valores_tensores_ct(ct_image):
    ct_image = ct_image.numpy()
    if ct_image.dtype == np.uint8 or ct_image.dtype == np.uint16:
        ct_image = ct_image / 256
    ct_image = torch.from_numpy(ct_image).float()

    return ct_image

def normalize3d(img, mean, std):
    ''' Normalizes a voxel Tensor (C x D x H x W) by mean and std. '''
    if len(mean) < 3 or len(std) < 3:
        raise TypeError('not enough means and standard deviations')
    for t, m, s in zip(img, mean, std):
        t.sub_(m).div_(s)
    return img

def trilinear_interpolation(tensor, target_size):
    # Redimensionar usando interpolación trilineal con el tamaño objetivo
    tensor_resized = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)
    return tensor_resized

def get_transform_data_augmentation(dim, modo = None):

    '''Esta funcion por me de devuelve las transformaciones que hay que aplicar para el aumento de datos, se puede elegir el modo pero por defecto es ramdom'''
    
    random_aug = random.randint(0, 5)
    # print('random_aug', random_aug)
    
    if modo == 'FandP':
        random_aug = 0
    elif modo == 'Flip':
        random_aug = 2
    elif modo == 'Crop':
        random_aug = 4
 
    # flip and crop
    if random_aug == 0 or random_aug == 1:
        print('FandP')
        transform = transforms.Compose(
            [transforms.RandomCrop(100),
             transforms.RandomHorizontalFlip(1), 
             transforms.Lambda(lambda ct_image : trilinear_interpolation(ct_image, dim))] )
    # flip
    elif random_aug == 2 or random_aug == 3: 
        print('Flip')
        transform = transforms.RandomHorizontalFlip(1)
        
    #crop
    elif random_aug == 4 or random_aug == 5: 
        print('Crop')
        transform = transforms.Compose(
            [transforms.RandomCrop(100),
             transforms.Lambda(lambda ct_image : trilinear_interpolation(ct_image, dim))] )
    # print(random_aug)
    return transform

#----------------------------------------------------------------------------------------------------------------

#DATA SET Y DATA LOADER

def make_dataset(dir, filetypes):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
    
class CustomImageDataset(data.Dataset):

    def __init__(self, ruta, ct_transform, target_transform, transform_data_augmentation):
        self.dir_AB = ruta
        self.transform_ct = ct_transform
        self.target_transform = target_transform 
        self.transform_data_augmentation = transform_data_augmentation
        slice_filetype = ['.pt'] 
        self.AB_paths = sorted(make_dataset(self.dir_AB, slice_filetype))
        self.longuitud = len(self.AB_paths)
    
    def __getitem__(self, index):
        
        AB_path = self.AB_paths[index]
        #Cargar el archivo utilizando el nombre de la variable
        loaded = torch.load(AB_path)
        # Acceder a los tensores por separado
        tensor_ct = loaded["imagen"]
        dose_val = loaded["dosis"]
        max_dosis = torch.max(dose_val)

        pixel_spacing = 0.9765619999999999
        space_betweent_slices = 2.5

        #Contamos los pixeles

        ptv_mask = torch.zeros_like(dose_val)
        recto_mask = torch.zeros_like(dose_val)
        vejiga_mask = torch.zeros_like(dose_val)
    
        ptv_mask = tensor_ct[1] == 1 
        
        recto_mask = tensor_ct[0] == 1
     
        vejiga_mask = tensor_ct[0] == 0.75

        pixeles_ptv = torch.sum(ptv_mask).item()
        pixeles_recto = torch.sum(recto_mask).item()
        pixeles_vejiga = torch.sum(vejiga_mask).item()

        volumen_ptv = (pixeles_ptv*(pixel_spacing**2)*space_betweent_slices)/1000
        #print("volumen_ptv", volumen_ptv)
        
        volumen_recto = (pixeles_recto*(pixel_spacing**2)*space_betweent_slices)/1000
        #print("volumen_recto", volumen_recto)
        
        volumen_vejiga = (pixeles_vejiga*(pixel_spacing**2)*space_betweent_slices)/1000
        #print("volumen_vejiga", volumen_vejiga)

        volumenes = (volumen_ptv, volumen_recto, volumen_vejiga)
        
        #Transformaciones
        
        tensor_ct = self.transform_ct(tensor_ct)
        dose_val = self.target_transform(dose_val)

        return tensor_ct, dose_val, max_dosis, volumenes
            

    def __len__(self):
        return self.longuitud

    def name(self):
        return 'CustomImageDataset'

#------------------------------------------------------------------------------------------------------------------------

#GENERADOR

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm3d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2), 
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down 
        
        
        # Initialization
        for m in self.conv:
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu' if act == 'relu' else 'leaky_relu')
        
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Block2(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, act = 'relu', use_dropout = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm3d(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2), 
        )

        # Initialization
        for m in self.conv:
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu' if act == 'relu' else 'leaky_relu')
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down 
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generador(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size= 4, stride= 2, padding= 1, padding_mode="reflect"), # 1-3-128**3  -- 1-64-64**3    
            nn.LeakyReLU(0.2),
        )
        
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False) # # 1-64-64**3  -- 1-128-32**3 
        
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False) # 1-128-32**3 -- 1-256-16**3
        
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False) #256-512  #  1-256-16**3 -- 1-512-8**3
        
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) #512-512 1-512-8**3 -- 1-512-4**3
        
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) #512-512  1-512-4**3 -- 1-512-2**3
        
        #self.down6 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False) #512*512  1-512-2**3  -- 1-512-1**3
        self.down6 = nn.Sequential(
            nn.Conv3d(features * 8, features * 8, kernel_size = 3, stride = 1, padding = 1),  #512-512  1-512-2**3 --1-512-2**3
            nn.LeakyReLU(0.2)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(features * 8, features * 8, kernel_size = 3, stride = 1, padding = 1),  #512-512  1-512-2**3 -- 1-512-2**3
            nn.ReLU()
        )

        self.up1 = Block2(features * 8, features * 8, down=False, act="relu", use_dropout=True) #512-512 1-512-2**3 -- 1-512-2**3

        #self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True) #512-512 1-512-2**3 -- 1-512-4**3
        
        self.up2 = Block2(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True) #1024-512
        
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True) #1024-512
        
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False)#1024-512
        
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False)#1024-256
        
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False) #512-128
        
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False) # 256-64
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(features * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Initialization
        for m in self.initial_down:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        for m in self.down6:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        for m in self.bottleneck:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        for m in self.final_up:
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_normal_(m.weight)
        

    def forward(self, x):
        
        d1 = self.initial_down(x)
       
        d2 = self.down1(d1)
       
        d3 = self.down2(d2)
        
        d4 = self.down3(d3)
       
        d5 = self.down4(d4)
        
        d6 = self.down5(d5)
      
        d7 = self.down6(d6)  #2
      
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)    #2
        up2 = self.up2(torch.cat([up1, d7], 1)) #4
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

#------------------------------------------------------------------------------------------------------------------------------

#UTILS

def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
   

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return None

#-------------------------------------------------------------------------------------------------------------------------

def main ():
    
 
    gen = Generador(in_channels=3, features=64).to(DEVICE)

    #inicializamos los optimizadoes
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)
    
    load_checkpoint(MODELO_DIR, gen, opt_gen, LEARNING_RATE)

    # definomos las transformaciones para los datos 
    ct_transform = get_transform_ct(dim_3D)
    target_transform = get_transform_dosis(dim_3D)
    
    #definimos el modo de dataaug:  
    transform_data_augmentation = get_transform_data_augmentation(dim, MODO) 
    
    #Creamos dataset de testeo
    dataset_test = CustomImageDataset(TEST_DIR, ct_transform, target_transform, transform_data_augmentation)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    gen.eval()

    with torch.no_grad():
        for i, (x, y, z, v) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_fake = gen(x)
            
            #crear el directorio de salida si no existe
            os.makedirs(FOLDER_OUT, exist_ok=True)
            
            individual_folder = os.path.join(FOLDER_OUT, f"{i}")
            os.makedirs(individual_folder, exist_ok=True)
            output_file = os.path.join(individual_folder, f"{i}.pt")
            
            variable = { "ct": x, "dose": y, "pred": y_fake, "max_dosis": z, "volumenes": v}

            torch.save(variable, output_file)

    gen.train()

if __name__ == '__main__':
    main()












    