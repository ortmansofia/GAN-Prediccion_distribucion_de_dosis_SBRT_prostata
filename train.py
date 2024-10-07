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
parser.add_argument('--TRAIN_DIR', metavar='TRAIN_DIR', type=str, help='Directorio del conjunto de entrenamiento')
parser.add_argument('--TEST_DIR', metavar='TEST_DIR', type=str, help='Directorio del conjunto de test')
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

BETAS = (BETA1,BETA2)

#creamos carpetas para guardar 

output_folder = r"/users/sortman/PI-GAN-2/Nuestro_Pix2Pix/Modelos_Finales"

folder = os.path.join(output_folder, f"LEARNING_RATE{LEARNING_RATE},BATCH_SIZE{BATCH_SIZE},L1_LAMBDA{L1_LAMBDA},NUM_EPOCHS{NUM_EPOCHS},BETA1{BETA1},BETA2{BETA2},MODO{MODO}")

os.makedirs(folder, exist_ok=True)

#fijamos las semillas 
random.seed(42)
torch.manual_seed(42)

#------------------------------------------------------------------------------------------------
# definimos funcion que obtiene y guarda las prediccciones con el conjunto test, tambien guardamos los hiperparametros

def get_resultados(dic_perdida_epoc_gen_train, dic_perdida_epoc_disc_train, folder, gen, test_loader):

    #Generamos las predicciones
    gen.eval()

    with torch.no_grad():
        for i, (x, y, z) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            y_fake = gen(x)

            individual_folder = os.path.join(folder, f"{i}")
            os.makedirs(individual_folder, exist_ok=True)
            output_file = os.path.join(individual_folder, f"{i}.pt")
            
            variable = { "ct": x, "dose": y, "pred": y_fake, "max_dosis": z }

            torch.save(variable, output_file)

    gen.train()
    
    # Crear gráficos de pérdidas
    epochs = list(dic_perdida_epoc_gen_train.keys())  # Suponiendo que las épocas están en las claves
    
    G_losses_train = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in dic_perdida_epoc_gen_train.values()]
    D_losses_train = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in dic_perdida_epoc_disc_train.values()]
    
    # Crear el subplot
    fig, axs = plt.subplots(1, 1, figsize=(16, 12))
    
   # Gráfica de entrenamiento en la parte superior
    axs.plot(epochs, G_losses_train, label="Generator Loss Train")
    axs.plot(epochs, D_losses_train, label="Discriminator Loss Train")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")
    axs.set_title("Training Losses")
    axs.legend()
    axs.set_xticks(epochs[::5])  # Mostrar cada 5 épocas en el eje 

    # Guardar la gráfica
    output_path = os.path.join(folder, "loss_plot.png")
    plt.savefig(output_path)
    plt.close()

    # Calcular el promedio de los valores en cada diccionario
    average_perdida_epoc_gen_train = sum(dic_perdida_epoc_gen_train.values()) / len(dic_perdida_epoc_gen_train)
    average_perdida_epoc_disc_train = sum(dic_perdida_epoc_disc_train.values()) / len(dic_perdida_epoc_disc_train)
    
    # Agregar el promedio al diccionario
    dic_perdida_epoc_gen_train["average"] = average_perdida_epoc_gen_train
    dic_perdida_epoc_disc_train["average"] = average_perdida_epoc_disc_train
    
    # Almacenar los diccionarios en un archivo JSON
    resultados_train = {
        "dic_perdida_epoc_gen_train": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in dic_perdida_epoc_gen_train.items()},
        "dic_perdida_epoc_disc_train": {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in dic_perdida_epoc_disc_train.items()},
    }

    ruta_json = os.path.join(folder, "resultados_train.json")
    with open(ruta_json, "w") as f:
        json.dump(resultados_train, f, indent=4)

  
    #Guardamos hiperparametros
    ruta_txt = os.path.join(folder, "hiperparametros.txt")
    try:
         with open(ruta_txt, "w") as file:
            file.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
            file.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            file.write(f"L1_LAMBDA: {L1_LAMBDA}\n")
            file.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
            file.write(f"BETA1: {BETA1}\n")
            file.write(f"BETA2: {BETA2}\n")  
            file.write(f"MODO: {MODO}\n")
    except Exception as e:
        print(f"Error: {e}")

    return None

#----------------------------------------------------------------------------------------------------

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
        
        tensor_ct = self.transform_ct(tensor_ct)
        dose_val = self.target_transform(dose_val)
        
        random_selector = random.randint(0, 1)
        # print('random_selector:', random_selector)
        
        if random_selector == 0: 
            
            tensor_transform = torch.zeros([3, 257, 128, 128])
            tensor_transform[:,0:128,:,:] = tensor_ct 
            tensor_transform[0,129:257,:,:] = dose_val 
            
            tensor_transform = self.transform_data_augmentation(tensor_transform)
            tensor_ct = tensor_transform[:,0:128,:,:]
            dose_val = (tensor_transform[0,129:257,:,:]).unsqueeze(0)

        return tensor_ct, dose_val, max_dosis
            

    def __len__(self):
        return self.longuitud

    def name(self):
        return 'CustomImageDataset'

#--------------------------------------------------------------------------

#DISCRIMINADOR

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode= 'reflect'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2), 
        )

        # Initialization
        for m in self.conv:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        
    def forward(self, x):
        return self.conv(x)

class Discriminador(nn.Module):
    def __init__(self, in_channels = 1, features = [64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels+3, features[0], kernel_size= 4, stride=2, padding=1, padding_mode='reflect'), 
            nn.LeakyReLU(0.2), 
        )

        # Initialization
        for m in self.initial:
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        
        layers = list()
        in_channels = features[0]

        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride= 1 if feature == features[-1] else 2),
            )
            in_channels = feature

        self.model = nn.Sequential(*layers)
        
        self.final = nn.Sequential(
            nn.Conv3d(in_channels,1, kernel_size= 4, stride=1, padding=1, padding_mode='reflect'), 
        )

        # Initialization for final layer
        for m in self.final:
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
        

    def forward(self, x, y): 
        x1 = torch.cat([x,y], dim =1)
        x1 = self.initial(x1)
        x1 = self.model(x1)
        return self.final(x1)

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

#TRAIN

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    
    loop = tqdm(loader, leave=True)

    total_D_loss = 0
    total_G_loss = 0
    num_batches = len(loader)
    
    for idx, (x, y, z) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        
        # Train Discriminator
        y_fake = gen(x)
        D_real = disc(x, y) #aca devuelve una matriz con las prob de que sea real
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
            
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()

        d_scaler.step(opt_disc)
        d_scaler.update()

        total_D_loss += D_loss.item()

        # Train generator
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()

        g_scaler.step(opt_gen)
        g_scaler.update()

        total_G_loss += G_loss.item()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    avg_D_loss = total_D_loss / num_batches
    avg_G_loss = total_G_loss / num_batches

    return avg_D_loss, avg_G_loss
        
def main():

    # definomos las transformaciones para los datos 
    ct_transform = get_transform_ct(dim_3D)
    target_transform = get_transform_dosis(dim_3D)
    
    #definimos el modo de dataaug:  
    transform_data_augmentation = get_transform_data_augmentation(dim, MODO) 
    
    #definimos la ruta donde se encuentran los tensores de los pacientes: 
    dataset_train = CustomImageDataset(TRAIN_DIR, ct_transform, target_transform, transform_data_augmentation)

    #Creamos dataset de testeo
    dataset_test = CustomImageDataset(TEST_DIR, ct_transform, target_transform, transform_data_augmentation)
    
    dic_perdida_epoc_gen_train = dict()
    dic_perdida_epoc_disc_train = dict()
  
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
        
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
        
    #generamos los modelos
    disc = Discriminador(in_channels=1).to(DEVICE)
    gen = Generador(in_channels=3, features=64).to(DEVICE)
    
    #inicializamos los optimizadoes
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)
    
    # inicializamos las funciones de perdidas
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    for epoch in range(NUM_EPOCHS):
            
        D_loss_train, G_loss_train = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,)
    
        dic_perdida_epoc_gen_train[epoch ] = G_loss_train
        dic_perdida_epoc_disc_train[epoch ] = D_loss_train

        print(f"en la {epoch }: D_loss_train = {D_loss_train}")
        print(f"en la {epoch }: G_loss_train = {G_loss_train}")
    
        if SAVE_MODEL and epoch % 10 == 0:
            CHECKPOINT_GEN = os.path.join(folder, "model_gen.pth")
            CHECKPOINT_DISC = os.path.join(folder, "model_disc.pth")
            save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, CHECKPOINT_DISC)

    #Guardamos el modelo en la ultima epoca
    CHECKPOINT_GEN = os.path.join(folder, "model_gen_final.pth")
    CHECKPOINT_DISC = os.path.join(folder, "model_disc_final.pth")
    save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
    save_checkpoint(disc, opt_disc, CHECKPOINT_DISC)

    #Llamamos función para graficar perdidas
    get_resultados(dic_perdida_epoc_gen_train, dic_perdida_epoc_disc_train, folder, gen, test_loader)

if __name__ == '__main__':
    main()