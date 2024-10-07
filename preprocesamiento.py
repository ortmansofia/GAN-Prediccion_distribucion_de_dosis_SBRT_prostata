import os
import glob
import torch
import argparse
from dicompylercore import dicomparser as dp
from torchvision import transforms
import torch.nn.functional as F
import cv2
import re
import numpy as np


def get_rt_parser(ruta):
    
    ''' Esta funcion recibe una ruta con archivos DICOM de un paciente y te devuelve un
    diccionario con los objetos instanciados de la clase DicomParser diferenciados por su tipo
    '''
    
    # Crear diccionario para almacenar los rt_parsers
    rt_parsers = {'RTSTRUCT': None, 'RTDOSE': None, 'CT': []}

    # Obtener la lista de archivos dicom dentro de la subcarpeta del paciente
    archivos_dicom = glob.glob(os.path.join(ruta, "*.dcm*"))

    # Procesar cada archivo DICOM
    for archivo_dicom in archivos_dicom:
        # Crear el path completo al archivo DICOM
        ruta_dicom = os.path.join(ruta, archivo_dicom)
        # Crear un objeto DicomParser para el archivo DICOM
        dicom_parser = dp.DicomParser(ruta_dicom)
        # Obtenemos la informacion de ese objeto 
        series_info = dicom_parser.GetSeriesInfo()

        # Verificar el atributo 'Modality' 
        if series_info['modality'] == 'CT':
            rt_parsers['CT'].append(dicom_parser)
        elif series_info['modality'] == 'RTSTRUCT':
            rt_parsers['RTSTRUCT'] = dicom_parser
        elif series_info['modality'] == 'RTDOSE':
            rt_parsers['RTDOSE'] = dicom_parser

    return rt_parsers

def detectar_etiqueta(texto, etiqueta):
    '''Esta funcion utiliza expresiones regulares para reconocer las etiquetas de las estructuras'''

    patrones = [
        r'\b{}\b'.format(etiqueta),
        r'\b[^\s]*{}[^\s]*\b'.format(etiqueta),  
    ]
    for patron in patrones:
        coincidencias = re.findall(patron, texto, flags=re.IGNORECASE)
        if coincidencias:
            return coincidencias[0] 
            
    return None

def get_dic_posiciones(rt_parsers): 
    
    ''' Esta funcion te devuelve la posicion en la que se encuentra la estructura ordenadas en un diccionario'''
    
    dic_posiciones = dict()
    structure_parser = rt_parsers['RTSTRUCT']
    structures = structure_parser.GetStructures()
    
    for referencia in ['PTV', 'vejiga', 'recto', 'cuerpo']:
        aux = 0
        for posicion, dic_info in structures.items():
            match = detectar_etiqueta(dic_info['name'], referencia)
            if match is not None: 
                dic_posiciones[referencia] = posicion
                aux = 1
            if aux == 1: break
             
    return dic_posiciones, structure_parser

def get_coordenadas(dic_posiciones, structure_parser):
    
    ''' Esta funcion te devuelve las coordenadas de las estructuras'''
    
    dic_coordenadas = dict()
    dic_coordenadas_ok = dict()

    for etiqueta, posicion in dic_posiciones.items():
        coordenadas = structure_parser.GetStructureCoordinates(posicion) 
        dic_coordenadas[etiqueta] = coordenadas

    for etiqueta, coordenadas_sucias in dic_coordenadas.items():
        dic_aux = dict()  # Inicializar dic_aux en cada iteración del bucle exterior
        for z, lista_diccionarios in coordenadas_sucias.items():
            data = lista_diccionarios[0]['data']
            dic_aux[z] = data
        dic_coordenadas_ok[etiqueta] = dic_aux 

    return dic_coordenadas_ok

def get_tensor_ct(rt_parsers):

    '''Con esta funcion obtenemos el tensor RGB con la ct ubicada en el canal B'''
    
    informacion_cortes = dict()
    ct_parsers = rt_parsers['CT']
    
    for ct_parser in ct_parsers:
        coordenada_z = ct_parser.GetImageLocation()
        datos_ct = ct_parser.GetImageData()
        imagen_ct = ct_parser.GetImage()

        # Acceder a las filas y columnas
        filas = datos_ct['rows']
        columnas = datos_ct['columns']

        position_ct = datos_ct['position']
        x_ct = position_ct[0]
        y_ct = position_ct[1]

        pixel_spacing_ct = datos_ct['pixelspacing']

        # Almacenar la información en el diccionario
        informacion_cortes[coordenada_z] = {'coordenada_z': coordenada_z, 'imagen_ct': imagen_ct}
        
    # Obtener las coordenadas Z ordenadas
    coordenadas_z_ordenadas = sorted(informacion_cortes.keys())
    print("coordenadas_z_ordenadas", coordenadas_z_ordenadas)
    profundidad = len(coordenadas_z_ordenadas)
    
    # Crear un tensor RGB 
    tensor_rgb = torch.zeros((profundidad, filas, columnas, 3), dtype=torch.float32)

    # Crear un objeto ToTensor para convertir la imagen a tensor
    to_tensor = transforms.ToTensor()
    
    for i, coordenada_z in enumerate(coordenadas_z_ordenadas):
        # Acceder a la imagen CT del diccionario
        imagen_ct = informacion_cortes[coordenada_z]['imagen_ct']
    
        # Convertir la imagen a tensor
        tensor_imagen_ct = to_tensor(imagen_ct)
    
        # Asignar la imagen CT al canal B del tensor RGB
        tensor_rgb[i, :, :, 2] = tensor_imagen_ct[0]  

    return tensor_rgb, filas, columnas, coordenadas_z_ordenadas, datos_ct, informacion_cortes, x_ct, y_ct, pixel_spacing_ct

def get_tensor_estructuras(dic_coordenadas, tensor_rgb, filas, columnas, coordenadas_z_ordenadas):

    '''Con esta funcion obtenemos el tensor RGB con las estructuras colocadas en su canal correspondiente'''
    
    dic_mascaras = dict()
    dic_coordenadas_sinz_estructuras = dict()
    
    # vamos a preparar las coordenadas para conseguir las mascaras
    for estructura, coordenadas in dic_coordenadas.items():
        coordenadas_sinz = dict()
        
        # Iterar sobre los cortes y coordenadas
        for corte, coordenadas_conz in coordenadas.items():
            # Convertir a numpy array
            coordenadas_conz = np.array(coordenadas_conz, dtype=float)
            coordenadas_sin_z = coordenadas_conz[:, :2]
            # guardar las coordenadas por su z
            coordenadas_sinz[corte] = coordenadas_sin_z
            
        dic_coordenadas_sinz_estructuras[estructura] = coordenadas_sinz

    # vamos a formar las mascaras de las estructuras
    for estructura, coordenadas_sinz in dic_coordenadas_sinz_estructuras.items():
        
        dic_mascaras_aux = dict()
        
        for corte, coordenadas in coordenadas_sinz.items():
            # Centrar la figura (ajustar al centro de la imagen)
            coordenadas[:, :2] += [columnas // 2, filas // 2]
            coordenadas_ajustadas = (coordenadas).astype(np.int32)
            # Crear una imagen en blanco
            img = np.zeros((filas, columnas), dtype=np.uint8)
            
            # Dibujar el polígono en la imagen
            cv2.fillPoly(img, [coordenadas_ajustadas], 255)

            #guardamos
            dic_mascaras_aux[corte] = img
            
        dic_mascaras[estructura] = dic_mascaras_aux

    # Crear un objeto ToTensor para convertir la imagen a tensor
    to_tensor = transforms.ToTensor()
    
    # vamos a cargar las mascaras en el tensor
    for estructuras, dic_masc in dic_mascaras.items():
       
        for corte, mascara in dic_masc.items(): 
            corte_2 = round(float(corte), 0)
            for i, coordenada_z in enumerate(coordenadas_z_ordenadas):
                coordenada_z = round(float(coordenada_z), 0)
                if estructuras == "PTV" and corte_2 == coordenada_z:
                    #Convertir la imagen a tensor
                    tensor_mascara = to_tensor(mascara)
                    # Asignar la mascara al canal G del tensor RGB
                    tensor_rgb[i, :, :, 1] = tensor_mascara
                    
                if estructuras == "recto" and corte_2 == coordenada_z:
                    # Convertir la imagen a tensor
                    tensor_mascara = to_tensor(mascara)
                    # Asignar la imagen CT al canal B del tensor RGB
                    tensor_rgb[i, :, :, 0] = tensor_rgb[i, :, :, 0] + tensor_mascara
                    
                if estructuras == "vejiga" and corte_2 == coordenada_z:
                    # Convertir la imagen a tensor
                    tensor_mascara = to_tensor(mascara)
                    # Asignar la imagen CT al canal B del tensor RGB
                    tensor_rgb[i, :, :, 0] = tensor_rgb[i, :, :, 0] + tensor_mascara*0.75
                         
    return tensor_rgb, dic_coordenadas_sinz_estructuras, dic_mascaras
    

def get_tensor_dosis(rt_parsers, coordenadas_z_ordenadas):

    ''' Con esta funcion obtenemos el tensor de la dosis'''
    
    dic_dosis = dict()
    dic_dosis_array = dict()
    profundidad = len(coordenadas_z_ordenadas)

    # Crear un objeto ToTensor para convertir la imagen a tensor
    to_tensor = transforms.ToTensor()

    dosis_parser = rt_parsers['RTDOSE']

    data_dosis = dosis_parser.GetDoseData()
    
    dose_grid_scaling = data_dosis['dosegridscaling']
    
    position_dosis = data_dosis['position']
    x_dosis = position_dosis[0]
    y_dosis = position_dosis[1]

    pixel_spacing_dosis = data_dosis['pixelspacing']
    
    for i, coordenada_z in enumerate(coordenadas_z_ordenadas):
        dosis_array = np.array(dosis_parser.GetDoseGrid(coordenada_z), dtype=float)
        dic_dosis_array[coordenada_z] = dosis_array
        shape = dosis_array.shape
        filas = shape[0]
        columnas = shape[1]
        tensor = to_tensor(dosis_array)
        dic_dosis[i] = tensor

    # Crear un tensor
    tensor_dosis = torch.zeros((profundidad, filas, columnas), dtype=torch.float32)
    for i, tensord in dic_dosis.items():
        tensor_dosis[i, :, :] = tensord

    tensor_dosis = tensor_dosis*dose_grid_scaling 
    
    return tensor_dosis, dic_dosis_array, x_dosis, y_dosis, pixel_spacing_dosis

def get_tensor_dosis_interpolado(tensor_dosis, pixel_spacing_dosis, pixel_spacing_ct):

    '''Con esta funcion interpolamos el tensor de la dosis'''
    
    tensor_original = tensor_dosis
    
    factor_escala = pixel_spacing_dosis[0] / pixel_spacing_ct[0]
    
    # Utilizar F.interpolate para cambiar el espaciado de píxeles
    tensor_interpolado = F.interpolate(tensor_original, scale_factor=factor_escala, mode='nearest')
    tensor_interpolado = tensor_interpolado.permute(0,2,1)
    tensor_interpolado = F.interpolate(tensor_interpolado, scale_factor=factor_escala, mode='nearest')
    tensor_interpolado = tensor_interpolado.permute(0,2,1) #ACA ESTABA MAL
    
    # Verificar la nueva forma y el nuevo espaciado de píxeles
    nueva_forma = tensor_interpolado.size()
    nuevo_pixel_spacing = pixel_spacing_dosis[0]/factor_escala
    #print('nuevo_pixel_spacing:' ,nuevo_pixel_spacing)
    return tensor_interpolado

def get_tensor_recortado(tensor_rgb, dic_coordenadas_sinz_estructuras, tensor_interpolado, x_dosis, y_dosis, x_ct, y_ct, pixel_spacing_ct):

    ''' Con esta funcion obtenemos los tensor de dosis y RGB  recortados sin fondo y correlacionados espacialmente'''
    
    mascaras_rectan = dict()
    lista_x = []
    lista_y = []
    lista_xw = []
    lista_yh = []

    size_interp = tensor_interpolado.size()
    
    filas_interp = size_interp[1]
    columnas_interp = size_interp[2]
    profundidad_interp = size_interp[0]

    size_rgb = tensor_rgb.size()
   
    filas_rgb = size_rgb[1]
    columnas_rgb = size_rgb[2]
    profundidad_rgb = size_rgb[0]

    profundidad_total = profundidad_interp + profundidad_rgb + 1
    
    tensor_recorte = torch.zeros([profundidad_total, filas_rgb, columnas_rgb, 3])

    tensor_recorte[0:profundidad_rgb, :,:,:] = tensor_rgb

    lim_x = (x_ct - x_dosis)/pixel_spacing_ct[0]  # aca faltaria ver cuantos pixeles (multiplicando por el dcm)
    # print(x_ct)
    # print(x_dosis)
    # print(pixel_spacing_ct)
    lim_x = int(-lim_x)
    lim_x2 = int(lim_x + columnas_interp)
    # print(lim_x,lim_x2)
    lim_y = (y_ct - y_dosis)/pixel_spacing_ct[0]
    lim_y = int(-lim_y)
    lim_y2 = int(lim_y + filas_interp)
    # print(lim_y,lim_y2)

    # print(tensor_interpolado.size())
    # print(tensor_rgb.size())
    
    tensor_recorte[profundidad_rgb+1:profundidad_total, lim_y:lim_y2,lim_x:lim_x2,0] =  tensor_interpolado

    for corte, coordenadas in dic_coordenadas_sinz_estructuras["cuerpo"].items():
        coordenadas1 = [coordenadas.astype(int)]
        x, y, w, h = cv2.boundingRect(coordenadas1[0])
        lista_x.append(x)
        lista_y.append(y)
        lista_xw.append(x + w)
        lista_yh.append(y + h)

    min_x = min(lista_x)
    min_y = min(lista_y)
    max_xw = max(lista_xw)
    max_yh = max(lista_yh)
    
    tensor_recortado = tensor_recorte[:, min_y - 4 : max_yh + 4, min_x - 4 : max_xw + 4, :]

    tensor_rgb_recortado = tensor_recortado[0:profundidad_rgb, :,:,:]

    tensor_rgb_recortado = tensor_rgb_recortado.permute(3,0,1,2) # esto le cambie

    tensor_dosis_recortado = tensor_recortado[profundidad_rgb+1:profundidad_total, :,:,0]
    
    return tensor_rgb_recortado, tensor_dosis_recortado


def procesar_pacientes(root_folder, output_root_folder):
    subfolders = sorted(next(os.walk(root_folder))[1])
    for subfolder in subfolders:
        patient_folder = os.path.join(root_folder, subfolder)
        output_folder = output_root_folder
        
        # llamamos a la funcion que devuelve un diccionario con los rt parsers
        rt_parsers = get_rt_parser(patient_folder)
            
        # llamamos a la funcion que nos devuelve un diccionario de posiciones y el structure_parser, la misma tiene como argumento los rt_parsers
        dic_posiciones, structure_parser = get_dic_posiciones(rt_parsers)
           
        # llamamos a la funcion que nos devuelve el diccionario de coordenadas de cada estructura, tiene como argumento el diccionario de posiciones y structure_parser
        dic_coordenadas = get_coordenadas(dic_posiciones, structure_parser)
            
        # llamamos a la funcion que nos devuelve el tensor rgb (con la CT cargada en el canal B) y otras variables de utilidad, tiene como argumento los rt_parsers
        tensor_rgb, filas, columnas, coordenadas_z_ordenadas, datos_ct, informacion_cortes, x_ct, y_ct, pixel_spacing_ct = get_tensor_ct(rt_parsers)
            
        # llamamos a la funcion que nos devuelve el tensor rgb normalizado, con las estructuras y CT cargados en los canales correspondientes. 
        tensor_rgb_2, dic_coordenadas_sinz_estructuras, dic_mascaras = get_tensor_estructuras(dic_coordenadas, tensor_rgb, filas, columnas, coordenadas_z_ordenadas)
    
        # llamamos a la funcion que nos devuelve el tensor de dosis normalizado 
        tensor_dosis, dic_dosis_array, x_dosis, y_dosis, pixel_spacing_dosis = get_tensor_dosis(rt_parsers, coordenadas_z_ordenadas)
            
        # llamamos a la funcion que interpola el tensor de dosis
        tensor_interpolado = get_tensor_dosis_interpolado(tensor_dosis, pixel_spacing_dosis, pixel_spacing_ct)
            
        # llamamos a la funcion que nos devuelve el tensor recortado con dimensiones de [x,128,128,3]
        tensor_rgb_recortado, tensor_dosis_recortado = get_tensor_recortado(tensor_rgb, dic_coordenadas_sinz_estructuras, tensor_interpolado, x_dosis, y_dosis, x_ct, y_ct, pixel_spacing_ct)
        
        # Obtener el nombre del archivo sin la ruta y la extensión
        filename = os.path.splitext(os.path.basename(subfolder))[0]
            
        # Construir la ruta de salida para el tensor rgb
        output_file = os.path.join(output_root_folder, filename + ".pt")
        
        # Asignar el contenido de las variables a un diccionario
        variable = {"imagen": tensor_rgb_recortado, "dosis": tensor_dosis_recortado, "dic_mascaras": dic_mascaras, "czo": coordenadas_z_ordenadas}

        # Guardar el diccionario utilizando el nombre de la variable como nombre de archivo
        
        torch.save(variable, output_file)

def main():
    parser = argparse.ArgumentParser(description="Procesar pacientes y generar tensores.")
    parser.add_argument('--input', required=True, help='Ruta a la carpeta de entrada con los pacientes')
    parser.add_argument('--output', required=True, help='Ruta a la carpeta de salida para los tensores')
    args = parser.parse_args()
    procesar_pacientes(args.input, args.output)


if __name__ == "__main__":
    main()
