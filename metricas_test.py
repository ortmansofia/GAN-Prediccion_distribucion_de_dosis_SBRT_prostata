import matplotlib.pyplot as plt
import torch 
import numpy as np
import pandas as pd
from torch import nn
import os
import json
import argparse

def graficar_DVH(dic):
    # Crear una nueva figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for nombre, dic_dosis in dic.items():
        dosis = list(dic_dosis.keys())
        volumen = list(dic_dosis.values())
        
        # Normalizar el volumen a porcentajes
        volumen_porcentajes = [v / max(volumen) * 100 for v in volumen]
        
        ax.plot(dosis, volumen_porcentajes, label=nombre)
    
    ax.set_xlabel('Dosis (Gy)')
    ax.set_ylabel('Volumen (%)')
    ax.set_title('Histograma de Dosis-Volumen')
    ax.legend()
    ax.grid(True)
    
    return fig

def get_dic_dosis_estructuras(ct, dose, vector_dosis):
    '''
    funcion calcula histograma d-v , le ingresa una ruta de un tensor que contieen la ct (rgb, con las estructuras) y la dosis
    
    '''
    dic_dosis_estructuras = dict()
    dic_dosis = dict()

    ptv_mask = torch.zeros_like(dose)
    recto_mask = torch.zeros_like(dose)
    vejiga_mask = torch.zeros_like(dose)

    ptv_mask = ct[1] == 1 
 
    recto_mask = ct[0] == 1

    vejiga_mask = ct[0] == 0.75


    #ahora obtenemos la dosis por cada estructura
    dose_ptv = dose[ptv_mask]
    dose_recto = dose[recto_mask]
    dose_vejiga = dose[vejiga_mask]

    # Diccionario de nombres y estructuras de dosis para iterar
    estructuras_dosis = {
        'dose_ptv': [dose_ptv,ptv_mask],
        'dose_recto': [dose_recto,recto_mask],
        'dose_vejiga': [dose_vejiga, vejiga_mask]
    }


    for nombre, (dose_estructura, mask_estructura) in estructuras_dosis.items():
        dic_dosis = {}
        for valor in vector_dosis:
            mask_bool = (dose_estructura >= valor)   # Combinación de máscaras
            cantidad_verdaderos = torch.sum(mask_bool).item()  
            
            dic_dosis[valor] = cantidad_verdaderos
        dic_dosis_estructuras[nombre] = dic_dosis 

    return dic_dosis_estructuras, dose_ptv, dose_recto, dose_vejiga

def mse_tensores(dose, pred):

    mse = torch.mean((dose - pred) ** 2)
    return mse

# Función para calcular el promedio de RMSE por clave
def promedio_rmse(lista_resultados_rmse):
    # Usamos defaultdict para almacenar listas de RMSEs por clave
    rmse_por_clave = defaultdict(list)

    # Iteramos sobre cada diccionario de resultados
    for diccionario in lista_resultados_rmse:
        for clave, rmse in diccionario.items():
            rmse_por_clave[clave].append(rmse)

    # Calculamos el promedio de RMSE para cada clave
    promedio_rmse_por_clave = {clave: sum(rmses) / len(rmses) for clave, rmses in rmse_por_clave.items()}
    
    # Convertimos el resultado en un DataFrame
    df_promedios_rmse = pd.DataFrame(list(promedio_rmse_por_clave.items()), columns=['Clave', 'Promedio_RMSE'])
    
    return df_promedios_rmse

def calcular_rmse_diccionarios(diccionario1, diccionario2):
    resultados_rmse = {}

    # Verificar que ambos diccionarios tienen las mismas claves
    claves_comunes = set(diccionario1.keys()) & set(diccionario2.keys())

    for clave in claves_comunes:
        lista1 = diccionario1[clave]
        lista2 = diccionario2[clave]
        
        # Calcular RMSE para la clave actual
        try:
            rmse = calcular_rmse(lista1, lista2)
            resultados_rmse[clave] = rmse
        except ValueError as e:
            print(f"Error para la clave '{clave}': {e}")
    
    return resultados_rmse
    
def graficar_DVH_juntos(puntos_dosis, puntos_pred):
    #creamos los diccionarios 
    dic_porcentaje_volumen_por_estructura_real = {}
    dic_porcentaje_volumen_por_estructura_pred = {}
    
    # Crear una nueva figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Definir colores específicos para las etiquetas
    colores = {
        'Real': ['blue', 'orange', 'green'],
        'Pred': ['blue', 'orange', 'green']
    }
    
    # Contador para los colores
    color_index = 0

    # Graficar dosis real
    for idx, (nombre, dic_dosis) in enumerate(puntos_dosis.items()):
        dosis_real = list(dic_dosis.keys())
        volumen_real = list(dic_dosis.values())
        
        # Normalizar el volumen a porcentajes
        volumen_porcentajes_dosis = [v / max(volumen_real) * 100 for v in volumen_real]
        dic_porcentaje_volumen_por_estructura_real[nombre] = volumen_porcentajes_dosis
        
        color = colores['Real'][idx % 3]  # Reutilizar colores si hay más de tres etiquetas
        ax.plot(dosis_real, volumen_porcentajes_dosis, label=nombre, color=color)

    # Graficar dosis predicha
    for idx, (nombre, dic_dosis_pred) in enumerate(puntos_pred.items()):
        dosis_pred = list(dic_dosis_pred.keys())
        volumen_pred = list(dic_dosis_pred.values())

        # Normalizar el volumen a porcentajes
        volumen_porcentajes_pred = [v / max(volumen_pred) * 100 for v in volumen_pred]
        dic_porcentaje_volumen_por_estructura_pred[nombre] = volumen_porcentajes_pred
        
        color = colores['Pred'][idx % 3]  # Reutilizar colores si hay más de tres etiquetas
        ax.plot(dosis_pred, volumen_porcentajes_pred, label=f'{nombre} (pred)', linestyle='--', color=color)

    ax.set_xlabel('Dosis (Gy)')
    ax.set_ylabel('Volumen (%)')
    ax.set_title('Histograma de Dosis-Volumen')
    ax.legend()
    ax.grid(True)

    return fig, dic_porcentaje_volumen_por_estructura_real, dic_porcentaje_volumen_por_estructura_pred
    
    
def get_DVH(path, ptv_lista_rmse, vejiga_lista_rmse, recto_lista_rmse, total_lista_rmse, ruta_paciente, lista_promedio_RMSE_DVH):

    resultados = dict()
    puntos_del_DVH = dict()
                
    dic_dosis_esc_real = dict()
    dic_dosis_esc_pred = dict()

    loaded = torch.load(path)
                    
    # Acceder a los tensores por separado
    
    dose_tensor = loaded["dose"]
    ct_tensor = loaded["ct"]
    pred_tensor = loaded["pred"]
    max_dosis_tensor = loaded["max_dosis"]

    print('dose', dose_tensor.shape)
    print('ct', ct_tensor.shape)
    print('pred', pred_tensor.shape)
    print('maxdosis', max_dosis_tensor.shape)
    

    batch_size = pred_tensor.size(0)
    print('batch_size', batch_size)
    
    for i in range(batch_size):
        
        dose = dose_tensor[i]
        ct = ct_tensor[i]
        pred = pred_tensor[i]
        max_dosis = max_dosis_tensor[i]
    
        pred = pred.squeeze(0)
        dose = dose.squeeze(0)
    
        dose = (dose* 0.5 + 0.5).cpu()
        ct = (ct* 0.5 + 0.5).cpu()
        pred = (pred* 0.5 + 0.5).cpu()
    
        dose = dose*max_dosis
        pred = pred*max_dosis
        
        # definimos el limite para el percentil 10
        threshold = 0.1 * 36.25
        
        dose_filtrado = dose[dose >= threshold]

        #Filtramos pred
        pred_filtrado = pred[dose>= threshold]
    
        # Calcular los valores mínimos y máximos para ambas imágenes
        min_value = min(dose.min(), pred.min())
        max_value = max(dose.max(), pred.max())
    
        #vector de dosis 
        vector_dosis = np.linspace(min_value, max_value, 200)
                    
        #obtenemos los dic que contienen valores de dosis por cada estrutura
        dic_dosis_esc_real, dose_ptv_real, dose_recto_real, dose_vejiga_real = get_dic_dosis_estructuras(ct, dose, vector_dosis)
        dic_dosis_esc_pred, dose_ptv_pred, dose_recto_pred, dose_vejiga_pred = get_dic_dosis_estructuras(ct, pred, vector_dosis)

        
    
        puntos_del_DVH = {'dic_dosis_esc_real': dic_dosis_esc_real,
                          'dic_dosis_esc_pred': dic_dosis_esc_pred}
                       
        #Graficamos y guardamos los DVH
        DVH_real = graficar_DVH(dic_dosis_esc_real)
        DVH_pred = graficar_DVH(dic_dosis_esc_pred)

        #graficamos los DVH superpuestos 
        DVH_juntos, dosis_porcentual_real, dosis_porcentual_pred = graficar_DVH_juntos(dic_dosis_esc_real, dic_dosis_esc_pred)

        # Calcular RMSE por estructura
        rmse_por_estructura_porcentual = calcular_rmse_diccionarios(dosis_porcentual_real, dosis_porcentual_pred)

        # lo agregamos a la lista
        lista_promedio_RMSE_DVH.append(rmse_por_estructura_porcentual)
        
        # Para mostrar la imagen en Jupyter Notebook (por ejemplo)
        # DVH_real.show()
        # DVH_pred.show()
     
        #Calculamos RMSE total
        rmse_total = np.sqrt(mse_tensores(dose_filtrado, pred_filtrado))
                        
        #calcular mse por estructura
    
        rmse_ptv = np.sqrt(mse_tensores(dose_ptv_real, dose_ptv_pred))
        rmse_recto = np.sqrt(mse_tensores(dose_recto_real, dose_recto_pred))
        rmse_vejiga = np.sqrt(mse_tensores(dose_vejiga_real, dose_vejiga_pred))
    
        # Añadir resultados a la lista
        resultados = {
            'rmse_ptv': [rmse_ptv.item()],
            'rmse_recto': [rmse_recto.item()], 
            'rmse_vejiga': [rmse_vejiga.item()], 
            'rmse_total': [rmse_total.item()]
        }
        
        ptv_lista_rmse.append(rmse_ptv.item())
        vejiga_lista_rmse.append(rmse_vejiga.item())
        recto_lista_rmse.append(rmse_recto.item())
        total_lista_rmse.append(rmse_total.item())
        
        # Guardamos los resultados
        subfolder_path = os.path.dirname(path)
        
        # Crear el nombre de la nueva carpeta
        nueva_carpeta = os.path.join(ruta_paciente, 'Metricas')
        
        # Crear la nueva carpeta si no existe
        os.makedirs(nueva_carpeta, exist_ok=True)
        
        dvh_real_file_path = os.path.join(nueva_carpeta, f'DVH_real{i}.jpg')
        dvh_pred_file_path = os.path.join(nueva_carpeta, f'DVH_pred{i}.jpg')
        dvh_junto_file_path = os.path.join(nueva_carpeta, f'DVH_junto{i}.jpg')
        
        puntos_file_path = os.path.join(nueva_carpeta, f'puntos{i}.pt')
        
        # Guardar los resultados en un archivo Excel
        resultados_df = pd.DataFrame(resultados)
        resultados_excel_path = os.path.join(nueva_carpeta, f'rmse_paciente_{i}.xlsx')
        resultados_df.to_excel(resultados_excel_path, index=False)
    
        torch.save(puntos_del_DVH, puntos_file_path)
    
        DVH_real.savefig(dvh_real_file_path, format='jpg')
        DVH_pred.savefig(dvh_pred_file_path, format='jpg')
        DVH_junto.savefig(dvh_junto_file_path, format='jpg')

        # Guardar RMSE en un archivo Excel
        df_rmse = pd.DataFrame(list(rmse_por_estructura_porcentual.items()), columns=['Estructura', 'RMSE'])
        ruta_excel_rmse = os.path.join(nueva_carpeta, f'rmse_por_estructura{i}.xlsx')
        df_rmse.to_excel(ruta_excel_rmse, index=False)
        
    return ptv_lista_rmse, vejiga_lista_rmse, recto_lista_rmse, total_lista_rmse, dose_ptv_real, dose_recto_real, dose_vejiga_real,dose_ptv_pred, dose_recto_pred, dose_vejiga_pred, pred_filtrado, dose_filtrado, lista_promedio_RMSE_DVH

def main():
    
    parser = argparse.ArgumentParser(description='Ruta para obtener metricas')
    parser.add_argument('ruta', metavar='ruta', type=str, help='Ingrese ruta')
    
    args = parser.parse_args()
    ruta = args.ruta

    root= r"/users/sortman/PI-GAN-2/Nuestro_Pix2Pix/Resultados" 
    
    directorio_principal = os.path.join(root, ruta)

    print("path",directorio_principal)

    ptv_lista_rmse = list()
    vejiga_lista_rmse = list()
    recto_lista_rmse = list()  
    total_lista_rmse = list()

    #listas para lamecnear las metricas estadisticas 
    list_mean_ptv_dose = list()
    list_mean_recto_dose = list()
    list_mean_vejiga_dose = list()
    list_mean_total_dose = list()
    list_mean_ptv_pred = list()
    list_mean_recto_pred = list()
    list_mean_vejiga_pred = list()
    list_mean_total_pred = list()
    list_std_ptv_dose = list()
    list_std_recto_dose = list()
    list_std_vejiga_dose = list()
    list_std_total_dose = list()
    list_std_ptv_pred = list()
    list_std_recto_pred = list()
    list_std_vejiga_pred = list()
    list_std_total_pred = list()

    #lista para los promedios de los RMSE de los DVH
    lista_promedio_RMSE_DVH = list ()
    
    if os.path.isdir(directorio_principal):
        for paciente in os.listdir(directorio_principal):
            ruta_paciente = os.path.join(directorio_principal, paciente)
            
            if os.path.isdir(ruta_paciente):
    
                for file in os.listdir(ruta_paciente):
                    file_path = os.path.join(ruta_paciente, file)
                    if not os.path.isdir(file_path):
                        if file.endswith('.pt'):
                            print(file_path)
                            
                            ptv_lista_rmse, vejiga_lista_rmse, recto_lista_rmse, total_lista_rmse, dose_ptv, dose_recto, dose_vejiga,pred_ptv, pred_recto, pred_vejiga, dose, pred, lista_promedio_RMSE_DVH  = get_DVH(file_path, ptv_lista_rmse, vejiga_lista_rmse, recto_lista_rmse, total_lista_rmse, ruta_paciente, lista_promedio_RMSE_DVH)
                            
                            # Obtener los valores medios
                            mean_dose_ptv = torch.mean(dose_ptv).item()
                            mean_dose_recto = torch.mean(dose_recto).item()
                            mean_dose_vejiga = torch.mean(dose_vejiga).item()
                            mean_dose_total = torch.mean(dose).item()
                        
                            mean_pred_ptv = torch.mean(pred_ptv).item()
                            mean_pred_recto = torch.mean(pred_recto).item()
                            mean_pred_vejiga = torch.mean(pred_vejiga).item()
                            mean_pred_total = torch.mean(pred).item()
                        
                            std_dose_ptv = torch.std(dose_ptv).item()
                            std_dose_recto = torch.std(dose_recto).item()
                            std_dose_vejiga = torch.std(dose_vejiga).item()
                            std_dose_total = torch.std(dose).item()
                        
                            std_pred_ptv = torch.std(pred_ptv).item()
                            std_pred_recto = torch.std(pred_recto).item()
                            std_pred_vejiga = torch.std(pred_vejiga).item()
                            std_pred_total = torch.std(pred).item()
                    
                            list_mean_ptv_dose.append(mean_dose_ptv)
                            list_mean_recto_dose.append(mean_dose_recto)
                            list_mean_vejiga_dose.append(mean_dose_vejiga)
                            list_mean_total_dose.append(mean_dose_total)
                            list_mean_ptv_pred.append(mean_pred_ptv)
                            list_mean_recto_pred.append(mean_pred_recto)
                            list_mean_vejiga_pred.append(mean_pred_vejiga)
                            list_mean_total_pred.append(mean_pred_total)
                    
                            list_std_ptv_dose.append(std_dose_ptv)
                            list_std_recto_dose.append(std_dose_recto)
                            list_std_vejiga_dose.append(std_dose_vejiga)
                            list_std_total_dose.append(std_dose_total)
                            list_std_ptv_pred.append(std_pred_ptv)
                            list_std_recto_pred.append(std_pred_recto)
                            list_std_vejiga_pred.append(std_pred_vejiga)
                            list_std_total_pred.append(std_pred_total)

    if len(ptv_lista_rmse) > 0:
                        
        ptv_mean_rmse = np.mean(ptv_lista_rmse)
        ptv_max_rmse = np.max(ptv_lista_rmse)
        ptv_min_rmse = np.min(ptv_lista_rmse)
    
        vejiga_mean_rmse = np.mean(vejiga_lista_rmse)
        vejiga_max_rmse = np.max(vejiga_lista_rmse)
        vejiga_min_rmse = np.min(vejiga_lista_rmse)
    
        recto_mean_rmse = np.mean(recto_lista_rmse)
        recto_max_rmse = np.max(recto_lista_rmse)
        recto_min_rmse = np.min(recto_lista_rmse)
    
        total_mean_rmse = np.mean(total_lista_rmse)
        total_max_rmse = np.max(total_lista_rmse)
        total_min_rmse = np.min(total_lista_rmse)
    
        # Crear un DataFrame con los resultados totales
        resultados_totales = {
            'ptv_mean_rmse': [ptv_mean_rmse],
            'ptv_max_rmse': [ptv_max_rmse],
            'ptv_min_rmse': [ptv_min_rmse],
            'vejiga_mean_rmse': [vejiga_mean_rmse],
            'vejiga_max_rmse': [vejiga_max_rmse],
            'vejiga_min_rmse': [vejiga_min_rmse],
            'recto_mean_rmse': [recto_mean_rmse],
            'recto_max_rmse': [recto_max_rmse],
            'recto_min_rmse': [recto_min_rmse],
            'total_mean_rmse': [total_mean_rmse],
            'total_max_rmse': [total_max_rmse],
            'total_min_rmse': [total_min_rmse]
        }
        
        df_resultados = pd.DataFrame(resultados_totales)
        
    # Guardar el DataFrame en un archivo Excel
    resultados_totales_path = os.path.join(directorio_principal, 'resultados_totales.xlsx')
    df_resultados.to_excel(resultados_totales_path, index=False)
    
    # Calcular los promedios de cada lista
    mean_ptv_dose_avg = sum(list_mean_ptv_dose) / len(list_mean_ptv_dose)
    mean_recto_dose_avg = sum(list_mean_recto_dose) / len(list_mean_recto_dose)
    mean_vejiga_dose_avg = sum(list_mean_vejiga_dose) / len(list_mean_vejiga_dose)
    mean_total_dose_avg = sum(list_mean_total_dose) / len(list_mean_total_dose)
    mean_ptv_pred_avg = sum(list_mean_ptv_pred) / len(list_mean_ptv_pred)
    mean_recto_pred_avg = sum(list_mean_recto_pred) / len(list_mean_recto_pred)
    mean_vejiga_pred_avg = sum(list_mean_vejiga_pred) / len(list_mean_vejiga_pred)
    mean_total_pred_avg = sum(list_mean_total_pred) / len(list_mean_total_pred)
    
    # Calcular los promedios de las desviaciones estándar
    std_ptv_dose_avg = sum(list_std_ptv_dose) / len(list_std_ptv_dose)
    std_recto_dose_avg = sum(list_std_recto_dose) / len(list_std_recto_dose)
    std_vejiga_dose_avg = sum(list_std_vejiga_dose) / len(list_std_vejiga_dose)
    std_total_dose_avg = sum(list_std_total_dose) / len(list_std_total_dose)
    std_ptv_pred_avg = sum(list_std_ptv_pred) / len(list_std_ptv_pred)
    std_recto_pred_avg = sum(list_std_recto_pred) / len(list_std_recto_pred)
    std_vejiga_pred_avg = sum(list_std_vejiga_pred) / len(list_std_vejiga_pred)
    std_total_pred_avg = sum(list_std_total_pred) / len(list_std_total_pred)
    
    # Crear un diccionario para organizar los datos
    data = {
        'Mean_PTV_Dose': [mean_ptv_dose_avg],
        'Mean_Recto_Dose': [mean_recto_dose_avg],
        'Mean_Vejiga_Dose': [mean_vejiga_dose_avg],
        'Mean_Total_Dose': [mean_total_dose_avg],
        'Mean_PTV_Pred': [mean_ptv_pred_avg],
        'Mean_Recto_Pred': [mean_recto_pred_avg],
        'Mean_Vejiga_Pred': [mean_vejiga_pred_avg],
        'Mean_Total_Pred': [mean_total_pred_avg],
        'Std_PTV_Dose': [std_ptv_dose_avg],
        'Std_Recto_Dose': [std_recto_dose_avg],
        'Std_Vejiga_Dose': [std_vejiga_dose_avg],
        'Std_Total_Dose': [std_total_dose_avg],
        'Std_PTV_Pred': [std_ptv_pred_avg],
        'Std_Recto_Pred': [std_recto_pred_avg],
        'Std_Vejiga_Pred': [std_vejiga_pred_avg],
        'Std_Total_Pred': [std_total_pred_avg]
    }
    
    # Crear el DataFrame
    df = pd.DataFrame(data)
    ruta_guardar = os.path.join(directorio_principal, 'promedios_metricas_dosis_predicciones.xlsx')
    
    # Guardar el DataFrame en un archivo Excel
    df.to_excel(ruta_guardar, index=False)

    #promedio del RMSE de los DVH 
    df_promedios_DVH = promedio_rmse(lista_promedio_RMSE_DVH)

    # Guardar el DataFrame en un archivo Excel con los promedios de DVH
    ruta_guardado = os.path.join(directorio_principal, 'promedios_rmse_dvh.xlsx')
    df_promedios_DVH.to_excel(ruta_guardado, index=False)

    return None
if __name__ == '__main__':
    main()