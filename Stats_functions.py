# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:44:09 2024

@author: raulb
"""

"Scripts de analisis de datos"

#Bibliotecas necesarias -------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
from scipy import stats

#Definimos una funcion que nos dara todas las estadisticas descriptivas -------
def statcats(x):
    #Definimos las estadisticas
    minimo = np.round(np.nanmin(x),4)
    maximo = np.round(np.nanmax(x),4)
    media = np.round(np.nanmean(x),4)
    mediana = np.round(np.median(x),4)
    varianza = np.round(np.nanvar(x, ddof = 1),4)
    ds = np.round(np.nanstd(x, ddof = 1), 4)
    sesgo = np.round(stats.skew(x, nan_policy= "omit"),4)
    kurtosis = np.round(stats.kurtosis(x, nan_policy = "omit"),4)
    
    #Guardamos en un df
    data = {
    'Estadisticas': ["minimo", "maximo", "media", "mediana", "varianza", "desv_std", 
               "sesgo", "curtosis"],
    'Valores': [minimo, maximo, media, mediana, varianza, ds, sesgo, kurtosis]
    }
    
    stats_df = pd.DataFrame(data)
    
    return stats_df

#Definimos un objeto que nos de los graficos necesarios ---------------------
class analisis:
    def __init__(self, x, var_type):
        self.x = x
        self.var_type = var_type
    
    def plot(self):
        
        #Caso 1. Variable continua
        if self.var_type == "continuo":
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            
            sns.boxplot(ax = axes[0], y = self.x, color = "#bb0000")
            
            sns.kdeplot(ax = axes[1], x = self.x, fill = True, color = "#800020")
            plt.ylabel(None)

            plt.tight_layout()
            plt.show()
        
        #Caso 2. Variable discreta
        elif self.var_type == "discreto":
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            
            sns.boxplot(ax = axes[0], y = self.x, color = "#008080")
            
            sns.histplot(ax = axes[1], x = self.x, fill = True, color = "#00a77d",
                         discrete = True)
            plt.ylabel(None)
            
            plt.tight_layout()
            plt.show()
        
        #Caso 3. Variable categorica
        elif self.var_type == "categorico":
            tabla = pd.value_counts(self.x)
            pal = sns.color_palette("Set2")
            plt.pie(tabla.values.tolist(), labels = tabla.index.tolist(), 
                    autopct='%.0f%%', colors = pal)
            plt.show()
        
        else: 
            print("var_type invalido")
    
    #Metodo que devuelve las estadisticas
    def sts(self):
        #Caso 1. Variables numericas
        if self.var_type == "continuo" or self.var_type == "discreto":
            t = statcats(self.x)
            return t 
        
        #Caso 2. Variable categorica
        elif self.var_type == "categorico":
            tabla = pd.value_counts(self.x)
            return tabla
        
        else:
            return None
            
            
#Definimos un objeto de correlacion entre variables ---------------------------
class correlacion:
    def __init__(self, x, y, x_type, y_type):
        self.x = x
        self.y = y
        self.x_type = x_type
        self.y_type = y_type
    
    #Funcion para mostar los graficos
    def plot(self, method = "pearson"):
        
        #Caso 1. Ambas variables numericas
        if self.x_type == "numerico" and self.y_type == "numerico":
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            
            #Scatterplot 
            sns.scatterplot(ax = axes[0], x = self.x, y = self.y, color = "maroon")
            
            #Correlacion
            ##Guardamos las variables en un df
            data = {"x": self.x, "y": self.y}
            df = pd.DataFrame(data)
            
            c = df.corr(method = method)
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(c, vmax = 1, center = 0, square = True, ax = axes[1],
                        linewidths=.5, cmap= cmap, cbar_kws={"shrink": 0.5})

            plt.tight_layout()
            plt.show()
        
        #Caso 2. Ambas variables categoricas
        if self.x_type == "categorico" and self.y_type == "categorico":
            sns.displot(x = self.x, discrete = True, hue = self.y, 
                        multiple = "stack")
            plt.ylabel(None)
            plt.show()
            
    #Metodo para la correlacion entre variables
    def corr(self, test = "chi-square"):
        if self.x_type == "numerico" and self.y_type == "numerico":
            
            #Guardamos variables en un df
            data = {"x": self.x, "y": self.y}
            df = pd.DataFrame(data)
            
            #Metodos de la correlacion
            metodo = ["pearson", "kendall", "spearman"]
            corr = []
            
            for m in metodo:
                c = np.round(df["x"].corr(df["y"], method = m),4)
                corr.append(c)
            
            dt = {"Metodo": metodo, "Valores": corr}
            df_cor = pd.DataFrame(dt)
            
            return df_cor
        
        #Ambas variables categoricas
        elif self.x_type == "categorico" and self.y_type == "categorico":
            
            #Guardamos variables en un df
            data = {"x": self.x, "y": self.y}
            df = pd.DataFrame(data)
            
            #Realizamos la tabla y la prueba
            tabla, valores = rp.crosstab(data["x"], data["y"], test = test)
            
            print(tabla)
            return valores
        
        else:
            return None