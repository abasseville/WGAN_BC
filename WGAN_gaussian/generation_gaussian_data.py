import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from fonctions_WGAN import *
from umap import UMAP



#-----------------------------------------------------------
# creation of initial dataset (from gaussian distribution)
#-----------------------------------------------------------


data = pd.DataFrame(columns=np.arange(50)) # 50 variables

## variance of each variable
std = np.random.uniform(0.2,5,50)

## 500 individuals from gaussian distribution centered in 0
for nb_gauss in range(500):
	new_gauss = pd.DataFrame([np.random.normal(0,std,50)], columns=np.arange(50))
	data = pd.concat([data,new_gauss])

data.reset_index(drop=True, inplace=True)



#---------------------------------
# generation of new individuals
#---------------------------------


it=30000
et=5
ech=20

G,C = WGAN(data, 300, 2, 6, nb_iterations=it, nb_etapes=et, nb_echantillons=ech)

# evolution of scores and Wassesrstein distance
graph(G,C)

# creation 
gen = create_indiv(200, G, data.columns)
over_data = pd.concat([data,gen])
cl = pd.Series(['Real']*500+['Fake']*200)



#-----------------------------------
# visualisation of the new dataset
#-----------------------------------


proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.2, metric='manhattan').fit_transform(over_data)
df_proj= pd.DataFrame({'X1': proj[:,0], 'X2': proj[:,1], 'class' : cl})

plt.figure()
liste_col = ['crimson','cornflowerblue']
for color, clas in zip(liste_col, pd.unique(cl)):
	plt.scatter(df_proj['X1'][df_proj['class']==clas], df_proj['X2'][df_proj['class']==clas], color=color, label=clas, s=15)
plt.title("UMAP of oversampled data")
plt.legend()
plt.show()
