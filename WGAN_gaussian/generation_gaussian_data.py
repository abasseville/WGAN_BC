import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from fonctions_WGAN import *
from umap import UMAP



#-----------------------------------------------------------
# initial dataset creation (from gaussian distribution)
#-----------------------------------------------------------


## standard deviation of each variable (50 variables)
std = np.random.uniform(0.2,5,50)

## 500 individuals from gaussian distribution centered in 0
data = pd.DataFrame(np.random.normal(0,std,size=(500,50)), columns=np.arange(50))



#---------------------------------
# new individuals generation
#---------------------------------


it=30000
et=5
ech=20

G,C = WGAN(data, 300, 2, 6, nb_iterations=it, nb_step=et, batch_size=ech)

# scores and Wasserstein distance evolution 
graph(G,C)

# creation 
gen = create_indiv(200, G, data.columns)
over_data = pd.concat([data,gen])
cl = pd.Series(['Real']*500+['Fake']*200)



#-----------------------------------
# new dataset visualization 
#-----------------------------------


proj = UMAP(n_components=2, n_neighbors=15, min_dist=0.2, metric='manhattan').fit_transform(over_data)
df_proj= pd.DataFrame({'X1': proj[:,0], 'X2': proj[:,1], 'class' : cl})

plt.figure()
liste_col = ['crimson','cornflowerblue']
for color, clas in zip(liste_col, pd.unique(cl)):
	plt.scatter(df_proj['X1'][df_proj['class']==clas], df_proj['X2'][df_proj['class']==clas], color=color, label=clas, s=15)
plt.title("oversampled data UMAP")
plt.legend()
plt.show()
