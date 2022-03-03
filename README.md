# WGAN_BC

In this project, we would like to **predict whether a breast cancer patient will relapse or not after treatment**.  
The problem is that in medicine, data is cruelly lacking : there are many gene measurements but for few patients : this is called the **curse of dimensionality (n<<p)** !  
We would like to expand our patient cohort by generating it thanks to deep learning tools. We were inspired by **Wasserstein Generative Adversarial Networks** (WGANs) first used to create very realistic pictures of non-existent people. And we adapted it to suit our health data.  

### WGAN_functions.py :
We developped the WGAN algorithm with pytorch. It consists of a generator (with ReLU activation function) and a critic (with GroupSort activation function).  
- *WGAN* function allows optimizing neural networks.  
- *graph* function allows plot graph of true and fake data scores attributed by the critic as well as the graph of the Wasserstein distance approximation between true and fake data distribution (calculated from scores previously mentioned).  
- Finally, *create_indiv* function allows generating as data as wanted, thanks to WGAN generator, to be used after the optimizing step.  

### generation_gaussian_data.py :
1. creation of gaussian dataset
2. WGAN optimization from this dataset and new data generation
3. UMAP visualization of the new dataset, composed of initial gaussian data and WGAN generated data
