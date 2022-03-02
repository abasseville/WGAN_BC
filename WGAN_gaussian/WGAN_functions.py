#!/usr/bin/python

from torch import nn, FloatTensor, mean
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import RMSprop
import pandas as pd
import matplotlib.pyplot as plt


pas=0.0001 # learning rate
c=0.01 # clipping parameter
size_z=2 # size of the random vector for generator input


class Generateur(nn.Module):

	def __init__(self,p,nb_neur,nb_cc_gen):

		super(Generateur,self).__init__()

		layers = [nn.Linear(size_z,nb_neur), nn.ReLU(inplace=True)]
		for i in range(nb_cc_gen-1):
			layers += [nn.Linear(nb_neur,nb_neur), nn.ReLU(inplace=True)]
		layers += [nn.Linear(nb_neur,p)]

		self.reseau = nn.Sequential(*layers)

		self.score_fake = []
		self.optimizer = RMSprop(self.parameters(), lr=pas)

	def forward(self, z):
		return self.reseau(z)


def groupsort(x):
	s=list(x.size())
	nc=s[0]
	s[0]=-1
	s.insert(1,nc//2)
	grouped_x = x.view(*s)
	sorted_grouped_x,_ = grouped_x.sort(dim=1,descending=True)
	sorted_x = sorted_grouped_x.view(*list(x.size()))
	return sorted_x

class GroupSort(nn.Module):

	def __init__(self):
		super(GroupSort, self).__init__()

	def forward(self, x):
		return groupsort(x)


class Critique(nn.Module):

	def __init__(self,p,nb_neur,nb_cc_cri):

		super(Critique,self).__init__()

		layers = [nn.Linear(p,nb_neur), GroupSort()]
		for i in range(nb_cc_cri-1):
			layers += [nn.Linear(nb_neur,nb_neur), GroupSort()]
		layers += [nn.Linear(nb_neur,1)]

		self.reseau = nn.Sequential(*layers)

		self.score_real = []
		self.W = []
		self.optimizer = RMSprop(self.parameters(), lr=pas)

	def forward(self, x):
		
		return self.reseau(x)



def WGAN(data, nb_neur, nb_cc_gen, nb_cc_cri, nb_iterations, nb_etapes, nb_echantillons):

	n,p=data.shape

	G = Generateur(p,nb_neur,nb_cc_gen)
	C = Critique(p,nb_neur,nb_cc_cri)

	i=0

	while (i < nb_iterations) and (C.W==[] or sum([abs(num) for num in C.W[-50:]])!=0):


		# monitoring the progress of the algorithm
		e=nb_iterations//20
		if i%e==0 and i!=0: 
			print(f'{int(i/nb_iterations*100)}%')
			print(f'Estimated Wasserstein distance : {C.W[-20:]}')
		elif i==nb_iterations-1: print('100%')


		for param in C.parameters():
			param.requires_grad=True

		for k in range(nb_etapes):

			C.optimizer.zero_grad()

			for param in C.parameters():
				param.data.clamp_(-c,c)

			sample_x = Variable(FloatTensor(data.sample(nb_echantillons).values))
			sample_z = Variable(Normal(0,1).sample((nb_echantillons,size_z)))

			score_x = mean(C(sample_x))
			score_z = mean(C(G(sample_z).detach()))
			loss_C = -score_x + score_z

			C.score_real.append(float(score_x))
			G.score_fake.append(float(score_z))
			C.W.append(-float(loss_C))

			loss_C.backward()
			C.optimizer.step()

		for param in C.parameters():
			param.data.clamp_(-c,c)
			param.requires_grad=False

		G.optimizer.zero_grad()

		sample_z = Variable(Normal(0,1).sample((nb_echantillons,size_z)))
		loss_G = -mean(C(G(sample_z)))

		loss_G.backward()
		G.optimizer.step()

		i += 1

	return G,C


def graph(G,C):
	plt.subplot(2,1,1)
	plt.plot(G.score_fake, linewidth=1, label='Generated data')
	plt.plot(C.score_real, linewidth=1, color='orange', label='Initial data')
	plt.xlabel('Step', fontsize=13)
	plt.ylabel('Average scores', fontsize=13)
	plt.legend(loc=1)
	plt.title("Average scores of fake and real data", fontsize=16)

	plt.subplot(2,1,2)
	plt.plot(C.W, linewidth=1)
	plt.xlabel('Step', fontsize=13)
	plt.ylabel('Average of C(x) - C(G(z))', fontsize=13)
	plt.title("Estimated Wasserstein distance", fontsize=16)
	plt.show()


def create_indiv(nb,gen,col):
	sample_z = Normal(0,1).sample((nb,size_z))
	new = gen.forward(sample_z)
	generated_indiv = pd.DataFrame(new.data.numpy(),columns=col)
	return generated_indiv

