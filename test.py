import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from torch.utils.data import Dataset
from tqdm import tqdm_notebook
from scipy.spatial import distance

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden

        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))

        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)


    def forward(self, q, ref):       # query and reference
        #self.batch_size = q.size(0)
        self.batch_size = q
        self.size = int(ref.size(0) / self.batch_size)
        #q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        #q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)

        # (B, size, dim) * (B, dim, 1)
        #u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        u = torch.bmm(torch.tanh(ref), v_view).squeeze(2)

        return u, ref

class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
    
    def forward(self, x, h, c,dim):       # query and reference
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c

class GPN(nn.Module):
    
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        #self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        #self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        
        #self.embedding_x = nn.Linear(n_feature, n_hidden)
        #self.embedding_all = nn.Linear(n_feature, n_hidden)
        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
        
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        
        
        # =============================
        # vector context
        # =============================
        
        #x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (B, size)
        #X_all = X_all - x_expand
        
        # the weights share across all the cities
        x = x.unsqueeze(1)
        x = self.embedding_x(x)
        X_all_expand = X_all.unsqueeze(2)
        context = self.embedding_all(X_all_expand)
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
            '''
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
            '''
            
        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.view(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context/(self.city_size-1)))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context/(self.city_size-1)))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context/(self.city_size-1)))
        
        # LSTM encoder
        #h, c = self.encoder(x, h, c, self.city_size)
        # query vector
        #q = h
        size = x.size(0)
        # pointer
        u, _ = self.pointer(size, context)
        
        latent_u = u.clone()
        
        u = 100 * torch.tanh(u) + mask
        if latent is not None:
            u += self.alpha * latent
            
        return F.softmax(u, dim=1), h, c, latent_u
    # read tsplip
import pandas as pd
import numpy as np
import time
import sys
#import tsplib95
import math
from scipy.spatial import distance


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


#
# Reads instance data
#
def min_max(x,axis = None):
    min = x.min(axis = axis, keepdims=True)
    max = x.max(axis = axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#file_it = iter(read_integers(sys.argv[1]))

#number of input size
#n = next(file_it)
#A = [[next(file_it) for j in range(n)] for i in range(n)]


if(len(sys.argv) != 4):
    print('[Usage]')
    print('\tpython3 test.py <model1> <model2> <problem>')
    sys.exit()
#
# Reads instance data
#

load_root  = (sys.argv[1])
load_root2  = (sys.argv[2])
file_it = iter(read_integers(sys.argv[3]))

    # Number of points
n = next(file_it)

    # Distance between locations
input_a = np.array([[next(file_it) for j in range(n)] for i in range(n)], dtype='float32')
# Flow between factories
input_b = np.array([[next(file_it) for j in range(n)] for i in range(n)], dtype='float32')

#print(input_a, input_b)
#df = df1.T
#df.columns =['x','y']
#n_customer = problem.dimension
#n_point = n_customer + 1


device = torch.device('cuda:0')
model_first = torch.load(load_root).to(device)
model_second = torch.load(load_root2).to(device)

size_n =n
size = n**2
print("size:", size)
in_a, in_b = [], []

swaps = 0
while(swaps<n):
    if(input_a[0,0] != 0):
        input_a_swap = input_a.copy()   
        break;
    else:
        if(input_a[0,swaps] != 0):
            input_a_swap = input_a.copy()
            for j in range(n):
                input_a_swap[j,0] = input_a[j,swaps]
                input_a_swap[j,swaps] = input_a[j,0]
            break;
        swaps+= 1

#print(input_a)
#print("swap")
#print(input_a_swap)
for x in range(n):
    for y in range(n):
        in_a.append(input_a_swap[x,y])

for x in range(n):
    for y in range(n):
        in_b.append(input_b[x,y])

input = np.zeros((n**2,n**2))
for x in range(size):
    for y in range(size):
        input[x][y] = in_a[x]* in_b[y]
        input_new = np.array(input,dtype='float32')
        #print(input_new)
input_new = min_max(input_new)
#print(input_new)
'''
n_model = model.embedding_x.in_features
input = np.zeros((n_customer,n_customer))

#input_a = min_max(input_a)
max_a = np.amax(input_a)
for i in range(n_customer):
    for j in range(n_customer):
        if i == j:
           input_a[i][j] = max_a

print(input_a)


# Distance between locations
input_a = [[next(file_it) for j in range(n)] for i in range(n)]
    # Flow between factories
input_b = np.array([[next(file_it) for j in range(n)] for i in range(n)], dtype='float32')


load_root = sys.argv[2]
model = torch.load(load_root).cuda()

print(model.embedding_x.in_features)
n_model = model.embedding_x.in_features 
'''
B = 1     # process 

total_tour_len = 0
n_test = 1

# there are 1000 test data
#size =n_model

Z = torch.rand(B, size, size).clone()

#input_a = input_a.reshape(B, n_customer, n_customer)
#label2 = input_b.reshape(n_test, B, size, size)

for m in range(1):
    
    input_aa = min_max(input_new)
    b = torch.tensor(input_aa)
    X = b.to(device)

    X = X.repeat(B, 1, 1)
    mask = torch.zeros(B,size).to(device)
    mask_new = torch.zeros(B,n).to(device)


    R = 0
    solution = []
    route_1 = []
    reward = 0

    Y = X.view(B, size, size)           # to the same batch size
    x = Y[:,0,0]

    Y1 = torch.zeros(B, size).to(device)
    for y_ini in range(size_n):
        for x_ini in range(size_n):
            for z in range(B):
                Y1[z,(x_ini+size_n*y_ini)] = Y[z,y_ini*(size_n+1), x_ini*(size_n+1)].clone()
                #Y1 = Y[:,0,:]
    
    route_x, c_ini,idxss = [],[],[]
    h = None
    c = None
    '''    
    for k in range(size_n):
        for j in range(size_n):
            if((k*size_n + j)!=(j+ size_n*j)):
                 mask[0,k*size_n + j] += -np.inf
                 '''
    for k in range(size_n):
        if k == 0:
            output, h, c, _ = model_first(x=x, X_all=Y1, h=h, c=c, mask=mask)
            #sampler = torch.distributions.Categorical(output)
            idx = torch.argmax(output, dim=1)         # now the idx has B elements
            #r1 = Y1[[i for i in range(B)], idx.data].clone()
            idx_num_x =[]
            idx_num_y =[]
            for j in range(B):
                idx_num_x.append(int(idx[j]/size_n))
                idx_num_y.append((idx[j]%size_n))
                numpy_X = X.to('cpu').detach().numpy().copy() 
            for j in range(B):
                if j == 0:
                    number_x = idx_num_x[j]
                    number_y = idx_num_y[j]
                    new_X = torch.Tensor(numpy_X[j, number_x:number_x + size_n,number_y: number_y+size_n]).to(device)
                    new_X = new_X.view(1,size_n,size_n)
                if j > 0:
                    number_x = idx_num_x[j]
                    number_y = idx_num_y[j]
                    new_X_2 = torch.Tensor(numpy_X[j, number_x:number_x + size_n,number_y: number_y+size_n]).to(device)
                    new_X_2 = new_X_2.view(1,size_n,size_n)
                    new_X = torch.cat((new_X, new_X_2),0)                     
                    #print(idx.data, idx_num)
                new_idx = []
                '''
                for j in range(B):
                    if j == 0 :
                        new_idx = idx.data[j] -( idx_num[j] * size_n)
                        new_idx = new_idx.view(1)
                    if j > 0 :
                        new_idx_2 = (idx.data[j] - (idx_num[j]) * size_n)
                        new_idx_2 = new_idx_2.view(1)
                        new_idx = torch.cat((new_idx, new_idx_2), dim = 0)
                        '''
                Y1  = new_X[[i for i in range(B)],idx_num_x].clone()
                x  = new_X[[i for i in range(B)], idx_num_y,idx_num_x].clone()
                route_x = torch.Tensor(idx_num_x).to(device)
                route_x = route_x.view(1,B)
                TINY = 1e-15
                mask_new[[i for i in range(B)], idx_num_x] += -np.inf

        if k > 0:
            output, h, c, _ = model_second(x=x, X_all=Y1, h=h, c=c, mask=mask_new)
            #sampler = torch.distributions.Categorical(output)
            idx = torch.argmax(output, dim=1)         # now the idx has B elements
            Y0 = Y1.clone()
            Y1 = new_X[[i for i in range(B)], idx.data].clone()
            x  = Y1[[i for i in range(B)], idx.data].clone()
            now_route = idx.data.view(1,B)
            route_x = torch.cat((route_x,now_route),dim = 0)
            TINY = 1e-15
            mask_new[[i for i in range(B)], idx.data] += -np.inf
            route_r = torch.transpose(route_x,0,1)
            #print("print_r")
            #print(route_r)
    '''
        #R += torch.norm(Y1-Y_ini, dim=1u)
        #for z in range(B):
        #    R += Y[z,idxs[z] ,i_ini[z]]
        
        # self-critic base line
        mask = torch.zeros(B,size).to(device)
        C = 0
        baseline = 0

        output, h, c, _ = model(x=x, X_all=Y1, h=h, c=c, mask=mask)

      
        idx = torch.argmax(output, dim=1)
        
        x  = Y[[i for i in range(B)], idx.data]
        Y1 = Y[[i for i in range(B)],idx.data]
        #solution.append(x.numpy())
        route_1.append(idx.data[0].item())
        mask[[i for i in range(B)], idx.data] += -np.inf
        '''
    #solution.append(solution[0])
    #graph = np.array(solution)

sums = 0
route_ture_r = route_r.clone()
route_ture_r[0,0] = route_r[0,swaps]
route_ture_r[0,swaps] = route_r[0,0]

print(route_ture_r)
for i in range(n):
    for j in range(n):
        sum2 = input_a[i][j] * input_b[int(route_ture_r[0,i].item())][int(route_ture_r[0,j].item())]
        sums += sum2
print(sums)
