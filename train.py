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
#import model

import matplotlib
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
import pandas as pd
import sys

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
    
    
    def forward(self, size, ref):       # query and reference
        self.batch_size = size
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


    def forward(self, x, h, c):       # query and reference
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
#        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
#        self.encoder = LSTM(n_hidden)
        
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
        #h, c = self.encoder(x, h, c)

        # query vector
        #q = x
        size = x.size(0)
        # pointer
        u, _ = self.pointer(size, context)

        latent_u = u.clone()
        
        u = 100 * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
        return F.softmax(u, dim=1), h, c, latent_u


size_n = 7
learn_rate = 1e-3    # learning rate
B = 150    # batch_size
B_val = 25     # validation size
size_val = 250
steps = 2500    # training steps
n_epoch = 10   # epochs
save_root_1 = './model/model1.pt'
save_root_2 = './model/model2.pt'

if(len(sys.argv) != 2):
    print('[Usage]')
    print('\tpython3 train.py <desntiy of the training data (matrix)>')
    sys.exit()

print('=========================')
print('prepare to train')
print('=========================')
print('Hyperparameters:')
print('size', size_n)
print('learning rate', learn_rate)
print('batch size', B)
print('validation size', B_val)
print('steps', steps)
print('epoch', n_epoch)
#print('save root:', save_root)
print('=========================')

#model = GPN(n_feature=2, n_hidden=128).cuda()
#size = 150
#size_feature = 150 
model_first = GPN(n_feature=1, n_hidden=128).cuda()
model_second = GPN(n_feature=1, n_hidden=128).cuda()

device = torch.device('cuda:1')
model_first.to(device)
model_second.to(device)

# load model
# model = torch.load(save_root).cuda()

optimizer_first = optim.Adam(model_first.parameters(), lr=learn_rate)
optimizer_second = optim.Adam(model_second.parameters(), lr=learn_rate)

lr_decay_step = 2500
lr_decay_rate = 0.96
opt_scheduler_first = lr_scheduler.MultiStepLR(optimizer_first, range(lr_decay_step, lr_decay_step*1000,
                                     lr_decay_step), gamma=lr_decay_rate)
opt_scheduler_second = lr_scheduler.MultiStepLR(optimizer_second, range(lr_decay_step, lr_decay_step*1000,
                                     lr_decay_step), gamma=lr_decay_rate)
C = 0     # baseline
R = 0     # reward

val_mean = []
val_std = []
train_reward = []
train_loss = []

density = float(sys.argv[1])
    
#torch.cuda.memory_summary(device=None, abbreviated=False)


for epoch in range(n_epoch):
    for i in range(steps):
        optimizer_first.zero_grad()
        optimizer_second.zero_grad()

     
        #X = np.random.rand(B, size, 2)
        size = size_n**2
        X = torch.zeros(B,size,size).to(device)
        X1 = np.random.rand(B,size_n,size_n)

        sdfs = []
        for j in range(B):
            sparse = np.random.binomial(n=1, p=density, size=size)
            sdf = pd.DataFrame(sparse.reshape(size_n,size_n))
            sdf  = sdf.to_numpy()
            sdfs.append(sdf) 

        X1 = torch.Tensor(X1).to(device)
        x1 = torch.zeros(B, size).to(device)
        for x in range(size_n):
            for y in range(size_n):
                x1[:,(x*size_n)+y] = X1[:,x,y]
        sdfs = torch.Tensor(sdfs).to(device)
        sdfss = torch.zeros(B,size).to(device)
        
        for x in range(size_n):
            for y in range(size_n):
                sdfss[:,(x*size_n)+y] = sdfs[:,x,y]
       
        for x in range(B):
            swaps = 0
            while(swaps<size):
                 if(x1[x,0] != 0):
                     break;
                 else:
                     if(x1[x,swaps]!=0):
                         x1_copy = x1.detach()
                         for y in range(size):
                             x1[x,0] = x1[x,swaps]
                             x1[x,swaps] = x1[x,0]
                             break;
                 swaps+=1
        for x in range(B):
            swaps = 0
            while(swaps<size):
                 if(sdfss[x,0] != 0):
                     break;
                 else:
                     if(sdfss[x,swaps]!=0):
                         sdfss_copy = sdfss.detach()
                         for y in range(size):
                             sdfss[x,0] = sdfss_copy[x,swaps]
                             sdfss[x,swaps] = sdfss_copy[x,0]
                             break;
                 swaps+=1
        for x in range(size):
            for y in range(size):
              X[:,x,y] = x1[:,x] * sdfss[:,y]
        #X = X * sdfs
        #X = torch.Tensor(X).to(device)
       
        #mask = torch.zeros(B,size).cuda()
        mask =  torch.zeros(B,size).to(device)
        mask_new =  torch.zeros(B,size_n).to(device)
        R = 0
        logprobs = 0
        reward = []
        reward_r = 0

        #Y = X.view(B,size,2)
        Y = X.view(B,size,size)
        #x = Y[:,0,:]
        x = Y[:,0,0]
        #Y1 = Y[:,0,:]
        h = None
        c = None

        Y1 = torch.zeros(B, size).to(device)
        for y_ini in range(size_n):
            for x_ini in range(size_n):
                for z in range(B):
                   Y1[z,(x_ini+size_n*y_ini)] = Y[z,y_ini*(size_n+1), x_ini*(size_n+1)].clone()


        route = []
        #critics = torch.zeros(1).cuda() 
        beta = 0.95
        idxs = []
        for k in range(size_n):
            '''
            output, h, c, _ = model(x=x, X_all=Y1, h=h, c=c, mask=mask)

            sampler = torch.distributions.Categorical(output)
            idx = sampler.sample()         # now the idx has B elements
            r1 = Y1[[i for i in range(B)], idx.data].clone()
            '''
            if k == 0:
                output, h, c, _ = model_first(x=x, X_all=Y1, h=h, c=c, mask=mask)
                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                #r1 = Y1[[i for i in range(B)], idx.data].clone()
                idx_num_x =[]
                idx_num_y =[]
                for j in range(B):
                    idx_num_x.append(int(idx[j]/size_n))
                    idx_num_y.append(idx[j]%size_n)
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
                        new_X_2 = torch.Tensor(numpy_X[j, number_x:number_x + size_n, number_y:number_y+ size_n]).to(device)
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
                x   = new_X[[i for i in range(B)], idx_num_y,idx_num_x].clone()
                route_x = torch.Tensor(idx_num_x).to(device)
                route_x = route_x.view(1,B)
                mask_new[[i for i in range(B)], idx_num_x] += -np.inf
                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx_num_x]+TINY)
            if k > 0:
                output, h, c, _ = model_second(x=x, X_all=Y1, h=h, c=c, mask=mask_new)

                sampler = torch.distributions.Categorical(output)
                idx = sampler.sample()         # now the idx has B elements
                Y0 = Y1.clone()
                Y1 = new_X[[i for i in range(B)], idx.data].clone()
                x   = Y1[[i for i in range(B)], idx.data].clone()
                now_route = idx.data.view(1,B)
                route_x = torch.cat((route_x,now_route),dim = 0)
                logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY)
                mask_new[[i for i in range(B)], idx.data] += -np.inf
        route_r = torch.transpose(route_x,0,1)
        for j in range(size_n):
            for l in range(size_n):
                #reward = X[[m for m in range(B)],route_r[m,j],route_r[m,l]]
                for m in range(B):
                    if m == 0:
                        reward = X[m ,size_n*route_r[m,j].long()+route_r[m,j].long(),size_n*route_r[m,l].long()+route_r[m,l].long()]
                        reward = reward.view(1)
                    if m > 0:
                        reward_r = X[m ,size_n*route_r[m,j].long()+route_r[m,j].long(),size_n*route_r[m,l].long()+route_r[m,l].long()]
                        reward_r = reward_r.view(1)
                        reward = torch.cat((reward,reward_r),dim=0)
                R += reward
        #R += torch.norm(Y1-Y_ini, dim=1u)
        #for z in range(B):
        #    R += Y[z,idxs[z] ,i_ini[z]]
        
        # self-critic base line
        mask = torch.zeros(B,size).to(device)
        mask_new =  torch.zeros(B,size_n).to(device)
        C = 0
        baseline = 0

        #Y = X.view(B,size,2)
        Y = X.view(B,size,size)
        x= Y[:,0,0]
        Y1 = Y[:,0,:]
        h = None
        c = None
        route_cc = []
        c_ini, idxss = [], []

        Y1 = torch.zeros(B, size).to(device)
        for y_ini in range(size_n):
            for x_ini in range(size_n):
                for z in range(B):
                   Y1[z,(x_ini+size_n*y_ini)] = Y[z,y_ini*(size_n+1), x_ini*(size_n+1)].clone()


        for k in range(size_n):
            if k == 0:
                output, h, c, _ = model_first(x=x, X_all=Y1, h=h, c=c, mask=mask)
                #sampler = torch.distributions.Categorical(output)
                #idx = sampler.sample()         # now the idx has B elements
                idx = torch.argmax(output, dim=1)    # greedy baseline
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
                        new_X = torch.Tensor(numpy_X[j, number_x:number_x + size_n,number_y:number_y+ size_n]).to(device)
                        new_X = new_X.view(1,size_n,size_n)
                    if j > 0:
                        number_x = idx_num_x[j]
                        number_y = idx_num_y[j]
                        new_X_2 = torch.Tensor(numpy_X[j, number_x:number_x + size_n,number_y:number_y+ size_n]).to(device)
                        new_X_2 = new_X_2.view(1,size_n,size_n)
                        new_X = torch.cat((new_X, new_X_2),0)                     
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
                x   = new_X[[i for i in range(B)], idx_num_y,idx_num_x].clone()
                route_cc = torch.Tensor(idx_num_x).to(device)
                route_cc = route_cc.view(1,B)
                mask_new[[i for i in range(B)], idx_num_x] += -np.inf
                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx_num_x]+TINY)
            if k>0:
                output, h, c, _ = model_second(x=x, X_all=Y1, h=h, c=c, mask=mask_new)
                #sampler = torch.distributions.Categorical(output)
                #idx = sampler.sample()         # now the idx has B elements
                idx = torch.argmax(output, dim=1)    # greedy baseline
                Y0 = Y1.clone()
                Y1 = new_X[[i for i in range(B)], idx.data].clone()
                x  = Y1[[i for i in range(B)], idx.data].clone()
                now_route = idx.data.view(1,B)
                route_cc = torch.cat((route_cc,now_route),dim = 0)
                mask_new[[i for i in range(B)], idx.data] += -np.inf
        route_c = torch.transpose(route_cc,0,1)
        for j in range(size_n):
            for l in range(size_n):
                #reward = X[[m for m in range(B)],route_r[m,j],route_r[m,l]]
                for m in range(B):
                   if m == 0:
                        baseline = X[m ,size_n*route_c[m,j].long()+route_c[m,j].long(),size_n*route_c[m,l].long()+route_c[m,l].long()]
                        baseline = baseline.view(1)

                   if m > 0:
                        baseline_c = X[m ,size_n*route_c[m,j].long()+route_c[m,j].long(),size_n*route_c[m,l].long()+route_c[m,l].long()]
                        baseline_c = baseline_c.view(1)
                        baseline = torch.cat((baseline,baseline_c),dim=0)
                C += baseline
        #C = baseline 
        #train_reward.append(R.item())

        gap = (R-C).mean()
        loss = ((R-C-gap)*logprobs*10).mean()
        #train_loss.append(loss.item())
        loss.backward()
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model_first.parameters(),
                                           max_grad_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model_second.parameters(),
                                           max_grad_norm, norm_type=2)
        optimizer_first.step()
        opt_scheduler_first.step()
        optimizer_second.step()
        opt_scheduler_second.step()
        if i % 50 == 0:
            print("epoch:{}, batch:{}/{}, reward:{}"
                .format(epoch, i, steps, R.mean().item()))
            '''
            # R_mean.append(R.mean().item())
            # R_std.append(R.std().item())

            # greedy validation
            
            tour_len = 0

            #X_val = np.random.rand(B_val, size_val, 2)
            X_val = np.random.rand(B, size_feature, size_feature)
            X = X_val
            #train_reward.append(tour_len)
            print('validation tour length:', tour_len)
            '''
    print('save model : ')
    torch.save(model_first, save_root_1)
    torch.save(model_second, save_root_2)
