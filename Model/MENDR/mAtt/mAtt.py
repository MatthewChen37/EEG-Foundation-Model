import torch
import torch.nn as nn
from .spd import SPDTransform, SPDTangentSpace, SPDRectified
from ..safeSVD import SVD

'''
Modified from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')
    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        # To avoid division by 0 error
        tra = tra + (1E-5)*torch.ones(tra.shape).to(tra.device)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        # Notice how they also added 1e-5 originally
        cov = cov+(1e-5*identity)
        return cov 

class E2R(nn.Module):
    def __init__(self, epochs, device):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()
        self.device = device
    def patch_len(self, n, epochs):
        list_len=[]
        base = n//epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base*epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')
    
    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        x = x.to(self.device)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, device):
        super(AttentionManifold, self).__init__()
        
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.device = device
        self.q_trans = SPDTransform(self.d_in, self.d_out, self.device)
        self.k_trans = SPDTransform(self.d_in, self.d_out, self.device)
        self.v_trans = SPDTransform(self.d_in, self.d_out, self.device)

        self.svd = SVD.apply

    def tensor_log(self, t):#4dim
        '''
        output = torch.zeros(t.shape).to(self.device)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                u, s, v = self.svd(t[i, j, :, :])
                output[i, j] = u @ torch.diag_embed(torch.log(s)) @ v.permute(1, 0)
        return output
        '''
        batch = t.shape[0]
        epochs = t.shape[1]
        u, s, v = self.svd(t.view(batch * epochs, t.shape[2], t.shape[3]))
        u = u.view(batch, epochs, u.shape[1], u.shape[2])
        s = s.view(batch, epochs, s.shape[1])
        v = v.view(batch, epochs, v.shape[1], v.shape[2])
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)
        # condition: t is symmetric!
        #s, u = torch.linalg.eigh(t)
        #print(s, u)
        #print(torch.linalg.norm(u[0, 0, :, :,], dim=0))
        #print(u.shape)
        #return u @ torch.diag_embed(torch.log(s)) @ u.permute(0, 1, 3, 2)
        #u, s, v = torch.svd(t)
        #return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)
        
    def tensor_exp(self, t):#4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)
        '''
        batch = t.shape[0]
        epochs = t.shape[1]
        u, s, v = self.svd(t.view(batch * epochs, t.shape[2], t.shape[3]))
        u = u.view(batch, epochs, u.shape[1], u.shape[2])
        s = s.view(batch, epochs, s.shape[1])
        v = v.view(batch, epochs, v.shape[1], v.shape[2])
        return u @ torch.diag_embed(torch.exp(s)) @ v.permute(0, 1, 3, 2)
        '''
    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        batch = inner_multi.shape[0]
        epochs = inner_multi.shape[1]
        _, s, _= self.svd(inner_multi.view(batch * epochs, inner_multi.shape[2], inner_multi.shape[3]))
        s = s.view(batch, epochs, s.shape[1])
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov#[bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)
        
    def forward(self, x, shape=None):
        if len(x.shape)==3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)# patch:[b, #patch, c, c]
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # Don't need to be symmetric
        #assert torch.allclose(Q, Q.mT, atol=(10 ** -10)), f"Q: {Q}"
        #assert torch.allclose(K, K.mT, atol=(10 ** -10)), "K"
        #assert torch.allclose(V, V.mT, atol=(10 ** -10)), "V"

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
    
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1 )
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4])
        
        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)#now row is c.c.

        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape
