import torch
import torch.nn as nn
from .spd import SPDTransform, SPDTangentSpace, SPDRectified
from ..safeSVD import SVD

'''
Modified from https://github.com/CECNL/MAtt/blob/main/mAtt/mAtt.py
'''
svd = SVD.apply

class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
    def forward(self, x):
        dev = x.device
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        # To avoid division by 0 error
        tra = tra + (1e-7)*torch.ones(tra.shape).to(tra.device)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=dev).repeat(x.shape[0], 1, 1)
        # Notice how they also added 1e-5 originally
        cov = cov+(1e-7*identity)
        return cov 

class E2R(nn.Module):
    def __init__(self):
        super().__init__()
        self.signal2spd = signal2spd()
    def forward(self, x):
        # X is with shape [Batch, #patch, #encoded_h, #time_step]
        x_list = list(x.unbind(1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x

class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size, heads=1):
        super().__init__()
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.heads = heads

        self.q_trans = SPDTransform(self.heads*self.d_in, self.heads*self.d_out)
        self.k_trans = SPDTransform(self.heads*self.d_in, self.heads*self.d_out)
        self.v_trans = SPDTransform(self.heads*self.d_in, self.heads*self.d_out)
        #self.project_out = SPDTransform(self.heads*self.d_out, self.d_out)

    '''
    def forward(self, x, shape=None):
        if len(x.shape)==3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)

        x = x.to(torch.float) # patch:[b, #patch, c, c]
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        x = block_diag(x, self.heads)

        # repeat 
        Q = self.q_trans(x)
        K = self.k_trans(x)
        V = self.v_trans(x)

        Q = unblock_diag(Q, self.heads).view(bs, m, self.heads, self.d_out, self.d_out).permute(0, 2, 1, 3, 4).contiguous().view(bs*self.heads, m, self.d_out, self.d_out)
        K = unblock_diag(K, self.heads).view(bs, m, self.heads, self.d_out, self.d_out).permute(0, 2, 1, 3, 4).contiguous().view(bs*self.heads, m, self.d_out, self.d_out)
        V = unblock_diag(V, self.heads).view(bs, m, self.heads, self.d_out, self.d_out).permute(0, 2, 1, 3, 4).contiguous().view(bs*self.heads, m, self.d_out, self.d_out)

        # Don't need to be symmetric
        #assert torch.allclose(Q, Q.mT, atol=(10 ** -10)), f"Q: {Q}"
        #assert torch.allclose(K, K.mT, atol=(10 ** -10)), "K"
        #assert torch.allclose(V, V.mT, atol=(10 ** -10)), "V"
    
        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4])

        atten_energy = log_euclidean_distance(Q_expand, K_expand).view(bs*self.heads, m, m)
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)

        # calculate outputs(v_i') of attention module
        output = LogEuclideanMean(atten_prob, V).view(bs, self.heads, m, self.d_out, self.d_out).permute(0, 2, 1, 3, 4).contiguous()
        output = reblock_diag(output, self.heads)
        output = output.contiguous().view(-1, self.heads*self.d_out, self.heads*self.d_out)
        output = self.project_out(output) # Removes head dimensions
        return output, (bs, m, -1)
    '''

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

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
    
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1 )
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4])
        
        atten_energy = log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)#now row is c.c.
        
        # calculate outputs(v_i') of attention module
        output = LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape

def tensor_log(t):
    '''
    output = torch.zeros(t.shape).to(self.device)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            u, s, v = self.svd(t[i, j, :, :])
            output[i, j] = u @ torch.diag_embed(torch.log(s)) @ v.permute(1, 0)
    return output
    '''
    patch_num = None
    if len(t.shape) == 4:
        batch_size = t.shape[0]
        patch_num = t.shape[1]
        channel_num = t.shape[2]
        t = t.reshape(batch_size*patch_num, channel_num, channel_num)
    #s, u = torch.linalg.eigh(t)
    u, s, v = svd(t)
    output = u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 2, 1)
    if patch_num is not None:
        return output.reshape(batch_size, patch_num, channel_num, channel_num)
    else:
        return output
    # condition: t is symmetric!
    #s, u = torch.linalg.eigh(t)
    #print(s, u)
    #print(torch.linalg.norm(u[0, 0, :, :,], dim=0))
    #print(u.shape)
    #return u @ torch.diag_embed(torch.log(s)) @ u.permute(0, 1, 3, 2)
    #u, s, v = torch.svd(t)
    #return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

# https://github.com/pytorch/pytorch/issues/105225
# Not sure if this works so still writing a custom implementation
# This also allows for more control over the computation 
def tensor_exp(t):#4dim
    # condition: t is symmetric!
    patch_num = None
    if len(t.shape) == 4:
        batch_size = t.shape[0]
        patch_num = t.shape[1]
        channel_num = t.shape[2]
        t = t.reshape(batch_size*patch_num, channel_num, channel_num)
    #s, u = torch.linalg.eigh(t)
    u, s, v = svd(t)
    output = u @ torch.diag_embed(torch.exp(s)) @ v.permute(0, 2, 1)
    if patch_num is not None:
        return output.reshape(batch_size, patch_num, channel_num, channel_num)
    else:
        return output
    '''
    batch = t.shape[0]
    epochs = t.shape[1]
    u, s, v = self.svd(t.view(batch * epochs, t.shape[2], t.shape[3]))
    u = u.view(batch, epochs, u.shape[1], u.shape[2])
    s = s.view(batch, epochs, s.shape[1])
    v = v.view(batch, epochs, v.shape[1], v.shape[2])
    return u @ torch.diag_embed(torch.exp(s)) @ v.permute(0, 1, 3, 2)
    '''
def log_euclidean_distance(A, B):
    inner_term = tensor_log(A) - tensor_log(B)
    inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
    batch = inner_multi.shape[0]
    epochs = inner_multi.shape[1]
    #_, s, _= self.svd(inner_multi.view(batch * epochs, inner_multi.shape[2], inner_multi.shape[3]))
    s = torch.linalg.svdvals(inner_multi.view(batch * epochs, inner_multi.shape[2], inner_multi.shape[3]))
    s = s.view(batch, epochs, s.shape[1])
    final = torch.sum(s, dim=-1)
    return final

# This is a WEIGHTED version of the LogEuclideanMean
def LogEuclideanMean(weight, cov):
    # cov:[bs, #p, s, s]
    # weight:[bs, #p, #p]
    bs = cov.shape[0]
    num_p = cov.shape[1]
    size = cov.shape[2]
    cov = tensor_log(cov).view(bs, num_p, -1)
    output = weight @ cov#[bs, #p, -1]
    output = output.view(bs, num_p, size, size)
    return tensor_exp(output)

def WaveletLogEuclideanMean(x):
    # x is dict where each entry is [Batch_Size * #patches, C, C]
    x_input = dict()
    if len(x['delta'].shape) == 4:
        for band in x.keys():
            x_input[band] = x[band].clone().reshape(x[band].shape[0]*x[band].shape[1], x[band].shape[2], x[band].shape[3])
    else:
        for band in x.keys():
            x_input[band] = x[band].clone()
    combined_manifold_output = torch.stack(list(x_input.values()), dim=1)
    # Combined Manifold Output is something like [Batch_num * Patches, # of Wavelet Bands, C, C]
    combined_manifold_output = tensor_log(combined_manifold_output)
    combined_manifold_output = tensor_exp((combined_manifold_output.sum(dim=1, keepdim=True)) / combined_manifold_output.shape[1])

    if len(x['delta'].shape) == 4:
        combined_manifold_output = combined_manifold_output.reshape(x['delta'].shape[0], x['delta'].shape[1],
                                                                    x['delta'].shape[2], x['delta'].shape[3])
    return combined_manifold_output


def block_diag(x, heads):
    # x is a tensor of shape [Batch * #patches, C, C]
    batch_size, c, _ = x.shape
    
    # Create a block diagonal tensor
    block_size = c * heads
    result = torch.zeros(batch_size, block_size, block_size, device=x.device, dtype=x.dtype)
    
    # Place each block on the diagonal
    for i in range(heads):
        start_idx = i * c
        end_idx = (i + 1) * c
        result[:, start_idx:end_idx, start_idx:end_idx] = x
    
    return result

def unblock_diag(x, heads):
    # x is a tensor of shape [Batch * #patches, heads*C, heads*C]
    batch_size, total_c, _ = x.shape
    c = total_c // heads
    
    # Extract diagonal blocks one by one
    result = torch.zeros(batch_size, heads, c, c, device=x.device, dtype=x.dtype)
    
    for i in range(heads):
        start_idx = i * c
        end_idx = (i + 1) * c
        result[:, i, :, :] = x[:, start_idx:end_idx, start_idx:end_idx]
    return result

def reblock_diag(x, heads):
    # x is a tensor of shape [Batch, #patches, heads, C, C]
    batch_size, patches, heads, c, _ = x.shape
    result = torch.zeros(batch_size, patches, heads*c, heads*c, device=x.device, dtype=x.dtype)

    for i in range(batch_size):
        for j in range(patches):
            for k in range(heads):
                start_idx = k * c
                end_idx = (k + 1) * c
                result[:, :, start_idx:end_idx, start_idx:end_idx] = x[i, j, k, :, :]
    return result

