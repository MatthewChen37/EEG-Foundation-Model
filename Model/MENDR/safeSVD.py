import numpy as np
import torch


# Bless this https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/trg.py#L3-L4
# From https://github.com/jax-ml/jax/issues/2311
# Sometimes the singular values of the covariance matrix could be degenerate
# which causes the gradient during backpropagation to explode. There is nothing 
# that can be done about it because it is a characteristic of the data:
# https://discuss.pytorch.org/t/function-linalgsvdbackward0-returned-nan-values-in-its-0th-output/190410
# To handle this case, we do a safe inverse which adds a small epsilon
def safe_inverse(x, epsilon=1E-12):
	return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
	@staticmethod
	def forward(self, A):
		U, S, V = torch.svd(A)
		self.save_for_backward(U, S, V)
		return U, S, V
	'''
	@staticmethod
	def backward(self, dU, dS, dV):
		U, S, V = self.saved_tensors
		Vt = V.t()
		Ut = U.t()
		M = U.size(0)
		N = V.size(0)
		NS = len(S)

		F = (S - S[:, None])
		F = safe_inverse(F)
		F.diagonal().fill_(0)

		G = (S + S[:, None])
		G.diagonal().fill_(np.inf)
		G = 1/G 

		UdU = Ut @ dU
		VdV = Vt @ dV

		Su = (F+G)*(UdU-UdU.t())/2
		Sv = (F-G)*(VdV-VdV.t())/2

		dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
		if (M>NS):
			dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
		if (N>NS):
			dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
		return dA
	'''

	@staticmethod
	# Must be a decomposition of batches of matrices
	def backward(self, dU, dS, dV):
		U, S, V = self.saved_tensors
		# U: [B, N, N], S: [B, N], V: [B, N, N]
		Vt = V.permute(0, 2, 1)
		Ut = U.permute(0, 2, 1)
		M = U.size(1)
		N = V.size(1)
		NS = len(S[1])

		F = (S[..., None, :] - S[..., None])
		F = safe_inverse(F)
		F.diagonal().fill_(0)

		G = (S[..., None, :] + S[..., None])
		G.diagonal().fill_(np.inf)
		G = 1/G 

		UdU = Ut @ dU
		VdV = Vt @ dV

		Su = (F+G)*(UdU-UdU.permute(0, 2, 1))/2
		Sv = (F-G)*(VdV-VdV.permute(0, 2, 1))/2

		dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vt 
		return dA


# From https://arxiv.org/abs/2104.03821
class svdv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        u, s, v = torch.svd(M)  # s in a descending sequence.
        s = torch.clamp(s, min=1e-10)  # 1e-5
        ctx.save_for_backward(M, u, s, v)
        return u, s, v

    @staticmethod
    def backward(ctx, dL_du, dL_ds, dL_dv):
        M, u, s, v = ctx.saved_tensors
        K_t = geometric_approximation(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + torch.diag(dL_ds)).mm(u_t)
        return dL_dM

def geometric_approximation(s):
	dtype = s.dtype
	I = torch.eye(s.shape[0], device=s.device).type(dtype)
	p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
	p = torch.where(p < 1., p, 1. / p)
	a1 = s.repeat(s.shape[0], 1).t()
	a1_t = a1.t()
	a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
	a1 *= torch.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
	p_app = torch.ones_like(p)
	p_hat = torch.ones_like(p)
	for i in range(9):
		p_hat = p_hat * p
		p_app += p_hat
	a1 = a1 * p_app
	return a1