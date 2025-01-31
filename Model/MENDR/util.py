import numpy as np
import torch


# Bless this https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/trg.py#L3-L4
# From https://github.com/jax-ml/jax/issues/2311
def safe_inverse(x, epsilon=1E-12):
	return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
	@staticmethod
	def forward(self, A):
		U, S, V = torch.svd(A)
		self.save_for_backward(U, S, V)
		return U, S, V

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