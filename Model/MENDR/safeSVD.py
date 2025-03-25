import numpy as np
import torch
import torch.nn as nn


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

class robust_svd(nn.Module):

	def __init__(self):
		super(robust_svd, self).__init__()
	
	def forward(self, input):
		return svdv2.apply(input)

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

# Correction of numerically imprecise SPD matrices in a batch
# PyTorch port of: https://github.com/pyRiemann/pyRiemann/blob/f94d993a4fdf3c9e0865fe8e4d2a36895567dfeb/pyriemann/utils/base.py#L160
# Based on https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
def _nearest_sym_pos_def(S, reg=1e-6):
	"""Find the nearest SPD matrix.

	Parameters
	----------
	S : torch.tensor, shape (n, n)
		Square matrix.
	reg : float, default=1e-6
		Regularization parameter.

	Returns
	-------
	P : torch.tensor, shape (n, n)
		Nearest SPD matrix.
	"""
	svd = SVD.apply

	def regularize(X, reg):
		ei, ev = torch.linalg.eigh(X)
		if torch.min(ei) / torch.max(ei) < reg:
			X = ev @ torch.diag(ei + reg) @ ev.T
		return X

	A = (S + S.T) / 2
	_, s, V = svd(A)
	H = V.T @ (s[:, np.newaxis] * V) # np.newaxis works on torch tensors
	B = (A + H) / 2
	P = (B + B.T) / 2

	if is_pos_def(P):
		# Regularize if already PD
		return regularize(P, reg)

	spacing = torch.finfo(torch.linalg.norm(A).dtype).eps
	I = torch.eye(S.shape[0])  # noqa
	k = 1
	while not is_pos_def(P):
		mineig = torch.min(torch.real(torch.linalg.eigvals(P)))
		P += I * (-mineig * k ** 2 + spacing)
		k += 1

	# Regularize
	return regularize(P, reg)

def is_pos_def(X, tol=0.0):
	"""Check if all matrices are positive definite (PD).

	Parameters
	----------
	X : torch.Tensor, shape (..., n, n)
		The set of square matrices, at least 2D ndarray.
	tol : float, default 0.0
		Threshold below which eigen values are considered zero.
	Returns
	-------
	ret : bool
		True if all matrices are positive definite.
	"""
	square = is_square(X)
	eig_vals = _get_eigenvals(X)
	if not torch.isreal(eig_vals).all(): # Torch eigenvalues always returns a complex tensor 
		return False
	eig_vals = torch.real(eig_vals)
	positive_eig = torch.all(torch.real(eig_vals) > tol)
	return square and positive_eig

def _get_eigenvals(X):
	"""Private function to compute all eigen values."""
	n = X.shape[-1]
	X_reshaped = X.reshape((-1, n, n))
	eigvals = torch.linalg.eigvals(X_reshaped)
	return eigvals

def is_square(X):
    """Check if matrices are square.

    Parameters
    ----------
    X : torch.Tensor, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.
    Returns
    -------
    ret : bool
        True if matrices are square.
    """
    return X.ndim >= 2 and X.shape[-2] == X.shape[-1]