import numpy as np
import torch
from math import log

from Model.MENDR.mAtt.spd import *
from Model.MENDR.mAtt.mAtt import *
import Model.MENDR.safeSVD as SVD
# From https://github.com/adavoudi/spdnet/blob/master/tests/test_modules.py

class CTX:
    def __init__(self, saved_variables, needs_input_grad):
        self.saved_variables = saved_variables
        self.needs_input_grad = needs_input_grad

def assertTensorEqual(a, b, tolerance=1e-4):
    return (a.sub(b).abs().max() < tolerance).data.item() == 1

spd = torch.from_numpy(np.asarray([
    [4.2051,1.1989,0.6229],
    [1.1989,4.1973,0.6028],
    [0.6229,0.6028,3.5204]
], np.float32))
spd = spd.unsqueeze(0)

grad_mat = torch.from_numpy(np.asarray([
    [1,1,1],
    [1,1,1],
    [1,1,1]
], np.float32))
grad_mat = grad_mat.unsqueeze(0)

def check_TangentSpace():
    desired_forward = torch.from_numpy(np.asarray([
        [1.3848,0.2849,0.1444],
        [0.2849,1.3837,0.1383],
        [0.1444,0.1383,1.2355]
    ], np.float32))

    desired_backward = torch.from_numpy(np.asarray([
        [ 0.1618,0.1625,0.1885],
        [ 0.1625,0.1631,0.1894],
        [ 0.1885,0.1894,0.2222]
    ], np.float32))

    forward = SPDTangentSpaceFunction.apply(spd)
    backward = SPDTangentSpaceFunction.backward(CTX([spd], [True]), grad_mat)
    
    forward_eq = assertTensorEqual(forward, desired_forward)
    backward_eq = assertTensorEqual(backward, desired_backward)

    print("Tangent space:", forward_eq, backward_eq)

    return (forward_eq and backward_eq)

def check_Rectified():
    desired_forward = torch.from_numpy(np.asarray([
        [4.2562,1.1498,0.6185],
        [1.1498,4.2443,0.6070],
        [0.6185,0.6070,3.5207]
    ], np.float32))

    desired_backward = torch.from_numpy(np.asarray([
        [0.9991,1.0000,1.0055],
        [1.0000,1.0008,0.9948],
        [1.0055,0.9948,0.9991]
    ], np.float32))

    epsilon = torch.FloatTensor([3.1])

    forward = SPDRectifiedFunction.apply(spd, epsilon)
    backward = SPDRectifiedFunction.backward(CTX([spd, epsilon], [True, False]), grad_mat)[0]
    forward_eq = assertTensorEqual(forward, desired_forward)
    backward_eq = assertTensorEqual(backward, desired_backward)

    return (forward_eq and backward_eq)

def check_UnTangentSpace():
    tang = SPDTangentSpaceFunction.apply(spd)
    untang = SPDUnTangentSpaceFunction.apply(tang)

    transform_assert = assertTensorEqual(spd, untang, tolerance=1e-4)
    return transform_assert

def check_eigh():
    from types import SimpleNamespace

    # Eigenvalues are 1, 2, and 4
    simple_spd = torch.from_numpy(np.array([[[2, 1, 0], [1, 3, 1], [0, 1, 2]]], np.float32)).float()
    S, U = SVD.eigh.apply(simple_spd)
    eigenvalue_assert = assertTensorEqual(S, torch.tensor([1, 2, 4]))

    input_self = SimpleNamespace()
    input_self.saved_tensors = S, U
    dA = SVD.eigh.backward(input_self, dS=torch.ones(1, 3), dU=torch.ones(1, 3, 3))

    expected_dA = torch.Tensor([[ 1.0241,  0.3588, -0.2388],
                                [ 0.5418,  1.3974, -0.6550],
                                [-0.2561, -0.6464,  0.5880]])

    dA_assert = assertTensorEqual(expected_dA, dA)

    if not dA_assert:
        print(expected_dA, expected_dA.shape)
        print(dA, dA.shape)

    if not eigenvalue_assert:
        print(S)

    return eigenvalue_assert and dA_assert

def check_TensorLog():
    attention_manifold = AttentionManifold(3, 3, "cpu")

    # Eigenvalues are 1, 2, and 4
    simple_spd = torch.from_numpy(np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], np.float32)).double()

    simple_spd = simple_spd[None, None, ...]

    assert simple_spd.shape == torch.Size([1, 1, 3, 3]), f'Shape: {simple_spd,shape}'

    tensor_log = attention_manifold.tensor_log(simple_spd)

    # Expected (as per the original mATT implementation)
    u, s, v = torch.svd(simple_spd)
    expected = u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)

    tensor_log_assert = assertTensorEqual(tensor_log, expected)

    return tensor_log_assert

def check_LogEuclideanMean():
    attention_manifold = AttentionManifold(3, 3, "cpu")

    # Eigenvalues are 1, 2, and 4
    simple_spd = torch.from_numpy(np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], np.float32)).double()

    simple_spd = simple_spd.repeat(2, 2, 1, 1)

    assert simple_spd.shape == torch.Size([2, 2, 3, 3]), f'Shape: {simple_spd,shape}'

    mock_weights = torch.diag(torch.ones(2)).repeat(2, 1, 1).double()

    assert mock_weights.shape == torch.Size([2, 2, 2]), f'Mock Weights Shape: {mock_weights}'

    log_euclidean_mean = attention_manifold.LogEuclideanMean(mock_weights, simple_spd)

    lem_assert = assertTensorEqual(log_euclidean_mean, simple_spd)

    if not lem_assert:
        print("Failed:", log_euclidean_mean, simple_spd)

    return lem_assert

def check_CustomLogEuclideanMean():
    attention_manifold = AttentionManifold(3, 3, "cpu")

    # Eigenvalues are 1, 2, and 4
    simple_spd = torch.from_numpy(np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], np.float32)).double()

    simple_spd = simple_spd.repeat(2, 2, 1, 1)

    assert simple_spd.shape == torch.Size([2, 2, 3, 3]), f'Shape: {simple_spd,shape}'

    mock_weights = torch.diag(torch.ones(2)).repeat(2, 1, 1).double()

    log_euclidean_mean = attention_manifold.LogEuclideanMean(mock_weights, simple_spd)

    simple_spd_log = attention_manifold.tensor_log(simple_spd)

    simple_spd_log = simple_spd_log.sum(dim=1, keepdim=True) / simple_spd_log.shape[1]

    simple_spd_LEM = attention_manifold.tensor_exp((simple_spd_log))

    lem_assert = assertTensorEqual(log_euclidean_mean, simple_spd_LEM)

    return lem_assert

def check_NearestSymPosDef():

    simple_non_spd = torch.from_numpy(np.array([[2, 1, 0.1],
                                                [1, 3, 1],
                                                [0, 1, 2]], np.float32)).float()

    
    output = SVD._nearest_sym_pos_def(simple_non_spd)

    nearest_sym_pos_def_assert = torch.allclose(output, output.mT)

    return nearest_sym_pos_def_assert


def check_NearestSymPosDef_2():
    simple_non_spd = torch.from_numpy(np.array([[2, 1, 0.1],
                                                [1, 3, 1],
                                                [0, 1, 2]], np.float32)).float()

    simple_non_spd = simple_non_spd.repeat(4, 1, 1)
    
    output = SVD.nearest_sym_pos_def(simple_non_spd)

    nearest_sym_pos_def_assert = True
    for out in output:
        nearest_sym_pos_def_assert = nearest_sym_pos_def_assert and torch.allclose(out, out.mT)

    return nearest_sym_pos_def_assert



units = {
    'Tangent space layer': check_TangentSpace,
    'Rectification layer': check_Rectified,
    'Untangent space layer': check_UnTangentSpace,
    #'Check eigh': check_eigh,
    'Tensor Log': check_TensorLog,
    'LogEuclideanMean': check_LogEuclideanMean,
    'Custom LEM': check_CustomLogEuclideanMean,
    'Nearest Sym Pos Def': check_NearestSymPosDef,
    'Nearest Sym Pos Def 2': check_NearestSymPosDef_2,
}

result = True

print('Performing unit test ...')
for index, (name, func) in enumerate(units.items()):
    current_result = func()
    result = result and current_result
    print('[%d/%d] %s : %s' % (index+1, len(units), name, current_result))


if result:
    print('All tests passed')
else:
    print('test failed')