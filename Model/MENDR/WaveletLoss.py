import torch

loss_fn = torch.nn.MSELoss()


def WaveletReconstructionLoss(inputs, outputs):
    delta_loss = loss_fn(inputs['delta'], outputs['delta']) + fft_loss(inputs['delta'], outputs['delta']) 
    theta_loss = loss_fn(inputs['theta'], outputs['theta'])  + fft_loss(inputs['theta'], outputs['theta'])
    alpha_loss = loss_fn(inputs['alpha'], outputs['alpha'])  + fft_loss(inputs['alpha'], outputs['alpha'])
    beta_loss =  loss_fn(inputs['beta'],  outputs['beta'])  + fft_loss(inputs['beta'], outputs['beta'])
    gamma_loss = loss_fn(inputs['gamma'], outputs['gamma'])  + fft_loss(inputs['gamma'], outputs['gamma'])
    loss = delta_loss + theta_loss + alpha_loss + beta_loss + gamma_loss
    return loss

def fft_loss(output, target):
    output_fft = torch.fft.fft(output, dim=-1)
    output_amplitude = torch.abs(output_fft)
    output_angle = torch.angle(output_fft)

    target_fft = torch.fft.fft(target, dim=-1)
    target_amplitude = torch.abs(target_fft)
    target_angle = torch.abs(target_fft)

    return (loss_fn(output_amplitude, target_amplitude) + (loss_fn(output_angle, target_angle))) * 1e-3