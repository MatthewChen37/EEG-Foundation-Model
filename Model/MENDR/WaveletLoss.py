import torch

loss_fn = torch.nn.MSELoss()

def WaveletReconstructionLoss(inputs, outputs):
    for band, input_data in inputs.items():
        if isinstance(input_data, torch.Tensor):
            inputs[band] = input_data.float()
    for band, output_data in outputs.items():
        if isinstance(output_data, torch.Tensor):
            outputs[band] = output_data.float()

    '''
    assert inputs['delta'].shape == outputs['delta'].shape, f"Input Shape {inputs['delta'].shape} Output Shape {outputs['delta'].shape}"
    assert inputs['theta'].shape == outputs['theta'].shape, f"Input Shape {inputs['theta'].shape} Output Shape {outputs['theta'].shape}"
    assert inputs['alpha'].shape == outputs['alpha'].shape, f"Input Shape {inputs['alpha'].shape} Output Shape {outputs['alpha'].shape}"
    assert inputs['beta'].shape == outputs['beta'].shape, f"Input Shape {inputs['beta'].shape} Output Shape {outputs['beta'].shape}"
    assert inputs['gamma'].shape == outputs['gamma'].shape, f"Input Shape {inputs['gamma'].shape} Output Shape {outputs['gamma'].shape}"
    '''

    delta_loss = loss_fn(inputs['delta'], outputs['delta']) + fft_loss(inputs['delta'], outputs['delta']) 
    theta_loss = loss_fn(inputs['theta'], outputs['theta'])  + fft_loss(inputs['theta'], outputs['theta'])
    alpha_loss = loss_fn(inputs['alpha'], outputs['alpha'])  + fft_loss(inputs['alpha'], outputs['alpha'])
    beta_loss =  loss_fn(inputs['beta'],  outputs['beta'])  + fft_loss(inputs['beta'], outputs['beta'])
    gamma_loss = loss_fn(inputs['gamma'], outputs['gamma'])  + fft_loss(inputs['gamma'], outputs['gamma'])

    loss_dict = {
        'delta': delta_loss.item(),
        'theta': theta_loss.item(),
        'alpha': alpha_loss.item(),
        'beta': beta_loss.item(),
        'gamma': gamma_loss.item(),
    }

    loss = delta_loss + theta_loss + alpha_loss + beta_loss + gamma_loss
    loss = loss.to(torch.float32)
    return loss, loss_dict

def fft_loss(output, target):
    output_fft = torch.fft.fft(output, dim=-1)
    output_amplitude = torch.abs(output_fft)
    output_angle = torch.angle(output_fft)

    target_fft = torch.fft.fft(target, dim=-1)
    target_amplitude = torch.abs(target_fft)
    target_angle = torch.abs(target_fft)

    return (loss_fn(output_amplitude, target_amplitude) + (loss_fn(output_angle, target_angle)))