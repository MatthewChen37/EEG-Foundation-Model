import torch
loss_fn = torch.nn.MSELoss()

def WaveletReconstructionLoss(inputs, outputs, loss_type='real'):
    '''
    for band, input_data in inputs.items():
        if isinstance(input_data, torch.Tensor):
            inputs[band] = input_data.float()
    for band, output_data in outputs.items():
        if isinstance(output_data, torch.Tensor):
            outputs[band] = output_data.float()

    assert inputs['delta'].shape == outputs['delta'].shape, f"Input Shape {inputs['delta'].shape} Output Shape {outputs['delta'].shape}"
    assert inputs['theta'].shape == outputs['theta'].shape, f"Input Shape {inputs['theta'].shape} Output Shape {outputs['theta'].shape}"
    assert inputs['alpha'].shape == outputs['alpha'].shape, f"Input Shape {inputs['alpha'].shape} Output Shape {outputs['alpha'].shape}"
    assert inputs['beta'].shape == outputs['beta'].shape, f"Input Shape {inputs['beta'].shape} Output Shape {outputs['beta'].shape}"
    assert inputs['gamma'].shape == outputs['gamma'].shape, f"Input Shape {inputs['gamma'].shape} Output Shape {outputs['gamma'].shape}"
    '''

    if loss_type == 'real':
        delta_real_loss = loss_fn(inputs['delta'], outputs['delta'])
        theta_real_loss = loss_fn(inputs['theta'], outputs['theta']) 
        alpha_real_loss = loss_fn(inputs['alpha'], outputs['alpha'])
        beta_real_loss =  loss_fn(inputs['beta'],  outputs['beta'])
        gamma_real_loss = loss_fn(inputs['gamma'], outputs['gamma'])

        real_loss_dict = {
            'delta_real_loss': delta_real_loss.item(),
            'theta_real_loss': theta_real_loss.item(),
            'alpha_real_loss': alpha_real_loss.item(),
            'beta_real_loss': beta_real_loss.item(),
            'gamma_real_loss': gamma_real_loss.item(),
        }

        return real_loss_dict

    elif loss_type == 'fft':
        delta_fft_loss = fft_loss(inputs['delta'], outputs['delta'])
        theta_fft_loss = fft_loss(inputs['theta'], outputs['theta'])
        alpha_fft_loss = fft_loss(inputs['alpha'], outputs['alpha'])
        beta_fft_loss = fft_loss(inputs['beta'], outputs['beta'])
        gamma_fft_loss = fft_loss(inputs['gamma'], outputs['gamma'])

        fft_loss_dict = {
            'delta_fft_loss': delta_fft_loss.item(),
            'theta_fft_loss': theta_fft_loss.item(),
            'alpha_fft_loss': alpha_fft_loss.item(),
            'beta_fft_loss': beta_fft_loss.item(),
            'gamma_fft_loss': gamma_fft_loss.item(),
        }

        return fft_loss_dict
    
    # Otherwise combine real and fft losses
    loss_dict = real_loss_dict | fft_loss_dict # New python 3.9 syntax
    return loss_dict

def fft_loss(output, target):
    hanning_window = torch.hann_window(output.shape[-1]).to(device)
    output_windowed = output.clone() * hanning_window.expand_as(output)
    target_windowed = target.clone() * hanning_window.expand_as(target)

    output_fft = torch.fft.fft(output_windowed, dim=-1)
    output_amplitude = torch.abs(output_fft)
    output_angle = torch.angle(output_fft)

    target_fft = torch.fft.fft(target_windowed, dim=-1)
    target_amplitude = torch.abs(target_fft)
    target_angle = torch.abs(target_fft)

    return (loss_fn(output_amplitude, target_amplitude) + (loss_fn(output_angle, target_angle)))