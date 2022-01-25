import torch


    
def minibatch_sample(sample_f, num_samples, dimensions, batch_size, device=torch.device('cpu'), context=None, x=None):
    
    if x is not None:
        ld = x[0].shape[0]
    elif context is not None:
        ld = context.shape[0]
    else:
        ld = 1
    
    samples = torch.zeros((ld, num_samples, dimensions), device=device)

    for i in range(-(-num_samples // batch_size)):
        start = i * batch_size
        stop = (i + 1) * batch_size
        n = min(batch_size, num_samples - start)
        if x is None:
            samples[:, start:stop, :] = torch.atleast_3d(sample_f(n, context=context).to(device).T).T
        else:
            samples[:, start:stop, :] = torch.atleast_3d(sample_f(x, n, context=context).to(device).T).T
        
    return samples