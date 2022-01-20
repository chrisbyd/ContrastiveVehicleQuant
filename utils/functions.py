import torch

def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), dim=1))

def intra_norm(feature, len_code, n_book):
    x = torch.split(feature, len_code, 1)

    for i in range(n_book):
        norm_tmp = torch.nn.functional.normalize(x[i], 1)
        if i == 0:
            innorm = norm_tmp
        else:
            innorm = torch.cat([innorm, norm_tmp], 1)
    return innorm




def my_soft_assignment(z_, x_, len_code, alpha, device):
    x = torch.split(x_, len_code, 1)
    y = torch.split(z_, len_code, 1)
    n_book = x_.shape[1] // len_code
    for i in range(n_book):
        xx = x[i]
        yy = y[i]
        diff = squared_distances(xx,yy)
        softmax_diff = torch.softmax(diff*(-alpha), 1)
        if i == 0:
            soft_des_tmp = torch.matmul(softmax_diff, yy)
            descriptor = soft_des_tmp
        else:
            soft_des_tmp = torch.matmul(softmax_diff, yy)
            descriptor = torch.cat([descriptor, soft_des_tmp], 1)
    return descriptor


def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, -1)

def norm_gallery(Z, descriptor, len_code, n_book):
    x = torch.split(descriptor, len_code, 1) # K[B, L]
    y = torch.split(Z, len_code, 1) # K [W, L]
    res = torch.zeros((descriptor.shape[0], 1))
    #print('gallery', Z.shape, descriptor.shape)
    code_len = int(descriptor.shape[1] // n_book)
    for n in range(n_book):
        xx = x[n] # [B, L, 1]
        yy = y[n] # [B, L, W]

        diff = - squared_distances(xx,yy)

        arg = torch.argmax(diff, dim=1)
        hit_code = y[n][arg]
        if n == 0:
            res = hit_code
        else:
            res = torch.cat((res, hit_code), dim=1)
    return res

def pq_dist():
    pass



