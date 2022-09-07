import torch
def fnrmls(v, f):
    '''

    :param v: B x V x 3
    :param f: F x 3
    :param normalized:
    :return:
    '''
    a = v[:, f[:, 0], :]
    b = v[:, f[:, 1], :]
    c = v[:, f[:, 2], :]
    fn = torch.cross(b - a, c - a, dim=2)
    fn = torch.nn.functional.normalize(fn, dim=2)
    return fn