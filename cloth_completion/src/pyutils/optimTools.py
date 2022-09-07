import torch
import numpy as np

def runLBFGSClosure(optimizer, closure):
    max_iter = optimizer.param_groups[0]['max_iter']
    ftol = optimizer.param_groups[0]['tolerance_change']

    for n in range(max_iter):
        loss = optimizer.step(closure)
        loss_value = loss.detach().cpu().numpy().item()
        print('Iteration: {}, loss: {}'.format(n, loss_value))

        if n > 0 and prev_loss is not None:
            loss_change = np.abs(prev_loss - loss_value)
            if loss_change <= ftol:
                break

        prev_loss = loss_value


def runSGDClosure(optimizer, closure, max_iter, tolerance_change):
    assert tolerance_change > 0.0

    for n in range(max_iter):
        loss = optimizer.step(closure)
        loss_value = loss.detach().cpu().numpy().item()
        print('Iteration: {}, loss: {}'.format(n, loss_value))

        if n > 0 and prev_loss is not None:
            loss_change = np.abs(prev_loss - loss_value)
            if loss_change <= tolerance_change:
                break

        prev_loss = loss_value
