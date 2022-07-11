class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale, decay_lr_at, init_lr, loop, warmup_iters):
  if loop < warmup_iters:
    lr = 0.0001 + (loop) * ((0.01 - 0.0001) / warmup_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
  elif warmup_iters < loop < decay_lr_at[0]:
    for param_group in optimizer.param_groups:
      param_group['lr'] = init_lr
  elif decay_lr_at[0] < loop < decay_lr_at[1]:
    for param_group in optimizer.param_groups:
      param_group['lr'] = init_lr*scale
  else:
    for param_group in optimizer.param_groups:
      param_group['lr'] = init_lr*scale

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)