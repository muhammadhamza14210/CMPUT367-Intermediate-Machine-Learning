class CyclicLRScheduler:
    def __init__(self, base_lr, max_lr, step_size_up, mode='triangular'):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.cycle_iter = 0
        self.lr = base_lr
    
    def get_lr(self):
        cycle = int(1 + self.cycle_iter / (2 * self.step_size_up))
        x = abs(self.cycle_iter / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_fn = lambda x: 1
        elif self.mode == 'triangular2':
            scale_fn = lambda x: 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_fn = lambda x: 0.99994 ** self.cycle_iter
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_fn(x)
        
        self.cycle_iter += 1
        return lr
    
    def reset(self):
        self.cycle_iter = 0
        self.lr = self.base_lr

    def update_optimizer_lr(self, optimizer):
        new_lr = self.get_lr()
        optimizer.set_learning_rate(new_lr)
