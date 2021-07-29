class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
    
    def report(self):
        return (self.sum / self.count)