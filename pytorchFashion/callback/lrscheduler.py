#encoding:utf-8
class StepLr(object):
    def __init__(self,optimizer,lr):
        super(StepLr,self).__init__()
        self.optimizer = optimizer
        self.lr = lr

    def step(self,epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr