#encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.util import AverageMeter,prepare_device

class Trainer(object):
    def __init__(self,model,
                 train_data,
                 val_data,
                 optimizer,
                 logger,
                 config,
                 metric = None,
                 criterion = None,
                 lr_scheduler = None,
                 train_from_scratch=None,
                 model_checkpoint = None,
                 training_monitor=None,
                 early_stopping=None,
                 writer = None,
                 verbose = 1):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = config.EPOCHS
        self.optimizer = optimizer
        self.logger = logger
        self.verbose = verbose
        self.writer = writer
        self.start_epoch = config.START_EPOCH
        self.training_monitor = training_monitor
        self.early_stopping = early_stopping
        self.train_from_scratch = train_from_scratch
        self.model_checkpoint = model_checkpoint
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.metric = metric
        # 计算gpu情况
        self.device, self.device_ids = prepare_device(config.N_GPU,logger)
        # 数据大小
        self.batch_num = len(train_data)
        # 进度条显示
        self.progressbar = ProgressBar(n_batch = self.batch_num)

        #******** 单机多GPU情况 ***********
        #整个过程可以这么理解:
        #首先将模型加载到一个指定设备上作为controller,
        # 然后将模型浅复制到多个设备中，将大batch数据也
        #等分到不同设备中， 每个设备分配到不同的数据，然后将所有设备计算得到
        #梯度合并用以更新controller模型的参数。
        if len(self.device_ids) > 1:
            # model = nn.DataParallel(model) 会将模型浅复制到所有可用的显卡中
            # （如果是我实验室的服务器，就是复制到8张卡中）,我们希望只占用显卡1和3,
            # 所以需要传入参数device_ids=[1,3]
            self.model = torch.nn.DataParallel(self.model,device_ids=self.device_ids)
        # model = model.cuda() 会将模型加载到0号显卡并作为controller.
        # 但是我们并不打算使用0号显卡。
        # 所以需要修改为：model = model.cuda(device_ids[0]),
        # 即我们将模型加载1号显卡并作为controller
        self.model = self.model.to(self.device)

        # 重新加载模型
        # 如果train_from_scratch = True,则加载best模型
        # 如果train_from_scratch = 10，则表明加载epoch 10的checkpoint模型
        if self.train_from_scratch:
            # 传进来是字符串形式
            train_from_scratch = eval(self.train_from_scratch)
            # 如果为True，则加载best 模型
            if isinstance(train_from_scratch, bool):
                resume_path = os.path.join(model_checkpoint.checkpoint_dir,'{}-model_best.pth'.format(config.ARCH))
            else:
                # 如果为数字，则加载制定epoch 模型
                resume_path = os.path.join(model_checkpoint.checkpoint_dir,
                                        '{}-checkpoint-epoch{}.pth'.format(config.ARCH, train_from_scratch))
            self._restore_checkpoint(resume_path = resume_path)

    # 查看模型结构
    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self.model)

    # 保存checkpoint信息
    def _save_info(self,epoch):
        state = {
            'arch': self.config.ARCH,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': {key:value for key,value in self.config.__dict__.items() if '__' not in key and key !='path'}
        }
        return state

    # 加载checkpoint模型
    def _restore_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        if self.model_checkpoint:
            self.model_checkpoint.best = checkpoint['best']
        # 判断模型结构是否对应
        if checkpoint['config']['arch'] != self.config.ARCH:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.optimizer['type']:
            self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                'Optimizer parameters not being resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    # 评估验证集
    def _valid_epoch(self,epoch):
        # eval()时，pytorch会自动把BN和DropOut固定住,
        # 不会取平均，而是用训练好的值.
        self.model.eval()
        val_losses = AverageMeter()
        val_acc = AverageMeter()
        # 不计算梯度
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                acc = self.metric(output = output,target=target)
                val_losses.update(loss.item(),data.size(0))
                val_acc.update(acc.item(),data.size(0))
            # 写入文件中
            self.writer.set_step(epoch, 'valid')
            self.writer.add_scalar('val_loss', val_losses.avg)
            self.writer.add_scalar('val_acc', val_acc.avg)
        return {
            'val_loss': val_losses.avg,
            'val_acc': val_acc.avg
        }

    # 训练数据集
    def _train_epoch(self,epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.train_data):
            start = time.time()
            # requires_grad 已经是Tensor的一个属性了
            # Tensor在0.4中，现在默认requires_grad=False的Variable了
            data = data.to(self.device)
            target = target.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(output = outputs,target=target)
            acc = self.metric(output=outputs,target=target)

            # 计算梯度并更新梯度
            #将上次迭代计算的梯度值清0
            self.optimizer.zero_grad()
            # 反向传播，计算梯度值
            # backward只能被应用在一个标量上，也就是一个一维tensor，或者传入跟变量相关的梯度
            loss.backward()
            # 更新权值参数
            self.optimizer.step()
            # 更新指标
            # 取得一个tensor的值(返回number), 用.item()
            train_loss.update(loss.item(),data.size(0))
            train_acc.update(acc.item(),data.size(0))
            # 是否打印训练过程
            if self.verbose >= 1:
                self.progressbar.step(epoch = epoch,
                                        batch_idx=batch_idx,
                                        loss  = loss.item(),
                                        acc = acc.item(),
                                        use_time = time.time() - start)
        # 写入tensorboard
        self.writer.set_step(epoch)
        self.writer.add_scalar('loss', train_loss.avg)
        self.writer.add_scalar('acc', train_acc.avg)

        # 训练log
        train_log = {
            'loss': train_loss.avg,
            'acc': train_acc.avg
        }
        return train_log

    # 拟合主函数
    def train(self):
        for epoch in range(self.start_epoch,self.epochs + 1):
            self.logger.info("\nEpoch {i}/{epochs}......".format(i=epoch, epochs=self.epochs))
            # 是否进行学习率的更新
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)
            logs = dict(train_log,**val_log)
            self.logger.info('\nEpoch: %d - loss: %.4f acc: %.4f - val_loss: %.4f - val_acc: %.4f'%(
                            epoch,logs['loss'],logs['acc'],logs['val_loss'],logs['val_acc'])
                             )
            # 本地保存训练过程
            if self.training_monitor:
                self.training_monitor.step(logs)
            # 当满足early_stopping时，停止训练
            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            # checkpoint
            if self.model_checkpoint:
                # 保存信息
                state = self._save_info(epoch)
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)