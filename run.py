#encoding:utf-8
import argparse
import torch
import numpy as np
from torch.optim import Adam
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorchFashion.train.trainer import Trainer
from pytorchFashion.utils.logginger import init_logger
from pytorchFashion.config import simplenet_config as config
from pytorchFashion.train.losses import CrossEntropy
from pytorchFashion.train.metrics import Accuracy
from pytorchFashion.callback.lrscheduler import StepLr
from pytorchFashion.io.data_loader import ImageDataIter
from pytorchFashion.model.cnn.simplenet import SimpleNet
from pytorchFashion.callback.earlystopping import EarlyStopping
from pytorchFashion.callback.modelcheckpoint import ModelCheckpoint
from pytorchFashion.callback.trainingmonitor import TrainingMonitor
from pytorchFashion.callback.writetensorboard import WriterTensorboardX

# 主函数
def main():
    # 路径变量
    checkpoint_dir = config.CHECKPOINT_PATH # checkpoint路径
    fig_path = config.FIG_PATH
    json_path = config.JSON_PATH
    # 初始化日志
    logger = init_logger(log_name=config.ARCH,
                         log_path=config.LOG_PATH)
    if args['seed'] is not None:
        logger.info("seed is %d"%args['seed'])
        np.random.seed(args['seed'])
        torch.manual_seed(args['seed'])
    # 加载数据集
    logger.info('starting load train data from disk')
    # trainIter = ImageDataIter(data_dir=config.TRAIN_PATH,
    #                           image_size = config.IMAGE_SIZE,
    #                           batch_size = config.BATCH_SIZE,
    #                           resize = 256,
    #                           random_crop= True,
    #                           # horizontallyFlip=True,
    #                           normailizer = {'mean':config.MEAN,'std':config.STD},
    #                           shuffle=True,
    #                           num_workers=config.NUM_WORKERS,
    #                           mode = 'train').data_loader()
    # Loading dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.FashionMNIST(root='./dataset/fashion',
                                          train=True,
                                          download=True,
                                          transform=transform)
    test_dataset = datasets.FashionMNIST(root='./dataset/fashion',
                                         train=False,
                                         download=True,
                                         transform=transform)
    # Loading dataset into dataloader
    trainIter = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True)
    valIter = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=False)
    # # 验证数据集
    # logger.info('starting load val data from disk')
    # valIter = ImageDataIter(data_dir=config.VAL_PATH,
    #                           image_size = config.IMAGE_SIZE,
    #                           batch_size = config.BATCH_SIZE,
    #                           resize=256,
    #                           random_crop=True,
    #                           normailizer={'mean': config.MEAN, 'std': config.STD},
    #                           shuffle=False,
    #                           num_workers=config.NUM_WORKERS,
    #                           mode = 'val').data_loader()
    # 初始化模型和优化器
    logger.info("initializing model")
    model = SimpleNet(num_classes = config.NUM_CLASSES)
    optimizer = Adam(params = model.parameters(),
                     lr = config.LEARNING_RATE,
                     weight_decay=config.WEIGHT_DECAY,
                    )
    # 写入TensorBoard
    logger.info("initializing callbacks")
    write_summary = WriterTensorboardX(writer_dir=config.WRITER_PATH,
                                       logger = logger,
                                       enable=True)
    # 模型保存
    model_checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir,
                                       mode= config.MODE,
                                       monitor=config.MONITOR,
                                       save_best_only= config.SAVE_BEST_ONLY,
                                       arch = config.ARCH,
                                       logger = logger)
    # eraly_stopping功能
    early_stop = EarlyStopping(mode = config.MODE,
                               patience = config.PATIENCE,
                               monitor = config.MONITOR)
    # 监控训练过程
    train_monitor = TrainingMonitor(fig_path = fig_path,
                                    json_path = json_path,
                                    arch = config.ARCH)
    lr_scheduler = StepLr(optimizer=optimizer,lr = config.LEARNING_RATE)
    # 初始化模型训练器
    logger.info('training model....')
    trainer = Trainer(model = model,
                      train_data = trainIter,
                      val_data = valIter,
                      optimizer = optimizer,
                      criterion=CrossEntropy(),
                      metric = Accuracy(topK=config.TOPK),
                      logger = logger,
                      config = config,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      early_stopping = early_stop,
                      writer= write_summary,
                      train_from_scratch=config.RESUME,
                      lr_scheduler=lr_scheduler
                      )
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s','--seed',default=1024,type = int,
                        help = 'seed for initializing training.')
    args = vars(ap.parse_args())
    main()
