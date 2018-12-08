#encoding:utf-8
from os import path

#****************** PATH **********************************
# 主路径
BASE_PATH = 'pytorchFashion'
# 训练数据集
TRAIN_PATH = path.sep.join([BASE_PATH,'dataset/train'])
#　验证数据集
VAL_PATH = path.sep.join([BASE_PATH,'dataset/val'])
#　预测数据集
TEST_PATH = path.sep.join([BASE_PATH,'dataset/test'])
# 模型运行日志
LOG_PATH = path.sep.join([BASE_PATH,'output'])
# 模型结果保存路径
JSON_PATH = path.sep.join([BASE_PATH,'output'])
# TSboard信息保存路径
WRITER_PATH = path.sep.join([BASE_PATH,'output/TSboard'])
# 模型保存路径
CHECKPOINT_PATH = path.sep.join([BASE_PATH,'output/checkpoints'])
# 图形保存路径
FIG_PATH = path.sep.join([BASE_PATH,'output'])


#****************** model config ******************
# 模型结构
ARCH = "CNN"
# 类别个数
NUM_CLASSES = 10

#  GPU个数
#  如果只写一个数字，则表示gpu标号从0开始，
#  并且默认使用gpu:0作为controller
#  如果以列表形式表示，即[1,3,5],则
#  我们默认list[0]作为controller

N_GPU = [0,1]

# 是否打乱数据
SHUFFLE = True
#batch的大小
BATCH_SIZE = 100
# 线程个数
NUM_WORKERS = 2
# 图像大小
IMAGE_SIZE = (227,227)
# epochs个数
EPOCHS = 15
START_EPOCH = 1
# 学习率
LEARNING_RATE = 0.001
# 动量
MOMENTUM = 0.9
#权重衰减因子
WEIGHT_DECAY = 1e-4
# 保存模型频率，当save_best_only为False时候，指定才有作用
SAVE_CHECKPOINTS_FREP = 20
# 是否重载模型
# 如果resume为True，表示加载bets模型
# 如果resume为epoch，则表示加载对应的epoch的checkpoint模型
RESUME = False
WORLD_SIZE = 1 # number pf nodes for distributed training
MODE = 'min'
MONITOR = 'val_loss'
# 标准化
MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]
# TOP
TOPK = 1
# early_stopping
PATIENCE = 20

# model checkpoint
SAVE_BEST_ONLY = True
