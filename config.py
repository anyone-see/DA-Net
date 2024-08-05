
class Config(object):
    def __init__(self):

        # 学习参数
        self.learning_rate_decrease_itr = 15  # 学习率减少迭代次数
        self.decay_factor = 0.94  # 衰减因子
        self.lr = 0.0001  # 学习率
        self.weight_decay = 0.0001  # 权重衰减
        self.beta1 = 0.9  # Adam优化器的β1值

        # 训练参数
        self.loss = 'silog_loss'  # 损失函数 LogDepthLoss L1Loss silog_loss
        self.variance_focus = 0.85
        self.optimizer = 'adam'  # 优化器选w
        self.batch_size = 32  # 批处理大小
        self.epochs = 200  # 训练的总循环周期
        self.modo = 'train'  # 模式选择
        self.display_freq = 30000  # 显示频率
        self.validation_freq = 30000  # 验证频率

        # 数据集参数
        self.dataset = 'BV1'  # 数据集名称
        self.audio_resize = False  # 音频裁减成与图片大小一致
        self.audio_normalization = False  # 音频归一化
        # 硬件和运行参数
        self.device = 'cuda'  # 设备选择使用GPU
        self.num_workers = 0  # 线程数
        self.replica_dataset_path = '/home/public/replica-dataset'  # replica数据集路径
        self.mp3d_dataset_path = '/home/public/mp3d-dataset'  # mp3d数据集路径
        self.bv1_dataset_path='/home/public/BatvisionV1'
        self.bv2_dataset_path='/home/public/BatvisionV2'

        if self.dataset == 'replica':
            self.max_depth = 14.104  # replica数据集最大深度
            self.audio_shape = [2, 257, 166]  # replica音频形状
            self.input_size = 128
            self.metadata_path = '/home/public/metadata/replica'  # mp3d数据集划分文件路径

        elif self.dataset == 'mp3d':
            self.max_depth = 10.0  # mp3d数据集最大深度
            self.audio_shape = [2, 257, 121]  # mp3d音频形状
            self.input_size = 128
            self.metadata_path = '/home/public/metadata/mp3d'  # mp3d数据集划分文件路径

        elif self.dataset == 'mp3d_mini':
            self.max_depth = 10.0  # mp3d数据集最大深度
            self.audio_shape = [2, 257, 121]  # mp3d音频形状
            self.input_size = 128

        elif self.dataset == 'BV2':
            self.max_depth = 30.0
            self.audio_shape = [2, 257, 487]
            self.input_size = 256

        elif self.dataset == 'BV1':
            self.max_depth = 12.0
            self.audio_shape = [2, 257, 276]
            self.input_size = 256

        self.model = {'name': 'BaseLine_Audio_My_Fusion',
                      # BaseLineAudio  BaseLine_Audio_My_Fusion  BaseLineAudioCVRFusion  BaseLineAudioCatFusion
                      'audio_shape': self.audio_shape,
                      'visual_encoder_type': 'tiny',
                      'max_depth': self.max_depth,
                      'pretrained': '/home/malong/project/pretrain_model/swin_tiny_patch4_window7_224.pth',
                      'decoder_channels': [1024, 512, 256, 256],
                      'bins_channels': 256,
                      'bins_drop': 0.5,
                      'fusion_num_blocks': 1,
                      'fusion_num_heads': 16,
                      'modal_fusion_ffn_drop': 0.5,
                      'modal_fusion_att_drop': 0.5,
                      'fusion_type': 'CMAFM',  # CMAFM    CMA
                      'input_size': self.input_size,
                      }

        # yy-dd-mm-hh-mm-ss
        self.run_start_time = ''
        self.expr_dir = ''
        