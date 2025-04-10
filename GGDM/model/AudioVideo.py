import torch
import torch.nn as nn
import torch.nn.functional as F
from .Resnet import resnet18, resnet34, resnet50


class AudioEncoder(nn.Module):
    def __init__(self, config=None, mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model = mask_model
        if config['text']["name"] == 'resnet18':
            self.audio_net = resnet18(modality='audio')
        self.config = config
    def forward(self, audio, step=0, balance=0, s=400, a_bias=0):

        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a, 1)  # [512,1]
        a = torch.flatten(a, 1)  # [512]
        return a

class VideoEncoder(nn.Module):
    def __init__(self, config=None, fps=1, mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        if config['visual']["name"] == 'resnet18':
            self.video_net = resnet18(modality='visual')
        self.fps = fps
        self.config  = config
    def forward(self, video, step=0, balance=0, s=400, v_bias=0):
        # print(video.size())
        # if self.config["dataset"]['dataset_name'] == 'AVE':
        #     print('true')
        #     video = video.permute(0, 2, 1, 3, 4).contiguous()
        v = self.video_net(video)

        (_, C, H, W) = v.size()
        B = int(video.size()[0] / self.fps)
        v = v.view(B, -1, C, H, W)


        v = v.permute(0, 2, 1, 3, 4)
        
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)
        return v


class AVClassifier(nn.Module):
    def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.audio_encoder = AudioEncoder(config, mask_model)
        self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
        self.hidden_dim = 512


        self.cls_a = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        self.cls_v = nn.Linear(self.hidden_dim, config['setting']['num_class'])
        self.cls_av = nn.Linear(self.hidden_dim*2,config['setting']['num_class'])

        # self.cls_b = nn.Linear(self.hidden_dim * 2 , config['setting']['num_class'])

    def forward(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)

        result_a = self.cls_a(a_feature)
        result_v = self.cls_v(v_feature)
        a_conf = torch.log(torch.sum(torch.exp(result_a), dim=1))
        v_conf = torch.log(torch.sum(torch.exp(result_v), dim=1))
        a_conf = a_conf / 10
        v_conf = v_conf / 10
        a_conf = torch.reshape(a_conf, (-1, 1))
        v_conf = torch.reshape(v_conf, (-1, 1))
        # result_b = self.cls_b(torch.cat((a_feature, v_feature), dim=1))
        result_b = self.cls_av(torch.cat([result_a,result_v],dim=-1))
        # result_b = result_a+result_v
        return result_b, result_a, result_v, (a_feature,a_conf), (v_feature,v_conf)

    def getFeature(self, audio, video):
        a_feature = self.audio_encoder(audio)
        v_feature = self.video_encoder(video)
        return a_feature, v_feature


# class AVClassifier(nn.Module):
#     def __init__(self, config, mask_model=1, act_fun=nn.GELU()):
#         super(AVClassifier, self).__init__()
#         self.audio_encoder = AudioEncoder(config, mask_model)
#         self.video_encoder = VideoEncoder(config, config['fps'], mask_model)
#         self.hidden_dim = 512
#
#         self.cls_a = nn.Linear(self.hidden_dim, config['setting']['num_class'])
#         self.cls_v = nn.Linear(self.hidden_dim, config['setting']['num_class'])
#
#         self.audio_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.audio_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.video_mu = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.video_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
#
#         self.av_mu = nn.Linear(self.hidden_dim*2,self.hidden_dim)
#         self.cls_b = nn.Linear(self.hidden_dim,config['setting']['num_class'])
#         # self.cls_b = nn.Linear(self.hidden_dim * 2 , config['setting']['num_class'])
#
#     def forward(self, audio, video):
#         a_feature = self.audio_encoder(audio)
#         v_feature = self.video_encoder(video)
#
#         a_mu = self.audio_mu(a_feature)
#         v_mu = self.video_mu(v_feature)
#
#         a_logvar = self.audio_logvar(a_feature)
#         v_logvar = self.video_logvar(v_feature)
#
#         av_mu = self.av_mu(torch.cat([a_mu,v_mu],dim=-1))
#
#         result_a = self.cls_a(a_mu)
#         result_v = self.cls_v(v_mu)
#         result_b = self.cls_b(av_mu)
#         # result_b = self.cls_b(torch.cat((a_feature, v_feature), dim=1))
#
#         # result_b = result_v + result_a
#
#         return result_b, result_a, result_v, (a_mu,av_mu,a_logvar), (v_mu,av_mu,v_logvar)
#
#     def getFeature(self, audio, video):
#         a_feature = self.audio_encoder(audio)
#         v_feature = self.video_encoder(video)
#         return a_feature, v_feature