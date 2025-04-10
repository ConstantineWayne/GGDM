import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .AudioVideo import AudioEncoder,VideoEncoder


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
           m.bias.data.fill_(0.0)

def mlp(dim,hidden_dim,output_dim,layers,activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]
    seq = [nn.Linear(dim,hidden_dim),activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim,hidden_dim),activation()]
    seq += [nn.Linear(hidden_dim,output_dim)]
    return nn.Sequential(*seq)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
class Discrim(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_labels,config=None):
        super().__init__()
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)

        self.mlp_audio = nn.Linear(input_dim,num_labels)
        self.mlp_video = nn.Linear(input_dim,num_labels)
        self.mlp_av = nn.Linear(input_dim*2,num_labels)
        if config['dataset']['dataset_name'] == 'AVE':
            self.mlp_av = nn.Linear(input_dim*2,num_labels)
        self.flag = True
        self.apply(weight_init)


    def forward(self,input):
        a_feature = self.audio_encoder(input[0])
        v_feature = self.video_encoder(input[1])

        out_a = self.mlp_audio(a_feature)
        out_v = self.mlp_video(v_feature)

        feature = torch.cat([a_feature,v_feature],dim=-1)
        if self.flag:
            out_av = self.mlp_av(feature)
        else:
            out_av = 0.5 * out_a + 0.5 * out_v
        return a_feature,v_feature,feature,out_a,out_v,out_av



