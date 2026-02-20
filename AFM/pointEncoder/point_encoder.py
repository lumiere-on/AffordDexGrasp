import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius_graph, global_max_pool
from load_pc import load_pc

class PointNet2Encoder(nn.Module):
    def __init__(self, out_channels=512): 
        super(PointNet2Encoder, self).__init__()


        self.sa1 = PointNetConv(nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        ))
        
        self.sa2 = PointNetConv(nn.Sequential(
            nn.Linear(128 + 3, 256), # 이전 채널(128) + 좌표(3)
            nn.ReLU(),
            nn.Linear(256, out_channels)
        ))

    def forward(self, data):
        # x: 특징, pos: 좌표, batch: 배치 인덱스
        x, pos, batch = data.x, data.pos, data.batch
        
        x = self.sa1(x, pos, batch)
        
        # 2. 두 번째 SA 계층
        x = self.sa2(x, pos, batch)
        
        global_feat = global_max_pool(x, batch) 
        
        return global_feat #'Embedding'을 반환
    
def main():
    pc=load_pc("path_to_encoded_pc.pt")  
    encoder=PointNet2Encoder(out_channels=512)
    
    encoder.forward(pc)
    encoder 