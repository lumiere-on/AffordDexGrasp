import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius_graph, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np

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
        
        idx = fps(pos, batch, ratio=0.5)
        edge_index = radius_graph(pos, batch, r=0.2)

        x = self.sa1(x, pos[idx], edge_index)
        pos, batch = pos[idx], batch[idx]
        
        # 2. 두 번째 SA 계층
        idx = fps(pos, batch, ratio=0.25)
        edge_index=radius_graph(pos, r=0.4, batch=batch)

        x = self.sa2(x, pos[idx], batch[idx])
        pos, batch = pos[idx], batch[idx]
        
        global_feat = global_max_pool(x, batch) 
        
        return global_feat #'Embedding'을 반환
    
def main():
    pc_path = "/home/jw/jw/UniDexGrasp/dexgrasp_generation/data/DFCdata/meshdatav3/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/pc.npy" 
    points = np.load(pc_path) #(3000, 3)
    
    pos = torch.from_numpy(points).to(torch.float)

    x=pos.clone()

    data=Data(x=x, pos=pos)
    data_batch = Batch.from_data_list([data])

    encoder = PointNet2Encoder(out_channels=512)
    encoder.eval()

    with torch.no_grad():
        embedding = encoder(data_batch)

    print(f"Final Embedding Shape: {embedding.shape}") # [1, 512]
    
    return embedding

if __name__ == "__main__":
    main()