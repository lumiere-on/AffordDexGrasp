#load initial point cloud for point encoder
import os
import torch
import trimesh

def load_pc(pc_path):
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
    
    pc=torch.load(pc_path)
    pc=trimesh.Trimesh(vertices=pc, process=False)

    return pc

if __name__=="__main__":
    pc_path = "path/to/point_cloud.pt"
    point_cloud = load_pc(pc_path)

    output_path=""
    with open(output_path, 'w') as f:
        f.write(str(point_cloud))
    print(point_cloud)