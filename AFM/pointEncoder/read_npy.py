import numpy as np
import trimesh

# 1. .npy 파일 확인
data_npy = np.load("/home/jw/jw/UniDexGrasp/dexgrasp_generation/data/DFCdata/meshdatav3/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/pc.npy")
print("NPY shape:", data_npy.shape) 
# 결과가 (4096, 3) 처럼 나오면 이게 포인트 클라우드입니다!

# 2. .obj 파일 확인
mesh = trimesh.load("/home/jw/jw/UniDexGrasp/dexgrasp_generation/data/DFCdata/meshdatav3/core/bottle-1a7ba1f4c892e2da30711cdbdbc73924/coacd/coacd_convex_piece_0.obj")
print("OBJ vertices:", mesh.vertices.shape)
print("OBJ faces:", mesh.faces.shape) # 면(faces) 정보가 있다면 Mesh입니다.