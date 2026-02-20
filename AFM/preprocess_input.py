'''
[Prepare input]
1. Concatenate noisy affordance feature + encoded pc
2. Concatenate feature vector of language and direction
'''

import os
import torch
import torch.nn as nn
import numpy as np 


class PreprocessInput:

    def __init__(self, aff_feat_path:str="", encoded_pc_path:str="", lang_feat_path:str="", dir_feat_path:str=""):
        self.aff_feat = None
        self.encoded_pc = None
        self.lang_feat = None
        self.dir_feat = None
        self.concatenated_aff_pc = None
        self.concatenated_lang_dir = None

    def load_aff_feat(aff_feat_path):
        if not os.path.exists(aff_feat_path):
            raise FileNotFoundError(f"Affordance feature file not found: {aff_feat_path}")
        
        aff_feat = torch.load(aff_feat_path)
        return aff_feat

    def load_encoded_pc(encoded_pc_path):
        if not os.path.exists(encoded_pc_path):
            raise FileNotFoundError(f"Encoded point cloud file not found: {encoded_pc_path}")
        
        encoded_pc = torch.load(encoded_pc_path)
        return encoded_pc

    def load_lang_feat(lang_feat_path):
        if not os.path.exists(lang_feat_path):
            raise FileNotFoundError(f"Language feature file not found: {lang_feat_path}")
        
        lang_feat = torch.load(lang_feat_path)
        return lang_feat

    def load_dir_feat(dir_feat_path):
        if not os.path.exists(dir_feat_path):
            raise FileNotFoundError(f"Direction feature file not found: {dir_feat_path}")
        
        dir_feat = torch.load(dir_feat_path)
        return dir_feat

    def concatenate_aff_pc(aff_feat, encoded_pc):
        concatenated_aff_pc = torch.cat((aff_feat, encoded_pc), dim=-1)
        return concatenated_aff_pc

    def concatenate_lang_dir(lang_feat, dir_feat):
        concatenated_lang_dir = torch.cat((lang_feat, dir_feat), dim=-1)
        return concatenated_lang_dir
    
    def main(self):
        self.concatenateted_aff_pc = self.concatenate_aff_pc(self.aff_feat, self.encoded_pc)
        self.concatenated_lang_dir = self.concatenate_lang_dir(self.lang_feat, self.dir_feat)

        return self.concatenated_aff_pc, self.concatenated_lang_dir


if __name__ == "__main__":
    
    processor=PreprocessInput("path/to/affordance_feature.pt", "path/to/encoded_point_cloud.pt", "path/to/language_feature.pt", "path/to/direction_feature.pt")

    processor.main()


