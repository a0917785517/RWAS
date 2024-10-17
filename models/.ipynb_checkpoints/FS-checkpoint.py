#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:13:21 2023

@author: yanyan
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import random

from dataclasses import dataclass
# from scipy.ndimage import uniform_filter1d
# from scipy.signal import medfilt

from models.Net import Net

class OPT:  #fit Evaluate.py
    nothingelse: str = 'nothingelse'
@dataclass
class FS:
    weights: str
    Backbone: str
    bool_pretrained: bool
    device: str = None

    def __post_init__(self):

        if self.device and self.device in ['cuda', 'cpu']:
            self.device = torch.device(self.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.opt = OPT()
        #self.opt = argparse.ArgumentParser().parse_args()
        for k, v in vars(self).items():
            self.opt.__setattr__(k, v)

        self.model = Net(
                        backbone=self.Backbone,
                        bool_pretrained=self.bool_pretrained
                        ).to(self.device)
        
        checkpoint = torch.load(self.weights, map_location=self.device)
        self.model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.model.regression_head.load_state_dict(checkpoint['RGHead_state_dict'])

        self.model.eval()

    def predict(self, image, out_w=None, out_h=None, inference_size=[512,288], panels=[["full"]], filtration=False):
        
        output_list = []
        panel = panels.copy()

        if out_w is None:
            out_w = image.shape[1]
        if out_h is None:
            out_h = image.shape[0]
        
        # Let panel list to normalization
        if panels[0][0] == "full":
            # print(out_w)
            panel[0] = [0,0,out_w,out_h]
        
        img_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        trf = self.trfor(inference_size[0], inference_size[1])

        # crop the panel's area
        cuts = []
        for pl in panel:
            cuts.append(img_[pl[1]:pl[3], pl[0]:pl[2]])

        # Preprocess and convert to tensor.
        img_tensor = torch.stack([trf(cut) for cut in cuts])    
        img_tensor = img_tensor.to(self.device)
        
        # model inference
        output = self.model(img_tensor)[:,:,0,0]

        # Calculate the results of each panel's corresponding coordinates on the original image based on the output batch 
        for sub_panel_index in range(output.shape[0]):
            sub_output = torch.sigmoid(output)[sub_panel_index]

            # sigmoid value multiplied by the total height of the panel plus the starting height of the panel
            sub_output = ((sub_output*(panel[sub_panel_index][3]-panel[sub_panel_index][1])) + panel[sub_panel_index][1]).long()

            sub_output = sub_output.cpu().numpy()

            ## random add +/- (Usually comments)
            # sub_output = self.randomelement(sub_output)
            ## slide window  (Usually comments)
            # sub_output = uniform_filter1d(sub_output, size=7)
            ## median filter  (Usually comments)
            # sub_output = medfilt(sub_output, kernel_size=7)
            
            # Linear interpolation maps the original array to a new array.
            new_array = self.linear_interpolation((panel[sub_panel_index][2]-panel[sub_panel_index][0]), sub_output)

            # Convert array to shape usable by cv2.polylines
            if filtration:
                new_array = self.filtration(new_array, panel[sub_panel_index])

            output_list.append(new_array)

        return output_list

    def trfor(self, in_w, in_h):
        # pre-processing transform image data into Tensor.
        trf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((in_h, in_w))
        ])
        return trf

    def linear_interpolation(self, width, output):
        new_array = np.zeros((width,), dtype=output.dtype)
        for i in range(width):
            # Calculate the indices in the original array and perform linear interpolation.
            original_index = i * len(output) // width
            fraction = i * len(output) / width - original_index
            if original_index < len(output) - 1:
                new_array[i] = (1 - fraction) * output[original_index] + fraction * output[original_index + 1]
            else:
                new_array[i] = output[original_index]
        return new_array

    def filtration(self, input_array, panel):
        FS_posi = input_array.reshape(-1, 1) # Change the shape of the array from (1280,) to (1280, 1)
        length = len(FS_posi)
        indices = np.arange(panel[0], panel[2]).reshape(-1, 1) # create an index array
        new_array = np.hstack((indices, FS_posi))
        return new_array

    def randomelement(self, input_array):
        random_numbers = np.random.randint(4, 10, size=input_array.shape)

        signs = np.random.choice([-1, 1], size=input_array.shape)

        # 將隨機數和隨機符號結合起來
        random_adjustments = random_numbers * signs

        input_array_plus_random = input_array + random_adjustments
        
        return input_array_plus_random

        