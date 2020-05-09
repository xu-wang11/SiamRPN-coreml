#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
# @Author: Xu Wang
# @Date: Tuesday, April 28th 2020
# @Email: wangxu.93@hotmail.com
# @Copyright (c) 2020 Institute of Trustworthy Network and System, Tsinghua University
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from onnx_coreml import convert
import coremltools
import onnxruntime

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)
    
    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls
    
    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)
        
        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls
    
    def forward(self, z, x):
        kernel_reg, kernel_cls = self.learn(z)
        return self.inference(x, kernel_reg, kernel_cls)


class SiamRPNInit(SiamRPN):

    def __init__(self, anchor_num=5):
        super().__init__(anchor_num=anchor_num)
    
    def forward(self, z):
        return self.learn(z)
    
    def to_onnx(self, model_path, save_path):
        self.load_state_dict(torch.load(model_path))
        dummy_input = torch.rand(1, 3, 127, 127)
        input_names = ["z"]
        output_names = ["kernel_reg", "kernel_cls"]
        # Convert the PyTorch model to ONNX
        torch.onnx.export(self,
                        dummy_input,
                        save_path,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names)

    def to_coreml(self, oonx_file, coreml_file):
        mlmodel = convert(model=oonx_file, minimum_ios_deployment_target='13')
        # Save converted CoreML model
        mlmodel.save(coreml_file)

class SiamRPNUpdate(SiamRPN):

    def __init__(self, anchor_num=5):
        super().__init__(anchor_num=anchor_num)
    
    def forward(self, x, kernel_reg, kernel_cls):
        return self.inference(x, kernel_reg, kernel_cls)
    
    def to_onnx(self, model_path, save_path):
        self.load_state_dict(torch.load(model_path))
        instance_img = torch.rand(1, 3, 271, 271)
        kernel_reg = torch.randn(20, 512, 4, 4)
        kernel_cls = torch.randn(10, 512, 4, 4)
        dummy_input = (instance_img, kernel_reg, kernel_cls)
        input_names = ["x", 'kernel_reg', 'kernel_cls']
        output_names = ["out_reg", "out_cls"]
        # Convert the PyTorch model to ONNX
        torch.onnx.export(self,
                        dummy_input,
                        save_path,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names)
                        
    def to_coreml(self, oonx_file, coreml_file):
        mlmodel = convert(model=oonx_file, minimum_ios_deployment_target='13')        
        mlmodel.save(coreml_file)

if __name__ == '__main__':
    init_model = SiamRPNInit()
    init_model.to_onnx('model.pth', 'siamrpn_init.onnx')
    init_model.to_coreml('siamrpn_init.onnx', 'siamrpn_init.mlmodel')

    update_model = SiamRPNUpdate()
    update_model.to_onnx('model.pth', 'siamrpn_update.onnx')
    update_model.to_coreml('siamrpn_update.onnx', 'siamrpn_update.mlmodel')
 
    # load input image
    f = cv2.FileStorage('init.yml',flags=0)
    exemplar_image = f.getNode('init_matrix')
    exemplar_image = exemplar_image.mat()
    exemplar_image = exemplar_image.astype(float)
    exemplar_image = np.transpose(exemplar_image, (2, 0, 1))
    exemplar_image = np.expand_dims(exemplar_image, axis=0)
    f.release()

    f = cv2.FileStorage('update.yml',flags=0)
    instance_image = f.getNode('update_matrix')
    instance_image = instance_image.mat()
    instance_image = instance_image.astype(float)
    instance_image = np.transpose(instance_image, (2, 0, 1))
    instance_image = np.expand_dims(instance_image, axis=0)
    f.release()

    # load model
    init_model = coremltools.models.MLModel('siamrpn_init.mlmodel')
    update_model = coremltools.models.MLModel('siamrpn_update.mlmodel')
    x1 = init_model.predict({'z': exemplar_image})
    
    # inference multi input model
    x2 = update_model.predict({'x': instance_image, 'kernel_reg': x1['kernel_reg'], 'kernel_cls': x1['kernel_cls']},useCPUOnly=True)
    ort_session = onnxruntime.InferenceSession("siamrpn_update.onnx")
    ort_inputs = {'x': instance_image.astype(np.float32), 'kernel_reg': x1['kernel_reg'], 'kernel_cls': x1['kernel_cls']}
    ort_outs = ort_session.run(None, ort_inputs)
    print("mlmodel: max value of out_reg:", x2['out_reg'].max())
    print("mlmodel: min value of out_reg:", x2['out_reg'].min())
    print("mlmodel: max value of out_cls:", x2['out_cls'].max())
    print("mlmodel: min value of out_cls:", x2['out_cls'].min()) 
    
    print("onnx: max value of out_reg:", ort_outs[1].max())
    print("onnx: min value of out_reg:", ort_outs[1].min())
    print("onnx: max value of out_cls:", ort_outs[0].max())
    print("onnx: min value of out_cls:", ort_outs[0].min()) 
    print("x2 is different with ort_outs")

