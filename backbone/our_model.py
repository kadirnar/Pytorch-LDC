import torch
import torch.nn as nn
from common import (
    CoFusion,
    DoubleConvBlock,
    GhostBottleneck,
    SingleConvBlock,
    UpConvBlock,
    weight_init,
)

class LDC(nn.Module):
    def __init__(self):
        super(LDC, self).__init__()

        # Input Block
        self.input_double_conv = DoubleConvBlock(3,32,32,stride=2)
        self.input_ghost_module_1 = GhostBottleneck(inp=32, hidden_dim=32, oup=32, kernel_size=3, stride=1, use_se=True)
        self.input_ghost_module_2 = GhostBottleneck(inp=32, hidden_dim=32, oup=96, kernel_size=3, stride=1, use_se=True)
        self.input_ghost_module_3 = GhostBottleneck(inp=96, hidden_dim=96, oup=96, kernel_size=3, stride=1, use_se=True)
        
        # Block-1
        self.ghost_module_1 = GhostBottleneck(inp=96, hidden_dim=96, oup=96, kernel_size=3, stride=1, use_se=True)

        # Block-2
        self.ghost_module_2 = GhostBottleneck(inp=128, hidden_dim=128, oup=128, kernel_size=3, stride=1, use_se=True)

        # Block-3
        self.ghost_module_3 = GhostBottleneck(inp=256, hidden_dim=256, oup=256, kernel_size=3, stride=1, use_se=True)


        # Residual block
        self.residual_single_conv_0 = SingleConvBlock(in_features=32, out_features=96, stride=1)
        self.residual_single_conv_1 = SingleConvBlock(in_features=96, out_features=128, stride=1)
        self.residual_single_conv_2 = SingleConvBlock(in_features=96, out_features=256, stride=1)
        self.residual_single_conv_3 = SingleConvBlock(in_features=128, out_features=256, stride=1)
        
        # USNet
        self.up_block_1 = UpConvBlock(96, 1)
        self.up_block_2 = UpConvBlock(256, 1)
        self.up_block_3 = UpConvBlock(256, 1)
        self.up_block_4 = UpConvBlock(256, 1)

        # fusion
        self.block_cat = CoFusion(4,4)  # cats fusion method

        # weight initialization
        self.apply(weight_init)

    def forward(self, x):
                
        # Block-1
        input_layer = self.input_double_conv(x)
        input_ghost_layer_2 = self.input_ghost_module_1(input_layer) # 16
        input_ghost_layer_3 = self.input_ghost_module_2(input_ghost_layer_2) # 32
        input_ghost_layer_4 = self.input_ghost_module_3(input_ghost_layer_3) # 32


        # Block 2 
        # residual block
        stage_1_ghost = self.ghost_module_1(input_ghost_layer_4) # 32
        stage_1_residual = self.residual_single_conv_0(input_ghost_layer_2) # 32
        stage_1_residual_32_64 = self.residual_single_conv_2(stage_1_residual) # 64
        
        stage_1 = self.ghost_module_1(stage_1_residual) # 32
        stage_1_residualv2_32_64 = self.residual_single_conv_2(stage_1) # 64
        stage_1_residual_2 = self.residual_single_conv_2(stage_1) # 64
        
        # Block 3
        # residual block
        stage_2_residual = self.residual_single_conv_1(input_ghost_layer_3) # 48
        stage_2_residual_48_64 = self.residual_single_conv_3(stage_2_residual) # 64
        
        stage_2 = self.ghost_module_2(stage_2_residual) # 48
        stage_2_residual_48_64v2 = self.residual_single_conv_3(stage_2) # 64
        
        stage_2_residual_2 = self.residual_single_conv_3(stage_2) # 64
        
        # Block 4
        # residual block
        stage_3_residual = self.residual_single_conv_2(input_ghost_layer_4) # 64
        
        stage_3 = self.ghost_module_3(stage_3_residual) # 64

        # stage cat
        stage_2_cat = stage_1_residual_2 + stage_2_residual_2 + stage_3 # 64
        stage_3_cat = stage_1_residual_32_64 + stage_2_residual_48_64 + stage_3_residual
        stage_4_cat = stage_1_residualv2_32_64 + stage_2_residual_48_64v2 + stage_3
 
        # upsampling blocks
        out_1 = self.up_block_1(input_ghost_layer_4) # bu doğru.
        out_2 = self.up_block_2(stage_2_cat)
        out_3 = self.up_block_3(stage_3_cat)
        out_4 = self.up_block_3(stage_4_cat)


        results = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results
