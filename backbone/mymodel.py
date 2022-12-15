import torch
import torch.nn as nn

from backbone.common import (
    CoFusion,
    DoubleConvBlock,
    GhostBottleneck,
    SingleConvBlock,
    UpConvBlock,
    conv3x3,
    weight_init,
)


class LDC(nn.Module):
    def __init__(self):
        super(LDC, self).__init__()

        # Input Block
        self.conv_block = DoubleConvBlock(
            3,
            16,
            16,
            stride=2,
        )

        # Block 2
        self.ghost_module_1 = GhostBottleneck(inp=16, hidden_dim=16, oup=64, kernel_size=3, stride=1, use_se=True)

        # Block 3
        self.ghost_module_2 = GhostBottleneck(inp=64, hidden_dim=64, oup=128, kernel_size=3, stride=1, use_se=True)

        # Block 4
        self.ghost_module_3 = GhostBottleneck(inp=128, hidden_dim=128, oup=256, kernel_size=3, stride=1, use_se=True)


        # Residual Conv
        
        self.residual_conv_1 = SingleConvBlock(16, 64, stride=1)
        self.residual_conv_1_2 = SingleConvBlock(16, 128, stride=1)
        self.residual_conv_1_3 = SingleConvBlock(16, 256, stride=1)
        self.residual_conv_2 = SingleConvBlock(64, 128, stride=1)
        self.residual_conv_2_2 = SingleConvBlock(64, 256, stride=1)
        self.residual_conv_3 = SingleConvBlock(128, 256, stride=1)
    

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(64, 1)
        self.up_block_3 = UpConvBlock(128, 1)
        self.up_block_4 = UpConvBlock(256, 1)

        # fusion
        self.block_cat = CoFusion(4, 4)  # cats fusion method

        # weight initialization
        self.apply(weight_init)

    def forward(self, x):
        assert x.ndim == 4, x.shape
        
        # Block 1
        block1 = self.conv_block(x)  # 16 # [8,16,352,352]
        
        residual_1_1 = self.residual_conv_1(block1) # 64
        residual_1_2 = self.residual_conv_1_2(block1) # 128
        residual_1_3 = self.residual_conv_1_3(block1) # 256

        # Block 2 Residual
        block2 = self.ghost_module_1(block1)  
        block2_residual_conv1 = residual_1_1 + block2
        residual_2 = self.residual_conv_2(block2_residual_conv1) # 128


        # block 3 Residual
        block_3 = self.ghost_module_2(block2_residual_conv1)
        block_3_residual_conv1 = residual_1_2 + block_3 + residual_2
        residual_3 = self.residual_conv_3(block_3_residual_conv1) # 256
        

        # block 4
        block_4 = self.ghost_module_3(block_3_residual_conv1)
        block_4_residual_conv1 = residual_1_3 + block_4 + residual_3

        # block 5
        # block_5 = self.ghost_module_4(block_4)  # 64 # [8,64,352,352]

        # upsampling blocks
        out_1 = self.up_block_1(block1)
        out_2 = self.up_block_2(block2_residual_conv1)
        out_3 = self.up_block_3(block_3_residual_conv1)
        out_4 = self.up_block_4(block_4_residual_conv1)

        results = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results


if __name__ == "__main__":
    from torchview import draw_graph

    model = LDC()
    model_graph = draw_graph(model, input_size=(1, 3, 352, 352), expand_nested=True)
    model_graph.visual_graph.view()
