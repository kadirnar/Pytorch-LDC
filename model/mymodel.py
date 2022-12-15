import torch
import torch.nn as nn
from common import GhostModule, GhostBottleneck, DoubleConvBlock, UpConvBlock, CoFusion, weight_init

    
class LDC(nn.Module):
    def __init__(self):
        super(LDC, self).__init__()
        
        # Input Block
        self.conv_block = DoubleConvBlock(3, 16, 16, stride=2,)
        
        #Block 2
        self.ghost_module_1 = GhostBottleneck(inp=16, hidden_dim=16, oup=64, kernel_size=3,
                                                stride=1, use_se=True)
        
        # Block 3
        self.ghost_module_2 = GhostBottleneck(inp=64, hidden_dim=64, oup=128, kernel_size=3,
                                                stride=1, use_se=True)
        
        # Block 4
        self.ghost_module_3 = GhostBottleneck(inp=128, hidden_dim=128, oup=256, kernel_size=3,
                                                stride=1, use_se=True)
    
        # Block 5
        self.ghost_module_4 = GhostBottleneck(inp=256, hidden_dim=256, oup=512, kernel_size=3,
                                                stride=1, use_se=True)
 
        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(64, 1)
        self.up_block_3 = UpConvBlock(128, 1)
        self.up_block_4 = UpConvBlock(256, 1)
        self.up_block_5 = UpConvBlock(512, 1)

        
        # fusion
        self.block_cat = CoFusion(5, 5)  # cats fusion method
        
        # weight initialization
        self.apply(weight_init)


    def forward(self, x):
        assert x.ndim == 4, x.shape
        # Block 1
        block_1 = self.conv_block(x)  # 16 # [8,16,352,352]

        # Block 2
        block_2 = self.ghost_module_1(block_1)  # 32 # [8,32,352,352]
        
        # block 3
        block_3 = self.ghost_module_2(block_2)  # 64 # [8,64,352,352]
        
        # block 4
        block_4 = self.ghost_module_3(block_3)  # 64 # [8,64,352,352]
        
        # block 5
        block_5 = self.ghost_module_4(block_4)  # 64 # [8,64,352,352]
    
  

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        
        results = [out_1, out_2, out_3, out_4, out_5]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)
        return results
