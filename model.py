# coding:utf-8
import torch
import torch.nn as nn
import os
from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m, x3d_l


class TSVedioChange(nn.Module):
    def __init__(self, inputChannel=3, numClass=2, modeltype="common"):
        super(TSVedioChange, self).__init__()
        
        configs = {
            "large":  (x3d_l,  "pretrainedModels//Large3D.pyth"),
            "common": (x3d_m,  "pretrainedModels//common3D.pyth"),
            "mini":   (x3d_s,  "pretrainedModels//small3D.pyth"),
            "tiny":   (x3d_xs, "pretrainedModels//tiny3D.pyth")
        }
        
        model_func, local_weight_path = configs.get(modeltype, configs["common"])
        
        self.inputChannel = inputChannel
        self.perception_frame = nn.Parameter(torch.zeros(1, inputChannel, 1, 1))
        nn.init.normal_(self.perception_frame, std=0.02)
        
        print(f"we are building  {modeltype} backbone model...")
        x3d_backbone = model_func(pretrained=False)
        
        if os.path.exists(local_weight_path):
            
            checkpoint = torch.load(local_weight_path, map_location="cpu")
            state_dict = checkpoint.get("model_state", checkpoint)
            x3d_backbone.load_state_dict(state_dict, strict=False)
            print(f"load pretrained mweights successfully: {local_weight_path}")
            
        else:
            print(f"can not load pretrained model successfully: {local_weight_path}")

        if inputChannel != 3:
            self._adapt_first_conv(x3d_backbone.blocks[0], inputChannel)

        self.enc_stem = x3d_backbone.blocks[0]
        self.enc_l1 = x3d_backbone.blocks[1]
        self.enc_l2 = x3d_backbone.blocks[2]
        self.enc_l3 = x3d_backbone.blocks[3]
        self.enc_l4 = x3d_backbone.blocks[4]
        
        enc_channels = [24, 48, 96, 192]
        self.temporal_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in enc_channels
        ])
        
        # decoder head
        self.decoder = DetectionDecoder(
            enc_channels=enc_channels,    
            num_classes=numClass
        )

    def _adapt_first_conv(self, module, in_channels):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv3d):
                old_conv = child
                new_conv = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=(old_conv.bias is not None)
                )
                with torch.no_grad():
                    copy_channels = min(3, in_channels)
                    new_conv.weight[:, :copy_channels, :, :, :] = old_conv.weight[:, :copy_channels, :, :, :]
                setattr(module, name, new_conv)
                return True
            else:
                if self._adapt_first_conv(child, in_channels):
                    return True
        return False

    def _decode_multi_temporal(self, features):
        """
        Decode temporal differences between adjacent frames.
        """
        fused_feats = []
        
        for i, feat in enumerate(features):
            t_minus_1 = feat[:, :, :-1] 
            t_current = feat[:, :, 1:]   
            
            concat_feat = torch.cat([t_minus_1, t_current], dim=1)
            
            B, C2, Tm1, H, W = concat_feat.shape
            
            flat_feat = concat_feat.permute(0, 2, 1, 3, 4).reshape(B * Tm1, C2, H, W)
            
            fused = self.temporal_fusions[i](flat_feat)
            
            fused_feats.append(fused)

        out = self.decoder(fused_feats)

        out = out.view(B, Tm1, out.shape[1], out.shape[2], out.shape[3])
        
        out = list(out.unbind(dim=1))
        
        return out

    def forward(self, x):
        
        if x.dim() == 4:
            B, C, H, W = x.shape
            t1, t2 = x[:, :C//2, :, :], x[:, C//2:, :, :]
            p_frame = self.perception_frame.expand(B, -1, H, W)
            video_volume = torch.cat([t1.unsqueeze(2), p_frame.unsqueeze(2), t2.unsqueeze(2)], dim=2)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            if C != self.inputChannel:
                raise ValueError(
                    f"5D input channel count must equal model inputChannel={self.inputChannel}"
                )
            video_volume = x.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError("Unsupported input shape.")

        x_enc = self.enc_stem(video_volume)
        feat_l1 = self.enc_l1(x_enc)
        feat_l2 = self.enc_l2(feat_l1)
        feat_l3 = self.enc_l3(feat_l2)
        feat_l4 = self.enc_l4(feat_l3)

        if x.dim() == 4:
            p_l1 = torch.mean(feat_l1, dim=2)
            p_l2 = torch.mean(feat_l2, dim=2)
            p_l3 = torch.mean(feat_l3, dim=2)
            p_l4 = torch.mean(feat_l4, dim=2)
            
            features = [p_l1, p_l2, p_l3, p_l4]
            logits = self.decoder(features)
            return logits

        return self._decode_multi_temporal([feat_l1, feat_l2, feat_l3, feat_l4])


class DetectionDecoder(nn.Module):
    """
    Multi-scale feature fusion
    """
    def __init__(self, enc_channels, num_classes):
        super().__init__()
        c1, c2, c3, c4 = enc_channels

        self.up4_3 = self._make_deconv_block(c4, c3)
        self.up3_2 = self._make_deconv_block(c3, c2)
        self.up2_1 = self._make_deconv_block(c2, c1)
        

        self.final_up = nn.Sequential(
            self._make_deconv_block(c1, 32), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def _make_deconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        l1, l2, l3, l4 = features

        x = self.up4_3(l4) + l3
        
        x = self.up3_2(x) + l2
        
        x = self.up2_1(x) + l1
        
        x = self.final_up(x)
        
        out = self.classifier(x)
        
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSVedioChange(inputChannel=3, numClass=2, modeltype="large").to(device)
    
    x = torch.randn(1, 4, 3, 256, 256).to(device)  # [B, T, C, H, W], C=3
    
    out = model(x)   # output [B, T-1, numClass, H_out, W_out]
    print(f"inference success！putput shape: {out[0].shape}")