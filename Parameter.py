# coding:utf-8
import torch
from thop import profile
import time
try:
    from AImodel.model import TSVedioChange as AInet
except:
    from model import TSVedioChange as AInet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    inputChannel, numClass = 3, 2
    
    batch_size = 4
    time_steps = 16
    channels = 3
    height, width = 512, 512

    input = torch.randn(batch_size, time_steps, channels, height, width).to(device)
    
    model = AInet(inputChannel=inputChannel, numClass=numClass, modeltype="common").to(device)  # tiny, mini, common, large
    
    flops, params = profile(model, (input,))  # Parameter
    print('diff_flops: G', flops/1e9, 'diff_params: M', params/1e6)
    
    time_s = time.time()                      # inference time
    result = model(input)
    
    time_e = time.time()
    time_all = time_e - time_s
    print("cost time:", time_all)
    
