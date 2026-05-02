import torch
def create(y, window_size, stride):
    y_w = y.unfold(0, window_size, stride) #(num_windows, 2, window_size)
    y_w = y_w.permute(0, 2, 1) #(num_windows, window_size, 2)
    return y_w