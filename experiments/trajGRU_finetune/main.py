import sys
sys.path.insert(0, '../../')
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.train_and_test import train_and_test
import os
from experiments.net_params import encoder_params, forecaster_params



### Config


batch_size = cfg.GLOBAL.BATCH_SZIE
max_iterations = 100000
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000

LR = 1e-4

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
trained_state_dict = torch.load(cfg.GLOBAL.MODEL_SAVE_DIR + '/../encoder_forecaster_45000.pth')

# Freeze outermost layer
for name, param in encoder_forecaster.named_parameters():
    if 'rnn1' in name:
        param.requires_grad = False
encoder_forecaster.load_state_dict(trained_state_dict)

# Select only grad = true params to update
params_to_update = []
for name, param in encoder_forecaster.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=LR)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.1)
mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.1)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

train_and_test(encoder_forecaster, optimizer, criterion, mult_step_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name)
