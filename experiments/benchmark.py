import sys
sys.path.insert(0, '..')
import copy
from experiments.net_params import *
import time
import pickle
import os

encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
encoder_forecaster1 = EF(encoder, forecaster)
encoder_forecaster2 = copy.deepcopy(encoder_forecaster1)

encoder_forecaster1 = encoder_forecaster1.to(cfg.GLOBAL.DEVICE)
encoder_forecaster2 = encoder_forecaster2.to(cfg.GLOBAL.DEVICE)
encoder_forecaster1.load_state_dict(torch.load(cfg.GLOBAL.MODEL_SAVE_DIR +  '/trajGRU_from_scratch/models/encoder_forecaster_88000.pth'))
encoder_forecaster2.load_state_dict(torch.load(cfg.GLOBAL.MODEL_SAVE_DIR + '/trajGRU_finetune/models/encoder_forecaster_89000.pth'))

models = OrderedDict({
    'trajGRU_from_scratch': encoder_forecaster1,
    'trajGRU_finetune': encoder_forecaster2,
})

model_run_avarage_time = dict()
with torch.no_grad():
    for name, model in models.items():
        print(name)
        is_deeplearning_model = (torch.nn.Module in model.__class__.__bases__)
        if is_deeplearning_model:
            model.eval()
        evaluator = HKOEvaluation(seq_len=OUT_LEN, use_central=False)
        hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_VALID,
                                     sample_mode="sequent",
                                     seq_len=IN_LEN + OUT_LEN,
                                     stride=cfg.HKO.BENCHMARK.STRIDE)
        model_run_avarage_time[name] = 0.0
        valid_time = 0
        while not hko_iter.use_up:
            valid_batch, valid_mask, sample_datetimes, _ = \
                hko_iter.sample(batch_size=1)
            if valid_batch.shape[1] == 0:
                break
            if not cfg.HKO.EVALUATION.VALID_DATA_USE_UP and valid_time > cfg.HKO.EVALUATION.VALID_TIME:
                break

            valid_batch = valid_batch.astype(np.float32) / 255.0
            valid_data = valid_batch[:IN_LEN, ...]
            valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
            mask = valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)
            mask.fill(1)

            if is_deeplearning_model:
                valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

            start = time.time()
            output = model(valid_data)
            model_run_avarage_time[name] += time.time() - start

            if is_deeplearning_model:
                output = output.cpu().numpy()

            output = np.clip(output, 0.0, 1.0)

            evaluator.update(valid_label, output, mask)

            valid_time += 1
        model_run_avarage_time[name] /= valid_time
        evaluator.save_pkl(osp.join('./benchmark_stat', name + '.pkl'))

with open('./benchmark_stat/model_run_avarage_time.pkl', 'wb') as f:
    pickle.dump(model_run_avarage_time, f)

for p in os.listdir('benchmark_stat'):
    e = pickle.load(open(osp.join('benchmark_stat', p), 'rb'))
    _, _, csi, hss, _, mse, mae, balanced_mse, balanced_mae, _ = e.calculate_stat()
    print(p.split('.')[0])
    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        print('thresh %.1f csi: avarage %.4f, last frame %.4f; hss: avarage %.4f, last frame %.4f;'
              % (thresh, csi[:, i].mean(), csi[-1, i], hss[:, i].mean(), hss[-1, i]))

    print(('mse: avarage %.2f, last frame %.2f\n' +
        'mae: avarage %.2f, last frame %.2f\n'+
        'bmse: avarage %.2f, last frame %.2f\n' +
        'bmae: avarage %.2f, last frame %.2f\n') % (mse.mean(), mse[-1], mae.mean(), mae[-1],
              balanced_mse.mean(), balanced_mse[-1], balanced_mae.mean(), balanced_mae[-1]))
