## Deep Learning approach towards Precipitation Nowcasting
Here you can find the source code used for the paper "Deep Learning Approach Towards Precipitation Nowcasting:
Evaluating Regional Extrapolation Capabilities". You can find the readme of the original pytorch implementation repository by [Zhizhong Huang](https://hzzone.github.io/) below.
All code was run using CUDA 11.3 with PyTorch 1.10.0 on Python 3.6.8.

### Configuration
You can find the configuration at `/nowcasting/config.py`. To be able to run the following scripts you may need to adjust
the used data paths.

### Preprocessing
The input radar png images are available at https://doi.org/10.5281/zenodo.6634851 and the corresponding pkl information files used by the models in `/hko_data`.

You can find the preprocessing scripts used to generate those files from the raw RADOLAN RX composite in the `/preprocessing` folder.

`generate_rx_plots.py` is used to generate the radar png images.  
`generate_rx_pkls.py` is used to generate the pkl information files. The radar png images need to exist to be able to run this.


### Training
The code that was used for training the model from scratch and finetuning the pretrained model can be found in
`/experiments/trajGRU_from_scratch/main.py` and `/experiments/trajGRU_finetune/main.py`

### Benchmark
The script to run the benchmarks can be found under `experiments/benchmark.py`.

To run the benchmarks for the finished models their weights need to be placed in the model save directory.
The ones used in the paper are available under https://doi.org/10.5281/zenodo.6634844

---
## TrajGRU PyTorch Implementation

### Introduction
This repo has implemented a pytorch-based encoder-forecaster model with RNNs including (TrajGRU, ConvLSTM) to do precipitation nowcasting. For more information about TrajGRU, please refer to [HKO-7](https://github.com/sxjscience/HKO-7).

If you are interested in my implementation of ConvLSTM and TrajGRU, please see [ConvLSTM](https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py) and [TrajGRU](https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/trajGRU.py). It is assumed that the input shape should be <img src="https://latex.codecogs.com/gif.latex?S*B*C*H*W" title="S*B*C*H*W" />. All of my implementation have been proved to be effective in HKO-7 Dataset. Hopefully it helps your research.

### Train
Firstly you should apply for HKO-7 Dataset from [HKO-7](https://github.com/sxjscience/HKO-7), and modify somelines in config.py to find the dataset path.
Secondly and last, run `python3 experiments/trajGRU_balanced_mse_mae/main.py`, and then run `python3 experiments/trajGRU_frame_weighted_mse/main.py` since I have finetuned the model on the basis of model trained in last step.

### Environment
Python 3.6+, PyTorch 1.0 and Ubuntu or macOS.

### Demo
![](demo.gif)

### Performance
The performance on HKO-7 dataset is below.

<table>
	<tbody>
		<tr>
			<td colspan="5" align="center">CSI</td>
			<td colspan="5" align="center">HSS</td>
			<td align="center">Balanced MSE</td>
			<td align="center">Balanced MAE</td>
		</tr>
		<tr>
			<td  align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;0.5" title="r \geq 0.5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;2" title="r \geq 2" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;5" title="r \geq 5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;10" title="r \geq 10" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;30" title="r \geq 30" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;0.5" title="r \geq 0.5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;2" title="r \geq 2" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;5" title="r \geq 5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;10" title="r \geq 10" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;30" title="r \geq 30" /></td>
			<td align="center"></td>
			<td align="center"></td>
		</tr>
		<tr>
			<td align="center">0.5496</td>
			<td align="center">0.4772</td>
			<td align="center">0.3774</td>
			<td align="center">0.2863</td>
			<td align="center">0.1794</td>
			<td align="center">0.6713</td>
			<td align="center">0.6150</td>
			<td align="center">0.5226</td>
			<td align="center">0.4253</td>
			<td align="center">0.2919</td>
			<td align="center">5860.97</td>
			<td align="center">15062.46</td>
		</tr>
	</tbody>
</table>

### Download

[Dropbox Pretrained Model](https://www.dropbox.com/sh/i5goltdq83dmkvc/AABBe5wTuEQF5j3VSMszVQSaa?dl=0)

### Citation

```
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
@inproceedings{xingjian2015convolutional,
  title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
  author={Xingjian, SHI and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
  booktitle={Advances in neural information processing systems},
  pages={802--810},
  year={2015}
}
```

