## Installation 

* The code is tested with pytorch 1.3 and python 3.6

## Download/Prepare Data

If using Cityscapes, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_DIR=<path_to_cityscapes>
```

* Download Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)

If using Cityscapes Autolabelled Images, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_CUSTOMCOARSE=<path_to_cityscapes>
```

If using Mapillary, download Mapillary data, then update `config.py` to set the path:
```python
__C.DATASET.MAPILLARY_DIR=<path_to_mapillary>
```


## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.

### Run inference on Cityscapes

Dry run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i -n
```
This will just print out the command but not run. It's a good way to inspect the commandline. 

Real run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i
```

The reported IOU should be 86.92. This evaluates with scales of 0.5, 1.0. and 2.0. You will find evaluation results in ./logs/eval_cityscapes/...

### Run inference on Mapillary

```bash
> python -m runx.runx scripts/eval_mapillary.yml -i
```

The reported IOU should be 61.05. Note that this must be run on a 32GB node and the use of 'O3' mode for amp is critical in order to avoid GPU out of memory. Results in logs/eval_mapillary/...


## Train a model

Train cityscapes, using HRNet + OCR + multi-scale attention with fine data and mapillary-pretrained model
```bash
> python -m runx.runx scripts/train_cityscapes.yml -i
```

The first time this command is run, a centroid file has to be built for the dataset. It'll take about 10 minutes. The centroid file is used during training to know how to sample from the dataset in a class-uniform way.

## Train SOTA default train-val split
```bash
> python -m runx.runx  scripts/train_cityscapes_sota.yml -i
```
Again, use `-n` to do a dry run and just print out the command. If you run out of memory, try to lower the crop size or turn off rmi_loss.
