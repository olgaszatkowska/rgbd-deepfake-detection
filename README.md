# rgbd-deepfake-detection

## Setup

```
# create virtual enviroment
virtualenv -p /usr/bin/python3.10  rgbd-df-detect-venv

# activate it
source rgbd-df-detect-venv/bin/activate

# install
python -m pip install -r requirements.txt
```

## Running
```
# to view config
python train.py hydra.verbose=true

# with debug level (default: INFO)
python train.py hydra.job_logging.root.level=DEBUG

# disable saving to directory
python train.py hydra.job_logging.root.level=DEBUG hydra.run.dir=. hydra.output_subdir=null
```

## Logs
```
# run tensorboard server
tensorboard --logdir=lightning_logs_tf
```

### Third-Party Code
The following files were copied from the [rgbd-depthfake](https://github.com/gleporoni/rgbd-depthfake) project:
- `data/faceforensics.py`
- `data/data_loader.py`

These files are included to ensure compatibility with the original dataset preprocessing and data loading procedures. I intend to conduct further research on models with a similar RGBD input structure, and retaining these base components allows for consistent benchmarking and experimentation.