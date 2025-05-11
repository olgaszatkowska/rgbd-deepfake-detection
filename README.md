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

### Third-Party Code
The following files were copied from the [rgbd-depthfake](https://github.com/gleporoni/rgbd-depthfake) project:
- `data/faceforensics.py`
- `data/data_loader.py`

With the purpose of 