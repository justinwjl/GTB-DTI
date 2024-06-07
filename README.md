# Benchmark on Drug Target Interaction Modeling from a Structure Perspective

Before you begin, you can install the required libraries using:

First, clone the repository to your local machine:
```bash
git clone https://github.com/justinwjl/GTB-DTI.git
cd your-repo
```

## Traning the Model
```python
python main.py -c config/model.yaml
```

The config.yaml file contains all the configurable parameters for training the model. You can edit this file to adjust parameters such as learning rate, batch size, and number of epochs.

## Memory Evaluation

Set the 'train' to 'memory_test' in the config.yaml file