# Benchmark on Drug Target Interaction Modeling from a Drug Structure Perspective

This is the official codebase of the paper *Benchmark on Drug Target Interaction Modeling from a Drug Structure Perspective*.

GTB-DTI is a comprehensive benchmark customized for GNN and
Transformer-based methodologies for DTI prediction.

## Reproduction
Before you begin, you can install the required libraries using:

First, clone the repository to your local machine:
```bash
git clone <repository-url>
cd GTB-DTI
```

### Traning the Model
```python
python main.py -c config/model.yaml
```

The config.yaml file contains all the configurable parameters for training the model. You can edit this file to adjust parameters such as learning rate, batch size, and number of epochs.

### Memory Evaluation

Set the 'train' to 'memory_test' in the config.yaml file
```bash
task:
  class: regression
  model:
    class: model
    param: 
  train: memory_test
```

## License
This codebase is released under the MIT License as in the [LICENSE](LICENSE) file.
