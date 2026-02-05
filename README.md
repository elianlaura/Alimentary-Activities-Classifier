# Deo-Classifier: Human Activity Recognition with Attention-BiGRU

## Table of Contents
- [Abstract](#abstract)
- [Files in This Repository](#files-in-this-repository)
- [Installation](#installation-of-the-conda-environment)
- [Dependencies](#dependencies)
- [Running the Scripts](#running-the-python-script)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Support](#support)

## Abstract

The Deo-Classifier is a deep learning project for Human Activity Recognition (HAR) using sensor data from accelerometers, gyroscopes, and magnetometers. It employs an Attention-enhanced Bidirectional Gated Recurrent Unit (BiGRU) model to classify activities such as drinking, eating, and other movements. The project includes training and testing scripts that handle data preprocessing, model training with focal loss for imbalanced classes, and evaluation with metrics like accuracy, F1-score, and Cohen's Kappa. Key features include self-attention mechanisms for better sequence understanding and custom callbacks for training monitoring. This framework is suitable for HAR, gesture recognition, and similar sequential classification tasks.

## Files in This Repository
- `cabigru_deo_train.py`: This script trains the Attention-BiGRU model on a specified dataset.
- `cabigru_deo_test.py`: This script evaluates the trained model on a test dataset.
- `model_builder.py`: Contains functions for building the neural network models.
- `plotting_utils.py`: Utilities for plotting training curves and results.
- `training_utils.py`: Helper functions for training processes.
- `utils_cab.py`: Additional utility functions.
- `requirements.txt`: List of Python dependencies with versions.
- `LICENSE`: Custom license with usage restrictions.
- `README.md`: This file.

### Installation of the Conda Environment

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).  
Open command prompt and run:  
```bash
conda update --all
conda create --name <env_name> python=3.9 --file requirements.txt
conda activate <env_name>
```

In the new conda environment, install TensorFlow (2.11 - 2.14) with GPU support according to the [official website](https://www.tensorflow.org/install/pip).

### Dependencies

- pandas==2.3.3
- numpy==1.26.4
- matplotlib==3.7.1
- tensorflow==2.15.0
- scikit-learn==1.3.0
- torch==1.13.1

### Running the Python Script

To run the scripts, open an Anaconda prompt, navigate to the cloned repository directory, and go into the `src` folder. Run the training script with optimal parameters:  

```bash
python src/cabigru_deo_train.py
```

For testing:  
```bash
python src/cabigru_deo_test.py
```

## Model Architecture

The model uses a Bidirectional Gated Recurrent Unit (BiGRU) enhanced with self-attention mechanisms for sequence classification. It incorporates categorical focal loss to handle class imbalances and includes custom callbacks for monitoring training progress, such as early stopping and learning rate scheduling.

## Results

*Performance metrics from experiments (to be updated with actual results):*  
- Balanced Accuracy: 86.55
- Sensitivity: [80.53 86.60 92.51]
- Specificity: [96.62 95.62 89.22]

Loss curves and evaluation reports are generated during testing.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue first to discuss.

## License

This project is licensed under a custom license. See the [LICENSE](LICENSE) file for details. Note: Commercial use and replication in other projects require prior written permission.

## Troubleshooting

- **GPU Issues**: Ensure TensorFlow is installed with GPU support and CUDA is properly configured.
- **Dependency Conflicts**: Use the specified versions in `requirements.txt` to avoid issues.
- **Data Format**: Ensure your dataset matches the expected format (time-series sensor data).

## Citation

If you use this project in your research, please cite:  

```
@misc{deo-classifier,
  title={Deo-Classifier: Human Activity Recognition with Attention-BiGRU},
  author={Your Name},
  year={2026},
  url={https://github.com/elianlaura/Deo-Classifier}
}
```

## Support

For further assistance, open an issue on [GitHub](https://github.com/elianlaura/Deo-Classifier/issues) or contact [elianlaura](https://github.com/elianlaura).