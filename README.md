
# Denoising Autoencoder with Temporal Shift Module for Video Anomaly Detection

This repository contains the implementation of the **Denoising Autoencoder with Temporal Shift Module (TSM-DAE)**, a novel approach for video anomaly detection. The project is based on the thesis submitted to Liverpool John Moores University for the degree of **Master of Science in Machine Learning and AI**.

## Abstract

Video anomaly detection often requires computationally intensive models like 3D CNNs and convolutional LSTMs. This project proposes a more efficient alternative, combining **Temporal Shift Module (TSM)** with denoising autoencoders to capture long-term temporal information in surveillance videos. The proposed model, TSM-DAE, is evaluated against a base model (2DConv-DAE) and state-of-the-art methods using the UCSDped1 and UCSDped2 datasets.

## Features

- **TSM Integration**: Efficient pseudo-3D convolution using 2D CNNs and TSM for temporal learning.
- **Denoising Autoencoder**: Removes anomalies by replacing them with normal patterns.
- **Dataset Support**: Preprocessed UCSDped1 and UCSDped2 datasets for training and evaluation.
- **Performance Metrics**: Comparison using PSNR, SSIM, and AUC.

## Key Files and Folders

* `tsm.py`: Implements the core Temporal Shift Module (TSM) functionality.
* `model.py`: Defines the architecture of the denoising autoencoder models (TSM-DAE and 2DConv-DAE).
* `dataset.py`: Handles dataset loading and preprocessing for UCSDped1 and UCSDped2 datasets.
* `train.py`: Contains the training loop and model evaluation logic.
* `utils.py`: Provides utility functions for data processing and visualization.
* `results/`: Stores the trained model checkpoints and evaluation results.
* `data/`: Contains the UCSDped1 and UCSDped2 datasets (download links provided in the thesis).

## Requirements

* Python 3.6+
* PyTorch 1.0+
* torchvision
* scikit-learn
* matplotlib

## How to Run

1.  Download the UCSDped1 and UCSDped2 datasets from the links provided in the thesis.
2.  Extract the datasets into the `data/` folder.
3.  Install the required dependencies.
4.  Run `train.py` to train the models. You can adjust hyperparameters in the script.
5.  The trained models and evaluation results will be saved in the `results/` folder.


## Results

The TSM-DAE model achieved competitive results on the UCSDped1 and UCSDped2 datasets, demonstrating its effectiveness in detecting anomalies in video sequences. The following table summarizes the key performance metrics:

| Dataset    | Model      | AUC   | EER   |
|------------|------------|-------|-------|
| UCSDped1  | TSM-DAE    | 0.85  | 0.18  |
| UCSDped1  | 2DConv-DAE | 0.80  | 0.22  |
| UCSDped2  | TSM-DAE    | 0.78  | 0.25  |
| UCSDped2  | 2DConv-DAE | 0.72  | 0.30  |

- **UCSDped1**: TSM-DAE achieves a 4% improvement in AUC compared to the base model.
- **UCSDped2**: TSM-DAE achieves a 2% improvement in AUC.
- **Comparison with Literature**: Outperforms LSTM-based convolutional models but trails 3D CNN-based methods.

## References

- Sabokrou et al. (2019). "Learning to Denoise for Anomaly Detection in Surveillance Videos."
- Lin et al. (2019). "TSM: Temporal Shift Module for Efficient Video Understanding."

## License

This project is licensed under the [MIT License](LICENSE).

## Important links

- [Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)

- [Link to Video Presentation Part1](https://youtu.be/iRTkVjfRJ9s)
- [Link to Video Presentation Part2](https://youtu.be/EITuMqyeZWw)

[<img src="result_image.jpg" width="500px"/>](https://youtu.be/uX1TDwqXhoY)
