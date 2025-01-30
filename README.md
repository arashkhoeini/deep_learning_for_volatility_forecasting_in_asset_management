# Volatility Prediction using Deep Learning

This repository contains code to predict stock volatility using deep learning techniques. The model leverages past return and volatility time series data to make predictions. Currently, the codebase supports Long Short-Term Memory (LSTM) networks and two loss functions: Mean Squared Error (MSE) and QLIKE.

## Model

The LSTM model implemented in this repository is based on the paper "Deep Learning for Volatility Forecasting in Asset Management".

## Features

- **LSTM Network**: Utilizes LSTM for time series prediction.
- **Loss Functions**: Supports MSE and QLIKE loss functions.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/arashkhoeini/volatility_prediction.git
cd volatility_prediction
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```bash
python train.py model.arch=LSTM train.criterion=MSE
```

You can switch the loss function to QLIKE by changing the `train.criterion` parameter.

You can also change any other configurations by editing the file configs/config.yml


## References

- [Deep Learning for Volatility Forecasting in Asset Management](https://link.springer.com/article/10.1007/s00500-022-07161-1)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

