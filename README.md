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
python train.py --model lstm --loss mse
```

You can switch the loss function to QLIKE by changing the `--loss` parameter:

```bash
python train.py --model lstm --loss qlike
```

## References

- [Deep Learning for Volatility Forecasting in Asset Management](https://link_to_paper)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

