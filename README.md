# Deep Learning for Volatility Forecasting in Asset Management

This repository contains unofficial implementation for the paper: "Deep Learning for Volatility Forecasting in Asset Management"

## Model

It supports two stm models:
- **LSTM1:** Which is trained using only time-series data of only one stock, and predicts the next day volatility for that stock.
- **LSTMn:** Which is trained using time-series data of all the stocks in the provided dataset, and predicts the next day volatility for all those stocks.

## Losses

It supports two losses to train and evaluate the model:

- **MSE**
- **QLIKE**

## Datasets

The original paper reports results of two datasets:

- **Dow Jones plus SPY:** Which includes 29 assets
- **NASDAQ 100 index**: THIS IS NOT SUPPORTED IN THIS CODEBASE YET

## Installation

```bash
git clone https://github.com/arashkhoeini/deep_learning_for_volatility_forecasting_in_asset_management.git
cd volatility_prediction
pip install -r requirements.txt
```

## Usage

To train the LSTM1 using Dow Jones dataset and MSE loss run the following command:

```bash
python train.py model.arch=LSTM1 dataset=DJI500 train.criterion=MSE
```

To train the LSTMn using Dow Jones dataset and QLIKE loss run the following command:

```bash
python train.py model.arch=LSTMn dataset=DJI500 train.criterion=QLIKE
```

## Disclaimer

The original paper does not report important hyper-parameters, including the learning rate! Therefore the results provided in the paper are not reproducible using this codebase, unless you find the exact same hyper-parameters.

## References

- [Deep Learning for Volatility Forecasting in Asset Management](https://link.springer.com/article/10.1007/s00500-022-07161-1)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

