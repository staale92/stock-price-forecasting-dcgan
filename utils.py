import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def process_dataframe(dataframe):
    datetime_series = pd.to_datetime(dataframe['date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df = dataframe.set_index(datetime_index)
    df = df.sort_values(by='date')
    df = df.drop(columns='date')
    return df

def plot_autocorrelations(y_value):
    # plots
    plt.figure(figsize=(20,10))
    lags = 100
    # acf
    axis = plt.subplot(2, 1, 1)
    plot_acf(y_value, ax=axis, lags=lags)
    # pacf
    axis = plt.subplot(2, 1, 2)
    plot_pacf(y_value, ax=axis, lags=lags)
    # show plot
    plt.show()

def scaler_function(X, y):
    # Normalize the data
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaler.fit(X)
    y_scaler.fit(y)

    return X_scaler, y_scaler

def get_X_y(X_data, y_data, n_steps_in, n_steps_out):
    X = list()
    y = list()
    past_y = list()

    length = len(X_data)
    for i in range(0, length, 1):
        X_value = X_data[i: i + n_steps_in][:, :]
        y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
        past_y_value = y_data[i: i + n_steps_in][:, :]
        if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
            X.append(X_value)
            y.append(y_value)
            past_y.append(past_y_value)

    return np.array(X), np.array(y), np.array(past_y)

def predict_index(dataset, X_train, n_steps_in, n_steps_out):

    train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

    return train_predict_index, test_predict_index


def split_train_test(data, train_dimension=0.8):
    train_size = round(len(data) * train_dimension)
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

def real_predicted_results(rescaled_Predicted_price, rescaled_Real_price, index_train, output_dim):
    predict_result = pd.DataFrame()
    for i in range(rescaled_Predicted_price.shape[0]):
        y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=index_train[i:i+output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    real_price = pd.DataFrame()
    for i in range(rescaled_Real_price.shape[0]):
        y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=index_train[i:i+output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    return predict_result, real_price

def get_test_plot(generator_model, X_test, y_test, y_scaler, index_test, output_dim):
    # Set output steps
    output_dim = y_test.shape[1]

    # Get predicted data
    y_predicted = generator_model(X_test)
    rescaled_real_y = y_scaler.inverse_transform(y_test)
    rescaled_predicted_y = y_scaler.inverse_transform(y_predicted)

    # Predicted price
    predict_result = pd.DataFrame()
    for i in range(rescaled_predicted_y.shape[0]):
        y_predict = pd.DataFrame(rescaled_predicted_y[i], columns=["predicted_price"],
                                 index=index_test[i:i + output_dim])
        predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)

    # Real price
    real_price = pd.DataFrame()
    for i in range(rescaled_real_y.shape[0]):
        y_train = pd.DataFrame(rescaled_real_y[i], columns=["real_price"], index=index_test[i:i + output_dim])
        real_price = pd.concat([real_price, y_train], axis=1, sort=False)

    predict_result['predicted_mean'] = predict_result.mean(axis=1)
    real_price['real_mean'] = real_price.mean(axis=1)

    # Plot the predicted result
    plt.figure(figsize=(16, 8))
    plt.plot(real_price["real_mean"])
    plt.plot(predict_result["predicted_mean"], color='r')
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
    plt.title("Test set", fontsize=20)
    plt.savefig('./outcome/test_predictions.png')
    
    # Calculate RMSE, MAE, and MAPE
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    MAE = mean_absolute_error(predicted,real)
    MAPE = mean_absolute_percentage_error(predicted,real)
    print('Test RMSE: ', RMSE)
    print('Test MAE: ', float(MAE))
    print('Test MAPE: ', "{0:.5f}%".format(MAPE))

    return predict_result, RMSE, MAE, MAPE
