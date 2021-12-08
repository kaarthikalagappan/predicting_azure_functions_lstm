import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf 

# Based on "How To Do Multivariate Time Series Forecasting Using LSTM" (https://analyticsindiamag.com/how-to-do-multivariate-time-series-forecasting-using-lstm/)

"""
Function that is used to round the predicted values since invocations don't happen at non-integer levels
"""
def roundPredictions(y_pred):
    return_values = [np.floor(i) if i % 1 <= 0.5 else np.ceil(i) for i in y_pred]
    print(return_values)
    return return_values

"""
To get the metrics after training and testing
Keyword arguments:
    y_true -- the true values from test data
    y_pred -- the values predicted by the model
"""
def timeseries_evaluation_metrics_func(y_true, y_pred):
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n') 


"""
The LSTM Model 
Keyword arguments:
    prediction_function_hash -- The hash value of the function to predict
    file_name -- File name where the CSV data file is present
"""
def LSTM_Model(prediction_function_hash, file_name = "data.csv"):
    if prediction_function_hash == "":
        print("Please pass in a valid function hash")
        exit(1)
    
    data = pd.read_csv(file_name)

    for i in data.select_dtypes('int').columns:
        le = LabelEncoder().fit(data[i])
        data[i] = le.transform(data[i]) 


    hist_window_hours = 3 # Use 3 hours of past data
    hist_window = int(hist_window_hours * 60) # since our data is in minutes
    horizon_hours = 1 # predict for the next 60 minutes
    horizon = int(horizon_hours * 60) # since our data is in minutes
    train_split_percentage = 0.8 # 80% for training
    columns = data.columns[1:].values
    # print(columns)
    prediction_column = [str(prediction_function_hash)]


    validate = data[columns].tail(horizon)
    data.drop(data.tail(horizon).index,inplace=True)


    # Using a mixmax scaler
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X_data = X_scaler.fit_transform(data[columns])
    # print(len(X_data))
    Y_data = Y_scaler.fit_transform(data[prediction_column]) 
    TRAIN_SPLIT = int(len(X_data) * train_split_percentage)

    # To process the given data into train and validation
    def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
        X = []
        y = []
        start = start + window
        if end is None:
            end = len(dataset) - horizon
        for i in range(start, end):
            indices = range(i-window, i)
            X.append(dataset[indices])
            indicey = range(i+1, i+1+horizon)
            y.append(target[indicey])
        return np.array(X), np.array(y) 


    x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
    x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon) 

    # Model training value constants
    batch_size = 256
    buffer_size = 150
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
    val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
    val_data = val_data.batch(batch_size).repeat() 


    # Defining the model itself
    lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), 
                                input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.LSTM(horizon)
    ])
    lstm_model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    lstm_model.summary() 


    model_path = 'Bidirectional_LSTM_Multivariate.h5'
    early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
    checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
    callbacks=[early_stopings,checkpoint] 
    history = lstm_model.fit(train_data,epochs=150,steps_per_epoch=100,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)

    # Plotting and saving the loss values
    plt.figure(figsize=(16,9))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'])
    # plt.show() 
    plt.savefig(f'{file_name}_train_valid_loss.png')


    # Testing
    data_val = X_scaler.fit_transform(data[columns].tail(hist_window))
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pred = lstm_model.predict(val_rescaled)
    pred_Inverse = Y_scaler.inverse_transform(pred)
    pred_Inverse 


    # Rounding values
    pred_Inverse[0] = roundPredictions(pred_Inverse[0])
    timeseries_evaluation_metrics_func(validate[prediction_column[0]],pred_Inverse[0])

    # Plotting and saving the predicted values vs actual values for visual comparison
    plt.figure(figsize=(16,9))
    plt.plot( list(validate[prediction_column[0]]))
    plt.plot( list(pred_Inverse[0]))
    plt.title("Actual vs Predicted")
    plt.ylabel("Number of Invocations")
    plt.legend(('Actual','predicted'))
    # plt.show() 
    plt.savefig(f'{file_name}_actual_pred_values.png')


if __name__ == "__main__":
   # Function to run when not called via 'import'
    LSTM_Model()