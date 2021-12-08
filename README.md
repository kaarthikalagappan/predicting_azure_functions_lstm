# Predicting Azure Function Invocations using LSTM
This is a experimental project where we try to predict when an Azure function will be invoked using the LSTM model. 3 hours of historical data is used to predict the invocation rate and times for the next one hour (these windows can be changed in the LSTM_model.py file).

### To Run:
Ensure that all the required library files are installed in the machine.
Data used for this project is located at: https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz and its corresponding documents is at: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md

Import the LSTM_model file into your main program and invoke the LSTM_Model function with the following parameters:
* prediction_function_hash - the hash value of the function to predict in the given file
* file_name - the path to the CSV file containing the data.

After the model is done training and testing, two images will be save showing the accuracy and loss values.