from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import keras
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.impute import SimpleImputer
from keras.layers import GaussianNoise



app = Flask(__name__)
CORS(app)

# Sample Tasks
tasks = [
    {"id": 1, "name": "Task 1"},
    {"id": 2, "name": "Task 2"},
    {"id": 3, "name": "Task 3"}
]

# API Endpoints
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    try:
        return jsonify(tasks)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/stock_data', methods=['GET'])
def get_stock_data():
    
    # Override how Pandas Datareader reads data
    yf.pdr_override()
    
    # Fetch stock data using YahooFinance
    today = datetime.now()
    start = datetime(today.year - 2, today.month, today.day)
    end = datetime(today.year, today.month, today.day)
    stock = request.args.get('stock_name')
    df = yf.download(stock, start, end)
    df.index = df.index.astype(str)
    return jsonify(df.to_dict())

    # # df.tail()

    # # Ticker for searching news
    # stockraw = stock[:-3]

    # # stockraw

    # # Getting data using html requests
    # link = f"https://news.google.com/search?q={stockraw}&hl=en-IN&gl=IN&ceid=IN%3Aen"
    # req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    # webpage = urlopen(req).read()
    # # print(webpage)

    # # Extracting headlines and the dates when they were published to find sentiments
    # with requests.Session() as c:
    #     soup = BeautifulSoup(webpage, 'html.parser')

    #     # Extract titles and datetimes simultaneously
    #     titles = soup.find_all('a', class_='JtKRv')
    #     datetimes = soup.find_all('time', class_='hvbAAd')

    #     # Create lists to store extracted data
    #     dates_list = []
    #     titles_list = []

    #     for title, datetime_tag in zip(titles, datetimes):
    #         title_text = title.text.strip()  # Remove leading/trailing whitespace from title
    #         datetime_str = datetime_tag['datetime']
    #         formatted_datetime = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S%z')  # Parse datetime string

    #         # Append data to respective lists
    #         dates_list.append(formatted_datetime.date())
    #         titles_list.append(title_text)

    #     # Create DataFrame
    #     df2 = pd.DataFrame({'Date': dates_list, 'Headline': titles_list})

    #     # Print DataFrame
    #     # print(df2.head)

    # # Loading the tokenizer to generate tokens and model to perform sentiment analysis
    # tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    # model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    # # Finding the encodings for each of the headlines found
    # encoded_headlines = []
    # for headline in df2['Headline']:
    #     encoded_headline = tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
    #     encoded_headlines.append(encoded_headline)
    # # print(encoded_headlines)

    # # Finding the sentiment scores of each headline
    # # Initialize an empty list to store the tweet sentiments
    # headline_sentiments = []

    # # Iterate over the encoded tweets
    # for encoded_headline in encoded_headlines:
    #     # Pass the encoded tweet to the model
    #     output = model(**encoded_headline)

    #     # Extract the scores
    #     scores = output[0][0].detach().numpy()
    #     scores = softmax(scores)

    #     # Calculate the sentiment score between -1 and 1
    #     sentiment_score = scores[2] - scores[0]  # Assuming 'labels' are ordered as [negative, neutral, positive]

    #     # Append the sentiment score to the list
    #     headline_sentiments.append(sentiment_score)

    # # Print the tweet sentiments
    # # for i in headline_sentiments:
    # #     print(i)

    # # Add the tweet sentiments as a new column to the DataFrame
    # df2['Sentiments'] = headline_sentiments

    # # Print the DataFrame with the added column
    # # print(df2)

    # # Convert 'Date' column to datetime if it's not already
    # df2['Date'] = pd.to_datetime(df2['Date'])

    # # Group by 'Date' and calculate the average sentiment for each day
    # average_sentiments = df2.groupby('Date')['Sentiments'].mean().reset_index()

    # # Print the DataFrame with one row per date and the average sentiment for each day
    # # print(average_sentiments)

    # # Merge the dataframes on the 'Date' column with a left join to keep all rows from df
    # merged_df = pd.merge(df, average_sentiments, on='Date', how='left')

    # # Fill missing sentiment scores with 0
    # merged_df['Sentiments'].fillna(0, inplace=True)

    # # If you want to ensure that 'Sentiments' column is of numeric type
    # merged_df['Sentiments'] = pd.to_numeric(merged_df['Sentiments'])

    # # If you want to overwrite the 'Sentiments' column with 0 where there's no match
    # merged_df['Sentiments'].fillna(0, inplace=True)

    # # merged_df.tail(20)

    # # print(merged_df.index.dtype)

    # # Isolating dates to plot graphs more conveniently
    # # train_dates = pd.to_datetime(merged_df['Date'])

    # # print(train_dates)

    # # Isolate variables for training
    # # cols = list(merged_df)[1:8]
    # # print(cols)
    # # cols2 = list(merged_df)[1:7]
    # # print(cols2)
    # # cols3 = list(merged_df)[1:6]
    # # print(cols3)

    # # Displaying Open, High, Low, Close and Adjusted Closing price
    # # df_for_display = merged_df[cols3].astype(float)
    # # df_for_display.plot.line()

    # # Displaying traded Volume
    # # df_for_display = merged_df['Volume'].astype(float)
    # # df_for_display.plot.line()

    # # Displaying Sentiments
    # # df_for_display = merged_df['Sentiments'].astype(float)
    # # df_for_display.plot.line()

    # # Storing all data to be fed to the model
    # # df_for_training = merged_df[cols].astype(float)

    # # df_for_training

    # # Plotting the type of distribution of each column in df_for_training
    # # df_for_training.hist(alpha=0.5, figsize=(20,10))
    # # plt.show()

    # # Selecting the features
    # dataFrame = merged_df

    # # dataFrame

    # # To handle missing values
    # imputer = SimpleImputer(missing_values=np.nan)

    # # Removing date and changing index
    # dataFrame.drop(columns=['Date'], inplace=True)

    # # Handling Missing values
    # dataFrame = pd.DataFrame(imputer.fit_transform(dataFrame), columns=dataFrame.columns)
    # dataFrame = dataFrame.reset_index(drop=True)
    # # print(dataFrame.shape)

    # # Applying feature scaling
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # df_scaled = scaler.fit_transform(dataFrame.to_numpy())
    # df_scaled = pd.DataFrame(df_scaled, columns=list(dataFrame.columns))
    # # print(list(dataFrame.columns))
    # # print(df_scaled.shape)
    # target_scaler = MinMaxScaler(feature_range=(0, 1))
    # df_scaled[['Open', 'Close']] = target_scaler.fit_transform(dataFrame[['Open', 'Close']].to_numpy())
    # df_scaled = df_scaled.astype(float)
    # # print(df_scaled.shape)

    # # Function to create sequences
    # def singleStepSampler(df, window):
    #     xRes = []
    #     yRes = []
    #     for i in range(0, len(df) - window):
    #         res = []
    #         for j in range(0, window):
    #             r = []
    #             for col in df.columns:
    #                 r.append(df[col][i + j])
    #             res.append(r)
    #         xRes.append(res)
    #         yRes.append(df[['Open', 'Close']].iloc[i + window].values)
    #     return np.array(xRes), np.array(yRes)

    # # Dataset splitting
    # SPLIT = 0.85
    # (xVal, yVal) = singleStepSampler(df_scaled, 20)
    # X_train = xVal[:int(SPLIT * len(xVal))]
    # y_train = yVal[:int(SPLIT * len(yVal))]
    # X_test = xVal[int(SPLIT * len(xVal)):]
    # y_test = yVal[int(SPLIT * len(yVal)):]
    # # print(xVal.shape)
    # # print(yVal.shape)
    # # print(X_train.shape)
    # # print(y_train.shape)
    # # print(X_test.shape)
    # # print(y_test.shape)

    # # Building the model
    # multivariate_lstm = keras.Sequential()
    # multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
    # multivariate_lstm.add(keras.layers.Dropout(0.2))
    # multivariate_lstm.add(keras.layers.Dense(2, activation='linear'))
    # multivariate_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE'], optimizer='Adam')
    # multivariate_lstm.summary()

    # # Fitting the data to the model
    # history = multivariate_lstm.fit(X_train, y_train, epochs=100)

    # # list(merged_df.columns)

    # # Reload the data with the date index
    # dataFrame = merged_df

    # # Forecast Plot with Dates on X-axis
    # predicted_values = multivariate_lstm.predict(X_test)

    # d = {
    #     'Predicted_Open': predicted_values[:, 0],
    #     'Predicted_Close': predicted_values[:, 1],
    #     'Actual_Open': y_test[:, 0],
    #     'Actual_Close': y_test[:, 1],
    # }

    # d = pd.DataFrame(d)
    # d.index = dataFrame.index[-len(y_test):] # Assigning the correct date index

    # fig, ax = plt.subplots(figsize=(10, 6))
    # # highlight the forecast
    # highlight_start = int(len(d) * 0.9)
    # highlight_end = len(d) - 1 # Adjusted to stay within bounds
    # # Plot the actual values
    # plt.plot(d[['Actual_Open', 'Actual_Close']][:highlight_start], label=['Actual_Open', 'Actual_Close'])

    # # Plot predicted values with a dashed line
    # plt.plot(d[['Predicted_Open', 'Predicted_Close']], label=['Predicted_Open', 'Predicted_Close'], linestyle='--')

    # # Highlight the forecasted portion with a different color
    # plt.axvspan(d.index[highlight_start], d.index[highlight_end], facecolor='lightgreen', alpha=0.5, label='Forecast')

    # plt.title('Multivariate Time-Series forecasting using LSTM')
    # plt.xlabel('Dates')
    # plt.ylabel('Values')
    # ax.legend()
    # plt.show()

    # # Reload the data with the date index
    # dataFrame = merged_df

    # # Forecast Plot with Dates on X-axis
    # predicted_values = multivariate_lstm.predict(X_test)

    # # Array to compensate for inverse transform shape
    # additional_zeros = np.zeros((len(y_test),5))
    # additional_zeros_df = pd.DataFrame(additional_zeros)

    # # Adding zeros to y_test
    # y_test_extended = np.concatenate((y_test, additional_zeros), axis=1)

    # # Adding zeros to predicted_values
    # predicted_values_extended = np.concatenate((predicted_values, additional_zeros), axis=1)

    # # Inverse transform the data
    # y_test_inv = scaler.inverse_transform(y_test_extended)
    # predicted_values_inv = scaler.inverse_transform(predicted_values_extended)

    # # Create the DataFrame with inverse-transformed data
    # d = {
    #     'Predicted_Open': predicted_values_inv[:, 0],
    #     'Predicted_Close': predicted_values_inv[:, 1],
    #     'Actual_Open': y_test_inv[:, 0],
    #     'Actual_Close': y_test_inv[:, 1],
    # }

    # d = pd.DataFrame(d)
    # d.index = dataFrame.index[-len(y_test):]  # Assigning the correct date index

    # # Plotting
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Highlight the forecast
    # highlight_start = int(len(d) * 0.9)
    # highlight_end = len(d) - 1  # Adjusted to stay within bounds

    # # Plot the actual values
    # plt.plot(d[['Actual_Open', 'Actual_Close']][:highlight_start], label=['Actual_Open', 'Actual_Close'])

    # # Plot predicted values with a dashed line
    # plt.plot(d[['Predicted_Open', 'Predicted_Close']], label=['Predicted_Open', 'Predicted_Close'], linestyle='--')

    # # Highlight the forecasted portion with a different color
    # plt.axvspan(d.index[highlight_start], d.index[highlight_end], facecolor='lightgreen', alpha=0.5, label='Forecast')

    # plt.title('Future Price Predictions Using LSTM')
    # plt.xlabel('Dates')
    # plt.ylabel('Values')
    # ax.legend()
    # plt.show()
    
    # # Highlight the forecast
    # highlight_start = int(len(d) * 0.9)
    # highlight_end = len(d) - 1  # Adjusted to stay within bounds

    # # Prepare the data for JSON serialization
    # plot_data = {
    #     'actual_open': d['Actual_Open'][:highlight_start].tolist(),
    #     'actual_close': d['Actual_Close'][:highlight_start].tolist(),
    #     'predicted_open': d['Predicted_Open'].tolist(),
    #     'predicted_close': d['Predicted_Close'].tolist(),
    #     'highlight_start': str(d.index[highlight_start]),
    #     'highlight_end': str(d.index[highlight_end]),
    #     'title': 'Future Price Predictions Using LSTM',
    #     'x_label': 'Dates',
    #     'y_label': 'Values'
    # }

    # # Convert the plot data to JSON
    # plot_json = json.dumps(plot_data)

    # # Print or return the JSON data
    # # print(plot_json)

    # predicted_values_inv[-20:, :2]
    # return (plot_json)
    
    

if __name__ == '__main__':
    app.run(debug=True)
