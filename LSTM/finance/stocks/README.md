# Stock Prediction with LSTM Recurrent Networks

If someone could predict the daily or hourly price of a company's stock, they'd be a billionaire. But what if we could predict the overall trend of a company's stock?

### Predictions

The price of a company's stock is dependent on a lot of different variables. In order to predict the price, experts will use loads of different types of data to make their educated guess on what will happen. With the adoption of computers, this makes their life even easier as numbers can be crunched to help with the process. The one problem with using a strictly defined mathematical model is that the model might not work for other stocks or other types of assets. What if we could create a model that adapts well to the data it's looking at? Well, we can, by using Artificial Intelligence and Deep Learning.

### Recurrent Networks

Why use a recurrent network? We want to use a recurrent network because it allows the model to remember data that it's seen before. This is extremely useful in time series regression.

### LSTM Networks

So why don't we just use a SRN (Simple Recurrent Network). The problem with SRNs is the exploding and vanishing gradient issue. This makes back-propagation of the network harder or the network becomes too overfit. LSTM cells have the ability to forget data that is no longer relevant. They also have other abilities that don't need an explanation for right now.

### Bidirectional LSTM Recurrent Networks

To predict the stock market, we can't just assume the future will be exactly like the past. To help improve our accuracy for data the model hasn't seen, we'll use a Bidirectional approach to our recurrent network. A Bidirectional approach has one LSTMs prediction be the past data input for the another LSTM cell and vise versa. This allows to take the prediction into account when we're trying to make another prediction.


### Data

For the examples, previous closing price data is used to predict closing prices in the future. This works ok for getting the basic curve of the stock, stocks are extremely complicated and rely on a ton of different data points. Luckily, since we're using a computer we can take other data into account (in the future). Other stocks related to the current stock, market cap, company blog posts, news coverage, production line output, new government tariffs, video and audio transcriptions of news reports, and much more can be used to help fine-tune the model in order to help predict the future stock price. For non-numerical data such as news headlines and audio transcriptions, we can use NLP (Natural Language Processing) to gather sentiment data on those articles in order to refine the model. If we can get data into vector format, we can feed it to our model for training.

### Efficiency

For the models in this project, the run time is around 15 minutes, on a CPU of a MacBook Pro 2.9 GHz Intel i7 with 16GB of RAM. Obviously, the more data we add to our model, the longer the time it will take to train. Therefore we need to find data that directly affects the performance of the model. So if we had a choice between what BuzzFeedNews says vs. what Bloomberg says, we'd most likely want to take data from Bloomberg, as business is their focus.


### Measuring Error

To measure the error of our models we use the Root Mean Squared Error Statistic. The lower the RMSE, the better the model is. When training and testing, we need to remember that the model might be extremely accurate to the training data, but might lack in performance when trying to analyze data it hasn't seen yet. That's why it's important to calculate the RMSE for both the training and testing data set. If the RMSE is low for training and high for testing we know the model is too overfit to the training data, and we can either add more data or implement a higher dropout rate for our network.


### Examples

There is a prediction Jupyter Notebook on [TSLA Price Prediction](TSLA%20Price%20Prediction.ipynb), as well as python script that allows for the model to be trained on any stock. [stock_predictor.py](stock_predictor.py)

An example of using higher dimensional data as input to learn high level feature representations between data: [Multi-Dimensional input for Stock Predictions](Multi-Dimensional%20input%20for%20Stock%20Prediction.ipynb)
