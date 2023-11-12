"""
This is an example algorithm that shows what is required for the judges to run your code

The program will take in the data (by passing in its path), and will output a numpy file

use this program like: `python3 main_algo.py -p train_data_50.csv`
"""
from eval_algo import eval_actions
from collections import defaultdict
import pandas as pd
from pathlib import Path
import argparse
import numpy as np
import talib as ta # see https://github.com/TA-Lib/ta-lib-python


# Read the data from the csv file
def readData(path):
    df = pd.read_csv(path)

    # change date into datetime objects
    df["Date"] = pd.to_datetime(df["Date"])

    # set indexes
    df.set_index(["Ticker", "Date"], inplace=True)

    return df


# Used to convert the dataframe into a numpy array
# col: the column to convert into a numpy array
#      in this case it can be "Open", "Close", "High", "Low", "Volume", etc
def pricesToNumpyArray(df, col="Open"):
    tickers = sorted(df.index.get_level_values("Ticker").unique())

    prices = []

    for ticker in tickers:
        stock_close_data = df.loc[ticker][col]
        prices.append(stock_close_data.values)

    prices = np.stack(prices)
    return prices

class Algo:
    def __init__(self, data_path, cash=25000, slowSMA=40, fastSMA=5):
        self.df = readData(data_path)
        self.open_prices = pricesToNumpyArray(self.df, col="Open")
        self.trades = np.zeros(self.open_prices.shape)

        # Use the below to hold your portfolio state
        self.cash = cash
        self.positions = [0]*len(self.open_prices)
        self.short_positions = defaultdict(list)
        self.port_values = [0] * len(self.open_prices[0])
        # initialize the portfolio value
        self.port_values[0] = self.cash

        # Algo specific
        self.slowSMA = slowSMA
        self.fastSMA = fastSMA

    def runSMA(self):
        # calculate trades based off of SMA momentum strategy

        # first calculate the SMAs
        fast_smas = []
        slow_smas = []
        for stock in range(len(self.open_prices)): 
            fast_sma = ta.SMA(self.open_prices[stock], timeperiod=self.fastSMA)
            slow_sma = ta.SMA(self.open_prices[stock], timeperiod=self.slowSMA)
            fast_smas.append(fast_sma)
            slow_smas.append(slow_sma)

        # now calculate trades
        for day in range(1, len(self.open_prices[0])-1):
            self.port_values[day] = self.calcPortfolioValue(day)

            # loop through each stock for the given day
            for stock in range(len(self.open_prices)): 
                fast_sma = fast_smas[stock]
                slow_sma = slow_smas[stock]

                # Buy: fast SMA crosses above slow SMA
                if fast_sma[day] > slow_sma[day] and fast_sma[day-1] <= slow_sma[day-1]:
                    # we are trading the next day's open price
                    self.trades[stock][day+1] = 1
                    self.handleBuy(stock, day+1, 1)
                
                # Sell/short: fast SMA crosses below slow SMA
                elif fast_sma[day] < slow_sma[day] and fast_sma[day-1] >= slow_sma[day-1]:
                    # we are trading the next day's open price
                    self.trades[stock][day+1] = -1
                    self.handleSell(stock, day+1, 1)
                # else do nothing
                else:
                    self.trades[stock][day+1] = 0

        # calculate the final portfolio value (after trades occur)
        self.port_values[-1] = self.calcPortfolioValue(len(self.open_prices[0])-1)

    def saveTrades(self, path):
        # for convention please name the file "trades.npy"
        np.save(path, self.trades)

    def handleBuy(self, stock, day, numShares):
        # case 1: we have a positive position and are buying
        if self.positions[stock] >= 0:
                self.cash -= self.open_prices[stock][day] * numShares

                if not self.cashValid():
                    # TODO: may want to handle this differently
                    print("INVALID CASH AMOUNT, COULD NOT AFFORD TRANSACTION")

                self.positions[stock] += numShares

        # case 2: we have short a position and are buying
        elif self.positions[stock] < 0:
            buy_amount = min(numShares - abs(self.positions[stock]), 0)
            short_close_amount = min(abs(self.positions[stock]), numShares)

            while short_close_amount > 0:
                # the amount of positions to close in the current short order
                positions_to_close = min(
                    self.short_positions[stock][0][1], short_close_amount
                )
                short_close_amount -= positions_to_close

                # the short value is = (short price - curr price)*amount of shares shorted
                self.cash += (
                    self.short_positions[stock][0][0] - self.open_prices[stock][day]
                ) * positions_to_close

                if positions_to_close == self.short_positions[stock][0][1]:
                    self.short_positions[stock].pop(0)

                else:
                    # subtract the amount of closed shares from the given price
                    self.short_positions[0][1] -= positions_to_close

            if buy_amount > 0:
                self.cash -= self.open_prices[stock][day] * buy_amount
                self.positions[stock] += buy_amount

    def handleSell(self, stock, day, numShares):
        # this handles selling and shorting
        # case 3: we have a positive position and are selling/shorting
        numShares = -abs(numShares)
        if self.positions[stock] > 0:
            sell_amount = min(abs(numShares), self.positions[stock])
            short_amount = max(abs(numShares) - self.positions[stock], 0)

            self.cash += self.open_prices[stock][day] * sell_amount
            self.positions[stock] -= sell_amount

            if short_amount > 0:
                short_price_amount = [self.open_prices[stock][day], short_amount]
                self.short_positions[stock].append(short_price_amount)

                self.positions[stock] -= short_amount

        # case 4: we have a short position and are selling/shorting
        # or don't have any position yet
        elif self.positions[stock] <= 0 and numShares:
            short_amount = abs(numShares)

            short_price_amount = [self.open_prices[stock][day], short_amount]
            self.short_positions[stock].append(short_price_amount)

            self.positions[stock] -= short_amount

    def cashValid(self):
        return self.cash >= 0
    
    def calcShortValue(self, day):
        # calculates value of all the short positions on 'day'
        # note that this value can be positive or negative
        val = 0
        for stock in self.short_positions.keys():
            for short_price, short_amount in self.short_positions[stock]:
                val += (short_price - self.open_prices[stock][day]) * short_amount

        return val
    
    def calcPortfolioValue(self, day):
        # the sum of cash and positive positions (and their value on 'day') + short positions value
        # cash + long positions + short positions

        value = self.cash
        for stock in range(len(self.open_prices)):
            if self.positions[stock] > 0:
                value += self.open_prices[stock][day] * self.positions[stock]
        return value + self.calcShortValue(day)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Demo Algorithm")

    # note that this could be the training dataset or the test dataset
    # so we can't hardcode things!
    parser.add_argument(
        "-p",
        "--prices",
        help="path to stock prices csv file",
    )

    prices_path = parser.parse_args().prices

    if prices_path is None:
        print("Please provide a path to a stock prices csv file using: main_algo.py -p <path to file>")
        exit(1)

    algo = Algo(prices_path)

    algo.runSMA()

    # we can run the evaluation for ourselves here to see how our trades did
    # you very likely will want to make your own system to keep track of your trades, cash, portfolio value etc, inside the 
    # runSMA function (or whatever equivalent function you have)
    port_values, sharpe_ratio = eval_actions(algo.trades,algo.open_prices, cash=25000,verbose=True)

    print(sharpe_ratio)
    print(algo.port_values[-1])

    algo.saveTrades("trades.npy")
