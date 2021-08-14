import pandas as pd
import numpy as np
import xgboost
import pyupbit
import time
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 현재 내 코인 정보
def my_coin(upbit):
    my_balances = pd.DataFrame(upbit.get_balances())
    if len(my_balances)<2:
        print('판매 가능한 코인이 없습니다.')
    else:
        my_balances['coin_name'] = my_balances.unit_currency + '-' + my_balances.currency
        my_balances.reset_index(drop=True, inplace=True)
        my_balances = my_balances[pd.to_numeric(my_balances.balance) > 0]
        my_balances = my_balances[pd.to_numeric(my_balances.avg_buy_price) > 0]
    return my_balances

def f_rsi(df, line):
    df = df[-line:]
    a = df['close'] - df['open']
    b = np.sum(a[a >= 0])
    c = abs(np.sum(a[a < 0]))
    if (b == 0) & (c == 0):
        rsi = 50
    else:
        rsi = (b / (c + b) * 100)
    return rsi


def model_xgboost(df):
    y = df['close'][1:].values
    X = df.drop('close',axis=1)[:-1].values
    P = df.drop('close',axis=1)[-1:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                                     colsample_bytree=1, max_depth=7)
    r_sq = xgb_model.fit(X_train, y_train)
    predictions = r_sq.predict(P)
    result = predictions[0]
    return result

def model_coef(value):
    y = pd.DataFrame(value).values
    X = pd.DataFrame(range(1,len(value)+1)).values.reshape(-1,1)
    line_fitter = LinearRegression()
    line_fitter.fit(X, y)
    coef = line_fitter.coef_
    return coef[0][0]

def coin_point(coin_name, alpha):
    # input
    result = list()
    result_rsi = list()
    intervals = ['days', "minute240", "minute60", "minute30", "minute15", 'minute10', 'minute5', 'minute3', 'minute1']
    lines = [30, 30, 30, 30, 30, 30, 30, 30, 30]

    currency = pyupbit.get_current_price(coin_name)
    for idx in range(len(intervals)):
        # coin_name = tickers[0]
        # interval = intervals[0]
        # line = lines[0]
        #idx =0
        interval = intervals[idx]
        line = lines[idx]
        #현재 시세
        df = pd.DataFrame()
        while len(df)==0:
            df = pyupbit.get_ohlcv(coin_name, interval=interval, count=line)
            time.sleep(1)
        pred_price = model_xgboost(df)
        rsi = f_rsi(df,line)
        result.append(pred_price)
        result_rsi.append(rsi)
    result.reverse()
    result_rsi.reverse()
    coef_price = model_coef(result)
    coef_rsi = model_coef(result_rsi)
    ratio = coef_price/currency
    rsi =np.mean(result_rsi)
    mu = np.mean(result)
    sd = np.std(result)
    max_price = mu + alpha * sd
    min_price = mu - alpha * sd
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    dict = {'idx':nowDatetime, 'coin_name':coin_name, 'currency':currency, 'predict':mu, 'coef_price':coef_price,'max':max_price, 'min':min_price, 'rsi':rsi, 'coef_rsi':coef_rsi, 'ratio':ratio}
    #print(dict)
    return dict

#가격 소수점 설정

def round_price(price):
    if price < 10:
        price = round(price, 2)
    elif price < 100:
        price = round(price, 1)
    elif price < 1000:
        price = round(price, 0)
    elif price < 10000:
        price = round(price/5, 0)*5
    elif price < 100000:
        price = round(price/10, 0)*10
    elif price < 500000:
        price = round(price/50, 0)*50
    elif price < 1000000:
        price = round(price /100, 0) * 100
    else:
        price = round(price / 1000, 0) * 1000
    return price

def buy_coin(upbit, coin_name, price, investment):
    # coin_name = 'KRW-KMD'
    count = investment / price
    try:
        result = upbit.buy_limit_order(coin_name, price, count)
        print(result)
        time.sleep(1)
        if len(result) > 2:
            uuid = result.get('uuid')
            f = open('./buy_list/'+uuid+'.txt','w')
            f.close()
            # ask = pd.DataFrame(result, index=[0])
            # if os.path.isfile('buy_list.csv'):
            #     origin_df = pd.read_csv('buy_list.csv')
            #     ask = pd.concat([ask, origin_df], axis=0).reset_index(drop=True)
            # ask.to_csv('buy_list.csv', index=False)
        else:
            pass
    except:
        print('매수 error: ' + coin_name)

def sell_coin(upbit, balance, coin_name, price):
    price = round_price(price)
    if price * balance > 5000:
        try:
            result = upbit.sell_limit_order(coin_name, price, balance)
            time.sleep(1)
            if len(result) > 2:
                if len(result) > 2:
                    uuid = result.get('uuid')
                    f = open('./sell_list/' + uuid + '.txt', 'w')
                    f.close()
        except:
            pass
    else:
        print("판매 실패: "+coin_name +"은 판매 최소금액이 부족")

import os

def convert(set):
    return [*set, ]

def reservation_cancel(upbit):
    dir_list = ['./buy_list', './sell_list']
    for dir in dir_list:
        file_list = os.listdir(dir)
        if len(file_list)>=1:
            for file in file_list:
                uuid = file.replace('.txt','')
                upbit.cancel_order(uuid)
                os.remove(dir+'/'+file)
                time.sleep(1)
        else:
            print("예약이 없음:", dir)

def buy_job(upbit, coin_count, investment, coin_name_set = []):
    print('buy')
    df_search = pd.read_csv('./result/latest.csv')
    if len(coin_name_set)>=1:
        for coin_name in coin_name_set:
            df = df_search[df_search['coin_name'] == coin_name].reset_index(drop=True)
            #coin_name = df['coin_name']
            price = round_price(df['min'][0])
            buy_coin(upbit, coin_name, price, investment)
    else:
        df_selection = df_search[0:coin_count]
        buy_list = list()
        for idx in range(len(df_selection)):
            #idx = 0
            df = df_selection.iloc[idx,:]
            coin_name = df['coin_name']
            price = round_price(df['min'])
            buy_coin(upbit, coin_name, price, investment)


def sell_job(upbit, margin_ratio):
    print('sell')
    my_coin_list = my_coin(upbit)
    if len(my_coin_list) >= 1:
        df_search = pd.read_csv('./result/latest.csv')
        for idx in range(len(my_coin_list)):
            coin_name = my_coin_list['coin_name'].to_list()[idx]
            coin_df = my_coin_list[my_coin_list['coin_name'] == coin_name].reset_index(drop=True)
            sell_price = df_search[df_search['coin_name'] == coin_name]['max'].reset_index(drop=True)[0]
            min_benefit = float(coin_df.avg_buy_price[0]) * (1 + margin_ratio)
            balance = float(coin_df.balance[0])
            sell_coin(upbit, balance, coin_name, max(min_benefit, sell_price))


def coin_search(alpha):
    print('search')
    tickers = pyupbit.get_tickers(fiat="KRW")
    coin_info_set = list()
    for coin_name in tickers:
        coin_info = coin_point(coin_name, alpha)
        coin_info_set.append(coin_info)
        time.sleep(0.5)
    df = pd.DataFrame(coin_info_set).reset_index(drop=True)
    #df.sort_values('ratio', ascending=True, inplace=True)
    df.sort_values('ratio', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    name = str(df.iloc[0, 0]).replace(':','').replace('-','').replace(' ','')
    #df.to_csv('./result/'+name+'.csv', index=False)
    df.to_csv('./result/latest.csv', index=False)

