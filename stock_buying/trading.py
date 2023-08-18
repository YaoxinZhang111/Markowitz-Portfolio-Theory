import sys
sys.path.append('../')
from config import stock_list
stock_list.remove('sh.000001')
import baostock as bs
import pandas as pd
import datetime
import numpy as np
from keras.models import load_model
model = load_model('../model_20.h5')
ground = pd.read_csv('../data/sh.000001.csv')
max_value = ground.iloc[:,2:].max().values

windows = 24
captial = 200000

from scipy.optimize import minimize
def portfolio_performance(weights, returns, cov_matrix):
    returns = np.sum(returns*weights ) *12
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
    return std, returns


def minimize_volatility(weights, returns, cov_matrix):
    return portfolio_performance(weights, returns, cov_matrix)[0]

def mkws(returns,history_pct):
    # 马克维斯计算投资比例

    
    cov_matrix =np.cov(history_pct)
    
    num_assets = len(returns)
    args = (returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    initial_guess = num_assets*[1./num_assets,]

    optimal_volatility = minimize(minimize_volatility, initial_guess, args=args, bounds=bounds, constraints=constraints)
    
    
    # 输出结果
    weights = optimal_volatility['x']
    return weights 
    

start_day =  ((datetime.datetime.now())-datetime.timedelta(days=30*(windows+1))).strftime("%Y-%m-%d")

def df_tolist(df):
    # 将pandas.dataframe转list
    array_ = np.array(df)
    list_ = array_.tolist()
    return list_

datas = {}
#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

def get_ground_date(start_day):
    rs = bs.query_history_k_data_plus('sh.000001',
        "date",
        start_date=start_day,
        frequency="m", adjustflag="3")

    data_list = []

    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())

    result = pd.DataFrame(data_list, columns=rs.fields)
    ground_date = pd.to_datetime(result['date'])
    dates = ground_date.tolist()
    if len(dates) > windows:
        dates = dates[-windows:]
    elif len(dates) < windows:
        start_day = (start_day-datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        return get_ground_date(start_day)

    return dates,start_day

dates,start_day = get_ground_date(start_day=start_day)




#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
for code in stock_list:
    rs = bs.query_history_k_data_plus(code,
        "date,code,open,high,low,close,volume,turn,pctChg",
        start_date=start_day,
        frequency="m", adjustflag="3")
    

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result['date'] = pd.to_datetime(result['date'])
    result.iloc[:,2:] = result.iloc[:,2:].astype('float64')
    result['pctChg'] = result['pctChg']/100
    datas[code] = result
    print('fine:%s'% code)
    
#### 登出系统 ####
bs.logout()

date_data =[]
for date in dates:
    stock_data = []
    for stock in stock_list:
        data_stock = datas[stock]
        data_date = data_stock[data_stock['date'] == date]
        data_list = data_date.iloc[:,2:]
        stock_data.append(df_tolist(data_list)[0])
    date_data.append(stock_data)

data = np.array(date_data)
print(data.shape)
print(max_value.shape)

x_train = data/max_value
x_train = np.transpose(x_train, (1, 0, 2))

predict_value = (model.predict(x_train))*max_value[6]

history_pct = np.transpose(data[:,:,6], (1, 0))
action = mkws(predict_value[:,0], history_pct)
price = data[-1,:,3]

amounts = np.floor(action*captial/price/100)*100

log = pd.DataFrame(np.transpose([[dates[-1] for _ in stock_list],stock_list,amounts,price]),columns=['date','code','amount','buy_price'])

log.to_csv('./log/%s_trading_log.csv' % str(dates[-1])[:10])
