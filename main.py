import pandas as pd

import general_function as g
import pyupbit
import pandas as pd
import schedule
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError
print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 입력값
    # @access_key @secret_key 업비트로부터 할당받은 키
    # @alpha 신뢰구간, 이를 이용하여 구매값과 최소값을 설정
    # @investment 투자금액, 한번에 구매요청 금액
    # @coin_count 로직에 의한 선택되는 코인 갯수
    access_key = '5RsZuqMZ6T0tfyjNbIsNlKQ8LI4IVwLaYMBXiaa2'
    secret_key = 'zPKA1zJwymHMvUSQ2SqYWDgkxNgVfG7Z5jiNLcaJ'
    alpha = 1.5
    investment = 10000
    coin_count = 5
    coin_name_set = ['KRW-BTC', 'KRW-ETH']
    # @ratio 판매시 최소 마진에 해당하는 비율
    margin_ratio = 0.01

    # 가공 변수,
    upbit = pyupbit.Upbit(access_key, secret_key)

    # 스케줄링
    sched = BackgroundScheduler()
    sched.start()
    sched.add_job(g.coin_search, 'interval', seconds=60*5, id="search", args=[alpha])
    sched.add_job(g.buy_job, 'interval', seconds=60*10, id="buy_automation", args=[upbit, coin_count, investment])
    sched.add_job(g.buy_job, 'interval', seconds=60*10, id="buy_selection", args=[upbit, coin_count, investment, coin_name_set])
    sched.add_job(g.sell_job, 'interval', seconds=60*10, id="sell", args=[upbit, margin_ratio])
    sched.add_job(g.reservation_cancel, 'interval', seconds=60*30, id="cancel", args=[upbit])
