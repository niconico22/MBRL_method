import logging
import datetime
# ログレベルを DEBUG に変更
now = datetime.datetime.now()
#filename = 'log_' + now.strftime('%Y%m%d_%H%M%S') + '.csv'
formatter = '%(levelname)s : %(asctime)s : %(message)s'

logging.basicConfig(filename='logger.log',
                    level=logging.DEBUG, format=formatter)
# logging.StreamHandler()
# 従来の出力
logging.info('error{}'.format('outputting error'))
logging.info('warning %s %s' % ('was', 'outputted'))
# logging のみの書き方
a = 1
logging.info('info')
