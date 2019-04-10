#!/usr/bin/python
# -*- coding:utf-8 -*-
#%%

from predict import Statskeywords
from predict import StatsFeatures
import pre_cor
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

import logging.config
from datetime import datetime
from flask import Flask, request, jsonify

#%% 基本设置
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 日志记录
logging.config.fileConfig("conf/logger.conf")
logger = logging.getLogger("rotating")

# 分类模型
#correlation_pipeline = joblib.load("model/youlang_model.pkl.z")
#joblib.dump(correlation_pipeline, "model/youlang_model_2.pkl.z")
correlation_pipeline = joblib.load("model/youlang_model_2.pkl.z")

@app.route('/judge_correlation_i', methods=['POST'])
def judge_correlation_i():
    """
    判断相关性：二分类模型，判断某条数据（新闻）是否是行业、机构相关
    """
    start_time = datetime.now()
    records = request.json['record']
    logger.info('starting judge_correlation_i, {list_size: %d}' % (len(records)))

    try :
        words_list = pre_cor.handle_contents([record['content'] for record in records])
        correlation_res = correlation_pipeline.predict(words_list)

        ret_list = []
        for index, record_result in enumerate(records):
            id = int(record_result['id'])
            cor = int(correlation_res[index])

            ret_list.append({'id': id, 'cor': cor})
    except Exception as e:
        print(e)
        ret_list = [{'id': record['id'], 'cor': 2} for record in records]

    finally :
        # 返回结果
        logger.info('end judge_correlation_i: {ret_list: %d, lost_seconds: %ds}' % (
            len(ret_list), (datetime.now() - start_time).seconds))
        ret = {'docs': ret_list, 
               'elapsed_time': '%0.2f'%((datetime.now() - start_time).seconds)}

    return jsonify(ret)
    
#%%
if __name__ == '__main__':
    from werkzeug.contrib.fixers import ProxyFix   
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0', port=11000, threaded=True)  
    
    