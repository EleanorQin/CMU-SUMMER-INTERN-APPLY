import xgboost as xgb
import os
import sys, time, json
from pyspark import SparkContext
from math import pow, sqrt
from datetime import date, datetime
import numpy as np

today = str(date.today())
def diff_year(x):
    yelp_date_obj = datetime.strptime(x[0:10], "%Y-%m-%d")
    today_date_obj = datetime.strptime(today[0:10], "%Y-%m-%d")
    return (today_date_obj.year - yelp_date_obj.year) + (today_date_obj.month - yelp_date_obj.month)/12


def variance(rating_list, avg_star):
    _sum = 0
    for r in rating_list:
        _sum += pow((r - avg_star), 2)
    return sqrt(_sum/len(rating_list))


if __name__ == '__main__':
    start = time.time()

    if len(sys.argv) != 4:
        print("Not a valid input format!")
        exit(0)
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    business_file_path = os.path.join(folder_path, 'business.json')
    user_file_path = os.path.join(folder_path, 'user.json')
    train_file_name = os.path.join(folder_path, 'yelp_train.csv')

    sc = SparkContext(appName='dsci553Zhenqinhw3')
    sc.setLogLevel("ERROR")

    # user_id (review_count, yelp-since, avg_star)
    user_features = sc.textFile(user_file_path).map(lambda line: json.loads(line))\
        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars']))).collectAsMap()

    userf_len = len(user_features.keys())
    userf_num = 2
    sum_tmp = [0.0 for i in range(userf_num)]
    for k, v in user_features.items():
        for i in range(userf_num):
            sum_tmp[i] += v[i]
    user_features_avg = tuple([x/userf_len for x in sum_tmp])

    # business_id,(stars, review_count)
    business_features = sc.textFile(business_file_path).map(lambda line: json.loads(line))\
        .map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()
    bizf_len = len(business_features.keys())
    bizf_num = 2
    sum_tmp = [0.0 for i in range(bizf_num)]
    for k, v in business_features.items():
        for i in range(bizf_num):
            sum_tmp[i] += v[i]
    business_features_avg = tuple([x/bizf_len for x in sum_tmp])

    train_rdd = sc.textFile(train_file_name).filter(lambda row: row != 'user_id,business_id,stars')\
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1], float(row[2])))

    x_train_rdd = train_rdd.map(lambda x: (user_features[x[0]], business_features[x[1]], x[2]))\
        .map(lambda x: (tuple(list(x[0]) + list(x[1])), x[2]))

    x_train = x_train_rdd.keys().collect()
    y_train = x_train_rdd.values().collect()

    model = xgb.XGBRegressor()
    model.fit(np.array(x_train), np.array(y_train))

    test_file_rdd = sc.textFile(test_file_name)
    header = test_file_rdd.first()

    test_rdd = test_file_rdd.filter(lambda x: x != header).map(lambda row: row.split(','))\
        .map(lambda row: (row[0], row[1]))

    # ========== DEV code start ========#
    # test_rdd = test_file_rdd.filter(lambda x: x != header).map(lambda row: row.split(','))\
    #     .map(lambda row: (row[0], row[1], float(row[2])))
    # from sklearn.metrics import mean_squared_error
    # import numpy as np
    # test_data_dev_rdd = test_rdd.map(lambda x: (
    #     user_features[x[0]] if x[0] in user_features else tuple(user_features_avg),
    #     business_features[x[1]] if x[1] in business_features else tuple(business_features_avg),
    #     x[2])).map(lambda x: (tuple(list(x[0]) + list(x[1])), x[2]))
    #
    # x_test = test_data_dev_rdd.keys().collect()
    # y_true = test_data_dev_rdd.values().collect()
    # def rmse(y_true, y_pred):
    #     return np.sqrt(mean_squared_error(y_true, y_pred))
    # y_pred = model.predict(np.array(x_test))
    # print(rmse(y_true, y_pred))
    # ========== DEV code start ========#


    test_feature_rdd = test_rdd.map(lambda x: (
        user_features[x[0]] if x[0] in user_features else tuple(user_features_avg),
        business_features[x[1]] if x[1] in business_features else tuple(business_features_avg),
        x[0], x[1]
    )).map(lambda x: (tuple(list(x[0]) + list(x[1])), (x[2], x[3])))

    x_test = test_feature_rdd.keys().collect()
    u_b_list = test_feature_rdd.values().collect()
    y_pred = model.predict(np.array(x_test))

    with open(output_file_name, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(len(u_b_list)):
            uid, bid = u_b_list[i]
            y = y_pred[i]
            if i < len(u_b_list) - 1:
                f.write(uid + ',' + bid + ',' + str(y) + '\n')
            else:
                f.write(uid + ',' + bid + ',' + str(y))

    print('Duration:', time.time() - start)
    from rmse import task2_2
    task2_2()