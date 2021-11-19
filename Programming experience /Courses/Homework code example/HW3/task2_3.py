import xgboost as xgb
import os
import sys, time, json
from pyspark import SparkContext
from math import pow, sqrt
import joblib
import numpy as np
import pandas as pd


def model_based(sc, test_file_name, business_file_path, user_file_path, train_file_name):
    # user_id (review_count, yelp-since, avg_star)
    user_features = sc.textFile(user_file_path).map(lambda line: json.loads(line)) \
        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars']))).collectAsMap()

    userf_len = len(user_features.keys())
    userf_num = 2
    sum_tmp = [0.0 for i in range(userf_num)]
    for k, v in user_features.items():
        for i in range(userf_num):
            sum_tmp[i] += v[i]
    user_features_avg = tuple([x / userf_len for x in sum_tmp])

    # business_id,(stars, review_count)
    business_features = sc.textFile(business_file_path).map(lambda line: json.loads(line)) \
        .map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()
    bizf_len = len(business_features.keys())
    bizf_num = 2
    sum_tmp = [0.0 for i in range(bizf_num)]
    for k, v in business_features.items():
        for i in range(bizf_num):
            sum_tmp[i] += v[i]
    business_features_avg = tuple([x / bizf_len for x in sum_tmp])

    train_rdd = sc.textFile(train_file_name).filter(lambda row: row != 'user_id,business_id,stars') \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1], float(row[2])))

    x_train_rdd = train_rdd.map(lambda x: (user_features[x[0]], business_features[x[1]], x[2])) \
        .map(lambda x: (tuple(list(x[0]) + list(x[1])), x[2]))

    x_train = x_train_rdd.keys().collect()
    y_train = x_train_rdd.values().collect()

    model = xgb.XGBRegressor()
    model.fit(np.array(x_train), np.array(y_train))

    test_file_rdd = sc.textFile(test_file_name)
    header = test_file_rdd.first()

    test_rdd = test_file_rdd.filter(lambda x: x != header).map(lambda row: row.split(',')) \
        .map(lambda row: (row[0], row[1]))

    test_feature_rdd = test_rdd.map(lambda x: (
        user_features[x[0]] if x[0] in user_features else tuple(user_features_avg),
        business_features[x[1]] if x[1] in business_features else tuple(business_features_avg),
        x[0], x[1]
    )).map(lambda x: (tuple(list(x[0]) + list(x[1])), (x[2], x[3])))

    x_test = test_feature_rdd.keys().collect()
    y_pred = model.predict(np.array(x_test))

    def get_cnt_avg(x_r_iter):
        _sum = 0
        _cnt = 0
        for x, r in x_r_iter:
            _sum += r
            _cnt += 1
        return _sum / _cnt, _cnt

    # uid: [(b,r) ...]
    user_features2 = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(get_cnt_avg) \
        .collectAsMap()
    # print(len(user_biz_rating_list))

    # bid: [(user,r) ...]
    business_features2 = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(get_cnt_avg) \
        .collectAsMap()

    features_rdd = test_rdd.map(lambda x: (
        user_features[x[0]] if x[0] in user_features else tuple(user_features_avg),
        business_features[x[1]] if x[1] in business_features else tuple(business_features_avg),
        user_features2[x[0]] if x[0] in user_features2 else tuple(user_features_avg),
        business_features2[x[1]] if x[1] in business_features2 else tuple(business_features_avg))) \
        .map(lambda x: tuple(list(x[0]) + list(x[1]) + list(x[2]) + list(x[3])))
    #
    features = features_rdd.collect()

    return np.array(features), y_pred


def item_based(sc, train_file_name, test_file_name):
    N = 10
    co_rater_size = 3
    thershold = 30
    overall_avg = 3.71

    user_business_stars_rdd = sc.textFile(train_file_name).filter(lambda row: row != 'user_id,business_id,stars') \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1], float(row[2])))

    user_rdd = user_business_stars_rdd.map(lambda x: x[0]).distinct().persist()
    uid_index = user_rdd.zipWithIndex().collectAsMap()
    index_uid = user_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # print(len(uid_index))

    biz_rdd = user_business_stars_rdd.map(lambda x: x[1]).distinct().persist()
    bid_index = biz_rdd.zipWithIndex().collectAsMap()
    index_bid = biz_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # print(bid_index)

    user_business_stars_rdd = user_business_stars_rdd.map(lambda x: (uid_index[x[0]], bid_index[x[1]], x[2]))

    # (bid, rating)
    business_avg_rating = user_business_stars_rdd.map(lambda x: (x[1], (x[0], x[2]))) \
        .aggregateByKey((0, 0), lambda x, y: (x[0] + y[1], x[1] + 1), lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .map(lambda x: (x[0], float(x[1][0] / x[1][1]))).collectAsMap()
    # print(len(business_avg_rating))

    # _sum = 0
    # for k,v in business_avg_rating.items():
    #     _sum += v
    # print("overall avg", _sum/len(business_avg_rating)) # 3.71

    # uid: [(b,r) ...]
    user_biz_rating_list = user_business_stars_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list) \
        .collectAsMap()
    # print(len(user_biz_rating_list))

    # bid: [(user,r) ...]
    biz_user_rating_list = user_business_stars_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list) \
        .collectAsMap()
    # print(len(biz_user_rating_list))

    # print("finish init", time.time() - start)

    # ========== Predict ========== #
    W = {}  # (bi, bj): sim

    def item_based_rec(test_uid, test_bid):
        if test_uid not in uid_index:
            return overall_avg
        test_uid = uid_index[test_uid]

        if test_bid not in bid_index:
            _ = user_biz_rating_list[test_uid]
            _sum = sum([br[1] for br in _])
            return _sum / len(_)
        test_bid = bid_index[test_bid]

        user_ratings_of_test_bid = biz_user_rating_list[test_bid]
        user_ratings_dict_of_test_bid = {_[0]: _[1] for _ in user_ratings_of_test_bid}

        topN_cand = {}
        for biz, rating_list in user_biz_rating_list[test_uid]:
            neighbors = biz_user_rating_list[biz]
            neighbors_rating_dict = {_[0]: _[1] for _ in neighbors}

            co_rater = set(user_ratings_dict_of_test_bid.keys()) & set(neighbors_rating_dict.keys())
            if len(co_rater) < co_rater_size:
                continue

            b_pair = tuple(sorted([biz, test_bid]))
            r_un = neighbors_rating_dict[test_uid]
            if b_pair not in W:
                numerator = 0.0
                denominator_left = 1.0
                denominator_right = 1.0

                # # all rating average
                # left_avg = business_avg_rating[test_bid]
                # right_avg = business_avg_rating[biz]

                # co-rated item average
                left_avg = sum([user_ratings_dict_of_test_bid[_u] for _u in co_rater]) / len(co_rater)
                right_avg = sum([neighbors_rating_dict[_u] for _u in co_rater]) / len(co_rater)

                for rater in co_rater:
                    item1 = user_ratings_dict_of_test_bid[rater] - left_avg
                    item2 = neighbors_rating_dict[rater] - right_avg
                    numerator += item1 * item2
                    denominator_left += (pow(item1, 2))
                    denominator_right += (pow(item2, 2))

                wij = numerator / (sqrt(denominator_left) * sqrt(denominator_right))
                W[b_pair] = wij
                topN_cand[b_pair] = (wij, r_un)
            else:
                topN_cand[b_pair] = (W[b_pair], r_un)
        # for k,v in topN_cand.items():
        #     print([index_bid[_] for _ in k], v)

        if len(topN_cand) > thershold:
            topN = sorted(topN_cand.items(), key=lambda x: x[1][0], reverse=True)[:N]
            numerator = 0.0
            denominator = 0.0
            for line in topN:
                wij, r_un = line[1]
                numerator += r_un * wij
                denominator += abs(wij)
            if denominator == 0:
                return business_avg_rating[test_bid]
            else:
                sim = numerator / denominator
                return sim
        elif len(topN_cand) > 0:
            _ = biz_user_rating_list[test_bid]
            _sum = sum([br[1] for br in _])
            return _sum / len(_)
        else:
            _ = biz_user_rating_list[test_bid]
            _sum = sum([br[1] for br in _])
            return _sum / len(_)

    test_file_rdd = sc.textFile(test_file_name)
    header = test_file_rdd.first()
    test_user_business_rdd = test_file_rdd.filter(lambda row: row != header) \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))

    result_rdd = test_user_business_rdd.map(lambda x: ((x[0], x[1]), item_based_rec(x[0], x[1])))

    u_b_list = result_rdd.keys().collect()
    y_pred = result_rdd.values().collect()

    return u_b_list, y_pred


def get_features():
    folder_path = 'data/'
    test_file_name = 'data/yelp_val_in.csv'
    business_file_path = os.path.join(folder_path, 'business.json')
    user_file_path = os.path.join(folder_path, 'user.json')
    train_file_name = os.path.join(folder_path, 'yelp_train.csv')

    sc = SparkContext(appName='dsci553Zhenqinhw3')
    sc.setLogLevel("ERROR")

    # user_id (review_count, yelp-since, avg_star)
    user_features = sc.textFile(user_file_path).map(lambda line: json.loads(line)) \
        .map(lambda x: (x['user_id'], (x['review_count'], x['average_stars']))).collectAsMap()

    userf_len = len(user_features.keys())
    userf_num = 2
    sum_tmp = [0.0 for i in range(userf_num)]
    for k, v in user_features.items():
        for i in range(userf_num):
            sum_tmp[i] += v[i]
    user_features_avg = tuple([x / userf_len for x in sum_tmp])

    # business_id,(stars, review_count)
    business_features = sc.textFile(business_file_path).map(lambda line: json.loads(line)) \
        .map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).collectAsMap()
    bizf_len = len(business_features.keys())
    bizf_num = 2
    sum_tmp = [0.0 for i in range(bizf_num)]
    for k, v in business_features.items():
        for i in range(bizf_num):
            sum_tmp[i] += v[i]
    business_features_avg = tuple([x / bizf_len for x in sum_tmp])

    train_u_b_r_rdd = sc.textFile(train_file_name).filter(lambda row: row != 'user_id,business_id,stars') \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1], float(row[2])))

    test_file_rdd = sc.textFile(test_file_name)
    header = test_file_rdd.first()
    test_user_business_rdd = test_file_rdd.filter(lambda row: row != header) \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))

    def get_cnt_avg(x_r_iter):
        _sum = 0
        _cnt = 0
        for x, r in x_r_iter:
            _sum += r
            _cnt += 1
        return _sum / _cnt, _cnt

    # uid: [(b,r) ...]
    user_features2 = train_u_b_r_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(get_cnt_avg) \
        .collectAsMap()
    # print(len(user_biz_rating_list))

    # bid: [(user,r) ...]
    business_features2 = train_u_b_r_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(get_cnt_avg) \
        .collectAsMap()

    # x_train_rdd = test_u_b_r_rdd.map(lambda x: (user_features[x[0]], business_features[x[1]], x[2]))\
    #     .map(lambda x: tuple(list(x[0]) + list(x[1])))

    x_train_rdd = test_user_business_rdd.map(lambda x: (
        user_features[x[0]] if x[0] in user_features else tuple(user_features_avg),
        business_features[x[1]] if x[1] in business_features else tuple(business_features_avg),
        user_features2[x[0]] if x[0] in user_features2 else tuple(user_features_avg),
        business_features2[x[1]] if x[1] in business_features2 else tuple(business_features_avg))) \
        .map(lambda x: tuple(list(x[0]) + list(x[1]) + list(x[2]) + list(x[3])))
    #
    x_train = x_train_rdd.collect()
    return np.array(x_train)


def result_2_labels(item_base_y, model_base_y, train_df):
    train_df['prediction_item'] = item_base_y
    train_df['prediction_model'] = model_base_y
    train_df.stars.astype(np.float64)

    # item:0 ,model: 1
    def label_func(x):
        return 0 if abs(x['prediction_item'] - x['stars']) < abs(x['prediction_model'] - x['stars']) else 1

    train_df.set_index(['user_id', 'business_id'])
    train_df['label'] = train_df.apply(label_func, axis=1)

    return train_df['label']


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
    train_test_file_name = os.path.join(folder_path, 'yelp_val.csv')

    sc = SparkContext(appName='dsci553Zhenqinhw3')
    sc.setLogLevel("ERROR")

    _, item_base_y = item_based(sc, train_file_name, train_test_file_name)

    features, model_base_y = model_based(sc, train_test_file_name, business_file_path, user_file_path, train_file_name)

    # ==============DEV===================#
    # with open('class_data', 'wb') as f:
    #     joblib.dump({
    #         'u_b_list': u_b_list,
    #         'item_base_y': item_base_y,
    #         'model_base_y': model_base_y,
    #         'features': features
    #     }, f)

    # with open('class_data', 'rb') as f:
    #    data = joblib.load(f)
    #    u_b_list = data['u_b_list']
    #    item_base_y = data['item_base_y']
    #    model_base_y = data['model_base_y']
    #    features = data['features']
    # ==============DEV===================#

    train_df = pd.read_csv(train_test_file_name)
    train_y = result_2_labels(item_base_y, model_base_y, train_df)

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier()
    clf.fit(features, train_y)

    # cannot use model file since the different version of sklearn
    # print(clf.score(features, train_y))
    # joblib.dump(clf, 'model.joblib')
    # clf = joblib.load('model.joblib')

    u_b_list, item_base_y = item_based(sc, train_file_name, test_file_name)

    features, model_base_y = model_based(sc, test_file_name, business_file_path, user_file_path, train_file_name)

    clf.predict(features)
    # 0, 1, 1, 1
    class_labels = clf.predict(features)

    with open(output_file_name, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(len(u_b_list)):
            uid, bid = u_b_list[i]
            y = item_base_y[i] if class_labels[i] == 0 else model_base_y[i]
            if i < len(u_b_list) - 1:
                f.write(uid + ',' + bid + ',' + str(y) + '\n')
            else:
                f.write(uid + ',' + bid + ',' + str(y))

    print('Duration:', time.time() - start)