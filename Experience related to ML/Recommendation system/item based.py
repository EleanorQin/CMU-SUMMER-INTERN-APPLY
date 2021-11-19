import sys, time, csv, json
from pyspark import SparkContext, SparkConf, StorageLevel
import random
from math import pow, sqrt
from collections import defaultdict
from operator import add
from itertools import combinations

N = 10
co_rater_size = 0
overall_avg = 3.71

if __name__ == '__main__':
    start = time.time()

    if len(sys.argv) != 4:
        print("Not a valid input format!")
        exit(0)
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    conf = SparkConf().setMaster("local").setAppName("dsci553Zhenqinhw3")\
        .set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # ========== Train ========== #
    # (uid, bid, rating)
    user_business_stars_rdd = sc.textFile(train_file_name).filter(lambda row: row != 'user_id,business_id,stars')\
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1], float(row[2])))

    user_rdd = user_business_stars_rdd.map(lambda x: x[0]).distinct().persist()
    uid_index = user_rdd.zipWithIndex().collectAsMap()
    index_uid = user_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # print(uid_index)

    biz_rdd = user_business_stars_rdd.map(lambda x: x[1]).distinct().persist()
    bid_index = biz_rdd.zipWithIndex().collectAsMap()
    index_bid = biz_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).collectAsMap()
    # print(bid_index)

    user_business_stars_rdd = user_business_stars_rdd.map(lambda x: (uid_index[x[0]], bid_index[x[1]], x[2]))

    # (bid, rating)
    business_avg_rating = user_business_stars_rdd.map(lambda x: (x[1], (x[0], x[2])))\
        .aggregateByKey((0, 0), lambda x, y: (x[0] + y[1], x[1] + 1), lambda x, y: (x[0] + y[0], x[1] + y[1]))\
        .map(lambda x: (x[0], float(x[1][0]/x[1][1]))).collectAsMap()
    print(len(business_avg_rating))

    # _sum = 0
    # for k,v in business_avg_rating.items():
    #     _sum += v
    # print("overall avg", _sum/len(business_avg_rating)) # 3.71

    # uid: [(b,r) ...]
    user_biz_rating_list = user_business_stars_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list)\
        .collectAsMap()
    print(len(user_biz_rating_list))

    # bid: [(user,r) ...]
    biz_user_rating_list = user_business_stars_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(list)\
        .collectAsMap()
    print(len(biz_user_rating_list))

    print("finish init", time.time() - start)

    # ========== Predict ========== #
    W = {} # (bi, bj): sim

    def item_based_rec(test_uid, test_bid):
        if test_bid not in bid_index:
            return overall_avg

        if test_uid not in uid_index:
            return business_avg_rating[test_bid] if bid in business_avg_rating else overall_avg

        test_uid = uid_index[test_uid]
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
                for rater in co_rater:
                    item1 = user_ratings_dict_of_test_bid[rater] - business_avg_rating[test_bid]
                    item2 = neighbors_rating_dict[rater] - business_avg_rating[biz]
                    numerator += item1 * item2
                    denominator_left += (pow(item1, 2))
                    denominator_right += (pow(item2, 2))

                wij = numerator/(sqrt(denominator_left) * sqrt(denominator_right))
                W[b_pair] = wij
                topN_cand[b_pair] = (wij, r_un)
            else:
                topN_cand[b_pair] = (W[b_pair], r_un)
        # for k,v in topN_cand.items():
        #     print([index_bid[_] for _ in k], v)

        if len(topN_cand) > 0:
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
                return numerator/denominator
        else:
            return business_avg_rating[test_bid]


    test_file_rdd = sc.textFile(test_file_name)
    header = test_file_rdd.first()
    test_user_business_rdd = test_file_rdd.filter(lambda row: row != header)\
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))

    result = test_user_business_rdd.map(lambda x: ((x[0], x[1]), item_based_rec(x[0], x[1]))).collect()

    with open(output_file_name, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for i in range(len(result)):
            uid, bid = result[i][0]
            y = result[i][1]
            if i < len(result) - 1:
                f.write(uid + ',' + bid + ',' + str(y) + '\n')
            else:
                f.write(uid + ',' + bid + ',' + str(y))
    #
    print('Duration:', time.time() - start)

    from rmse import task2_1
    # task2_1('example_test.csv')
    task2_1('data/yelp_val.csv')