import sys, time, csv, json
from pyspark import SparkContext, SparkConf
import random, math
from collections import defaultdict
from operator import add
from itertools import combinations

num_of_hash_functions = 200
num_of_band = 100
num_of_rows = int(num_of_hash_functions / num_of_band)
m = 100000


def Jaccard_sim(b1_set, b2_set):
    return len(b1_set & b2_set) / len(b1_set | b2_set)


def set_add(x, v):
    x.add(v)
    return x


def reduce2list(l1, l2):
    return l1 + l2


if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 3:
        print("Not a valid input format!")
        exit(0)
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    sc = SparkContext(appName='dsci553Zhenqinhw3')
    sc.setLogLevel("ERROR")
    # csv line => business_id_1,business_id_2,similarity
    user_business_rdd = sc.textFile(input_file_path).filter(lambda row: row != 'user_id,business_id,stars')\
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))
    user_rdd = user_business_rdd.map(lambda x: x[0]).distinct().persist()
    user_index = user_rdd.zipWithIndex().collectAsMap()

    # biz_rdd = user_business_rdd.map(lambda x: x[1]).distinct().persist()
    # biz_index = user_business_rdd.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    # # index_biz = biz_rdd.collect()

    total_user_num = len(user_index.keys())

    # hash_function_id: [u1, u2, ..., un]
    hash_user_values = []
    for i in range(num_of_hash_functions):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        p = 49157
        hash_user_values.append([((a * x + b) % p) % total_user_num for x in range(total_user_num)])
    # print(hash_user_values)

    # min-hash
    biz_users_set_rdd = user_business_rdd.map(lambda x: (x[1], x[0])).combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)).persist()
    # biz_index_users_set = biz_users_set_rdd.map(lambda x: (x[0], x[1])).collectAsMap()
    signature_matrix = biz_users_set_rdd.flatMap(lambda x:
        ((x[0], min([hash_values[user_index[u]] for u in x[1]])) for hash_values in hash_user_values))\
        .groupByKey().mapValues(list).collect()

    # print(signature_matrix)

    # LSH


    print('Duration:', time.time() - start)
    # print(candidates)
    # with open(output_file_path,'w') as f:
    #     for pairs in candidates:
    #         biz_index_1, biz_index_2 = pairs
    #         b1_user_set = biz_index_users_set[biz_index_1]
    #         b2_user_set = biz_index_users_set[biz_index_2]
    #         sim = Jaccard_sim(b1_user_set, b2_user_set)
    #         if sim >= 0.5:
    #             f.write(json.dumps({
    #                 'business_id_1': index_biz[biz_index_1],
    #                 'business_id_2': index_biz[biz_index_2],
    #                 'similarity': sim
    #             }) + '\n')
