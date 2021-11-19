import os, time, sys
from pyspark import SparkContext, SparkConf
from itertools import combinations
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StringType
from typing import Set

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12")
from graphframes import *


def set_add(x, v):
    x.add(v)
    return x


if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 4:
        print("Not a valid input format!")
        exit(0)
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]

    sc = SparkContext(appName='dsci553Zhenqinhw4')
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)
    # user id,business id
    text_rdd = sc.textFile(input_file_path).filter(lambda row: row != 'user_id,business_id') \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))
    # (userid,(bid1,bid2,bid3)
    user_biz_set_rdd = text_rdd.combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)).collectAsMap()
    # print(len(user_biz_set_rdd))
    users = text_rdd.map(lambda x: x[0]).distinct().collect()
    # print(user_rdd.count())

    edge_user_pairs = set()
    for user_pair in combinations(users, 2):
        u1_set = user_biz_set_rdd[user_pair[0]]
        u2_set = user_biz_set_rdd[user_pair[1]]
        if len(u1_set & u2_set) >= filter_threshold:
            edge_user_pairs.add(tuple(sorted(user_pair)))
    # print(edge_user_pairs)

    node = set()
    for pairs in edge_user_pairs:
        node.add(pairs[0])
        node.add(pairs[1])
    # print(len(node))

    row = Row('id')

    vertices = sc.parallelize(list(node)).map(row).toDF()
    edges = sqlContext.createDataFrame(list(edge_user_pairs), ["src", "dst"])

    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    result_rdd = result.rdd.map(lambda x: (x[1], x[0])).combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)).map(lambda x: (sorted(x[1]))) \
        .sortBy(lambda x: (len(x), x[0])).collect()

    with open(community_output_file_path, 'w') as f:
        for line in result_rdd:
            f.write(str(line)[1:-1] + '\n')

    end = time.time()
    print("Duration:", end - start)