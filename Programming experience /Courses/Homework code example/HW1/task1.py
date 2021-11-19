import sys, json
from pyspark import SparkContext
from operator import add

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Not a valid input format!")
        exit(0)
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    sc = SparkContext(appName='dsci553Zhenqinhw1')
    review_rdd = sc.textFile(review_filepath).map(lambda row: json.loads(row))

    result = {}

    result['n_review'] = review_rdd.map(lambda row: row['review_id']).count()

    result['n_review_2018'] = review_rdd.map(lambda row: (row['review_id'], row['date']))\
        .filter(lambda x: x[1][:4] == "2018").count()

    result['n_user'] = review_rdd.map(lambda row: row['user_id']).distinct().count()

    result['top10_user'] = review_rdd.map(lambda row:(row['user_id'], 1)).reduceByKey(add)\
        .takeOrdered(10, lambda x: (-x[1], x[0]))
    result['n_business'] = review_rdd.map(lambda row: row['business_id']).distinct().count()

    result['top10_business'] = review_rdd.map(lambda row:(row['business_id'], 1)).reduceByKey(add)\
        .takeOrdered(10, lambda x: (-x[1], x[0]))

    with open(output_filepath, 'w') as f:
        json.dump(result, f)

