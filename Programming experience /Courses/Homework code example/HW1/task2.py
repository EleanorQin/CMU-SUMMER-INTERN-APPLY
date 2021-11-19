import sys, json, time
from pyspark import SparkContext
from operator import add

def customized_partition_func(key):
    return ord(key[0])

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Not a valid input format!")
        exit(0)
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition = int(sys.argv[3])

    sc = SparkContext(appName='dsci553Zhenqinhw1')
    sc.setLogLevel('ERROR')
    review_rdd = sc.textFile(review_filepath).map(lambda row: json.loads(row)).map(lambda row: (row['business_id'], 1))

    result = {}

    for mode in ['default', 'customized']:
        start_time = time.time()

        if mode == 'default':
            business_rdd = review_rdd.partitionBy(n_partition)
        else:
            business_rdd = review_rdd.partitionBy(n_partition, customized_partition_func)

        result_rdd = business_rdd.reduceByKey(add).takeOrdered(10, lambda x: (-x[1], x[0]))

        result[mode] = {
            'n_partition': business_rdd.getNumPartitions(),
            'n_items': business_rdd.glom().map(len).collect(),
            'exe_time': time.time() - start_time
        }

    with open(output_filepath, 'w') as f:
        json.dump(result, f)