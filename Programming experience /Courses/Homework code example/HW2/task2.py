import sys
import time
import csv
from operator import add
from itertools import combinations
from pyspark import SparkContext


def set_add(x,v):
    x.add(v)
    return x


def preprocess(input_file_path):
    raw_data = []
    first = True
    with open(input_file_path, 'r',  encoding='utf-8-sig') as f:
        reader = csv.reader(f,  delimiter=',', quoting=csv.QUOTE_ALL)
        for row in reader:
            # raw_data.append([row[0], row[1], row[5]])
            if not first:
                raw_data.append([row[0] + '-' + row[1], str(int(row[5]))])

            first = False

    output_f = 'customer_product.csv'
    headers = ["DATE-CUSTOMER_ID", "PRODUCT_ID"]
    with open(output_f, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        writer.writerow(headers)
        for row in raw_data:
            writer.writerow(row)

    return output_f, headers


def A_Priori(basket_iterator, support, basket_num):
    baskets = list(basket_iterator)
    threshold = support * len(baskets) / basket_num
    result = []
    k = 1
    k1_count = {}
    for basket in baskets:
        for item in basket:
            k1_count[item] = k1_count.get(item, 0) + 1
    c1 = sorted([k for k, cnt in k1_count.items() if cnt >= threshold])
    ck = c1
    while len(ck) > 0:
        result += ck
        k += 1
        # bid: freq -> 1: bid -> [bid, bid] -> (bid, bid): freq
        k_count = {}
        if k == 2:
            for _c in combinations(ck, 2):
                for basket in baskets:
                    if set(_c).issubset(basket):
                        _ = tuple(sorted(_c))
                        k_count[_] = k_count.get(_, 0) + 1
        else:
            for i in range(len(ck) - 1):
                for j in range(i + 1, len(ck)):
                    if ck[i][0:(k - 2)] == ck[j][0:(k - 2)]:
                        comb = set(ck[i]).union(set(ck[j]))
                        for basket in baskets:
                            if comb.issubset(basket):
                                _ = tuple(sorted(comb))
                                k_count[_] = k_count.get(_, 0) + 1
        ck = sorted([k for k, cnt in k_count.items() if cnt >= threshold])

    return result


def count_occurrences(basket_iterator, candidates):
    candidates_count = {}
    for basket in basket_iterator:
        for cand in candidates:
            if (type(cand) != tuple and cand in basket) or set(cand).issubset(basket):
                candidates_count[cand] = candidates_count.get(cand, 0) + 1
    return candidates_count.items()


def group_collection(collection):
    grouped = {}
    for item in collection:
        if type(item) == str:
            grouped[1] = grouped.get(1, []) + [item]
        else:
            grouped[len(item)] = grouped.get(len(item), []) + [item]

    return [sorted(row[1]) for row in sorted(grouped.items(), key=lambda kv: kv[0])]


if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 5:
        print("Not a valid input format!")
        exit(0)
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext(appName='dsci553Zhenqinhw2')
    sc.setLogLevel("ERROR")

    customer_product_filepath, headers = preprocess(input_file_path)

    text_rdd = sc.textFile(customer_product_filepath).filter(lambda row: row != ",".join(headers)).map(lambda row: row.split(','))
    basket_rdd = text_rdd.combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)
    ).filter(lambda x: len(x[1]) > filter_threshold).persist()

    basket_num = basket_rdd.count()
    # SON Phase 1
    candidates = basket_rdd.map(lambda x: x[1]).mapPartitions(lambda iterator: A_Priori(iterator, support, basket_num))\
        .distinct().collect()
    # SON Phase 2
    freq_item_rdd = basket_rdd.map(lambda x: x[1]).mapPartitions(lambda iterator: count_occurrences(iterator, candidates))\
        .reduceByKey(add).filter(lambda x: x[1] >= support).keys().collect()

    result = {"Candidates": group_collection(candidates), "Frequent Itemsets": group_collection(freq_item_rdd)}

    total_len = 0
    result = {"Candidates": group_collection(candidates), "Frequent Itemsets": group_collection(freq_item_rdd)}
    # print(result)
    total_len = len(result['Candidates']) + len(result['Frequent Itemsets'])

    i = 0
    with open(output_file_path,'w') as f:
        for label in ['Candidates', 'Frequent Itemsets']:
            f.write('%s:\n' % label)
            for line in result[label]:
                line_str = ",".join(["('"+item+"')" if type(item) == str else str(item) for item in line])
                i += 1
                if i < total_len:
                    f.write("%s\n\n" % line_str)
                else:
                    f.write(line_str)

    print("Duration:", time.time() - start)