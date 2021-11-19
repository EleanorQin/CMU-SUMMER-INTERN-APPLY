import sys, time
from operator import add
from itertools import combinations
from pyspark import SparkContext


def set_add(x,v):
    x.add(v)
    return x


def merge_tuple2set(comb):
    result = set()
    for item in comb:
        result = result.union(item)
    return result


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
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    sc = SparkContext(appName='dsci553Zhenqinhw2')
    sc.setLogLevel("ERROR")
    # csv line => user_id, business_id(case1) / bid, uid (case2)
    text_rdd = sc.textFile(input_file_path).filter(lambda row: row != 'user_id,business_id').map(lambda row: row.split(','))

    # uid, bid -> uid: {bid, bid, bid, ...}
    if case_number == 1:
        pass
    elif case_number == 2:
        text_rdd = text_rdd.map(lambda row: (row[1], row[0]))
    else:
        print("Unsupport case number: %d " % case_number)
        exit(0)

    # uid, bid -> uid: {bid, bid, bid, ...}
    basket_rdd = text_rdd.combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)
    ).persist()

    basket_num = basket_rdd.count()
    # SON Phase 1
    candidates = basket_rdd.map(lambda x: x[1]).mapPartitions(lambda iterator: A_Priori(iterator, support, basket_num))\
        .distinct().collect()
    # print(candidates)

    # SON Phase 2
    freq_item_rdd = basket_rdd.map(lambda x: x[1]).mapPartitions(lambda iterator: count_occurrences(iterator, candidates))\
        .reduceByKey(add).filter(lambda x: x[1] >= support).keys().collect()
    # print(freq_item_rdd)

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