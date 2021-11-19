import sys
import binascii
import time
import math
from blackbox import BlackBox
import random

num_hash_functions = 10
m = 7687
p = 69997
hash_a_list = random.sample(range(1, sys.maxsize - 1), num_hash_functions)
hash_b_list = random.sample(range(1, sys.maxsize - 1), num_hash_functions)

# input is user_id, output is the hash number
def myhashs(s):
    result =[]
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    for i in range(num_hash_functions):
        hashed_x = (hash_a_list[i] * x + hash_b_list[i]) % m
        binary_x = "{0:b}".format(hashed_x)
        result.append(binary_x)
    return result

if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 5:
        print("Not a valid input format!")
        exit(0)
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    bx = BlackBox()
    fpr_str = ''
    r_list = []
    sum_est = 0
    sum_gt = 0
    for t in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        ground_truth = len(set(stream_users))
        sum_gt += ground_truth
        r_list = []
        for i in range(num_hash_functions):
            a = hash_a_list[i]
            b = hash_b_list[i]
            max_R = -math.inf
            for su in stream_users:
                x = int(binascii.hexlify(su.encode('utf8')), 16)
                hashed_x = ((a * x + b) % p) % m
                binary_x = "{0:b}".format(hashed_x)
                num_tailing_zero = 0
                if binary_x != '0':
                    num_tailing_zero = len(binary_x) - len(binary_x.rstrip("0"))
                if num_tailing_zero > max_R:
                    max_R = num_tailing_zero
            r_list.append(max_R)

        r_list = [2**_ for _ in r_list]

        estimation = round(sum(r_list)/len(r_list))
        sum_est += estimation

        fpr_str += "%d,%d,%d\n" % (t, ground_truth, estimation)
        print(t, ground_truth, estimation)

    print(sum_est/sum_gt)
    with open(output_filename, 'w') as f:
        f.write('Time,Ground Truth,Estimation\n')
        f.write(fpr_str)

    print('Duration:', time.time() - start)