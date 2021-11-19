import sys
import binascii
import time
from blackbox import BlackBox
import random

num_hash_functions = 5
m = 69997
p = 1610612741
hash_a_list = random.sample(range(1, 1000000000000), num_hash_functions)
hash_b_list = random.sample(range(1, 1000000000000), num_hash_functions)

# input is user_id, output is the hash number
def myhashs(s):
    result =[]
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    for i in range(num_hash_functions):
        r = ((hash_a_list[i] * x + hash_b_list[i]) % p) % m
        result.append(r)
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
    fpr_str = ""
    bit_array = [0]*m
    previous_user_set = set()
    for t in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        fp = 0
        tn = 0

        for su in stream_users:
            hash_user = myhashs(su)
            seen = 0
            for hu in hash_user:
                if bit_array[hu] == 1:
                    seen += 1
                else:
                    bit_array[hu] = 1
            if seen == num_hash_functions:
                if su not in previous_user_set:
                    fp += 1
            else:
                tn += 1

            previous_user_set.add(su)

        fpr_str += "%d,%f\n" % (t, fp/(fp + tn))
        print(t, fp/(fp+tn))

    with open(output_filename, 'w') as f:
        f.write('Time,FPR\n')
        f.write(fpr_str)

    print('Duration:', time.time() - start)










