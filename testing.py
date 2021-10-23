# read the csv from internet

dict_code = {}
with open('test_code_2.txt', 'r') as f:
    for line in f:
        items, count = line.strip().split('\t')[:2]
        item = tuple(items.split(','))
        dict_code[item] = count
dict_lib = {}
with open('test_lib-4.txt', 'r') as f:
    for line in f:
        items, count = line.strip().split('\t')[:2]
        item = tuple(items.split(','))
        dict_lib[item] = count
count = 0
keys_lib = dict_lib.keys()
with open('record_counting.txt', 'w') as f_count:
    for key in keys_lib:
        if key not in dict_code.keys():
            print(f'{key}\tsupport: {dict_lib[key]}')
            # print(count)
            count += 1
        else:
            if dict_code[key] != dict_lib[key]:
                line = f'frequent set: {key} - Support:  code= {dict_code[key]}, lib = {dict_lib[key]}\n'
                f_count.write(line)
                print(key, dict_code[key], dict_lib[key])
                count += 1
print("Count miss match frequent set: ", count)
exit()
# test number of rules
rules_lib = {}
with open('test_rules_lib.txt') as lib:
    for line in lib:
        lhs, rhs, _, conf = line.strip().split('|')
        lhs = lhs.split(',')
        # print(lhs)
        lhs = sorted(list(map(int, lhs)))
        rhs = rhs.split(',')
        # print(rhs)
        rhs = sorted(list(map(int, rhs)))
        key = [tuple(lhs), tuple(rhs)]
        key = tuple(key)
        rules_lib[key] = round(float(conf), 2)
        # print(rules_lib)

rules_code = {}
with open('results/rule_ms_50_mc_0.8') as code:
    for line in code:
        lhs, rhs, _, conf = line.strip().split('|')
        lhs = lhs.split(',')
        lhs = sorted(list(map(int, lhs)))
        rhs = rhs.split(',')
        rhs = sorted(list(map(int, rhs)))
        key = [tuple(lhs), tuple(rhs)]
        key = tuple(key)
        rules_code[key] = round(float(conf), 2)
        # print(rules_code)
        # exit()
# print(rules_code)
count = 0
for key, val in rules_code.items():
    # print(key, val)
    if key not in rules_lib.keys():
        print(f'Missing rules: {key} conf: {rules_code[key]}')
        count += 1
print("Count miss match freq rules:", count)


