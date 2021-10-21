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
for key in keys_lib:
    if key not in dict_code.keys():
        print(f'{key}\tsupport: {dict_lib[key]}')
        # print(count)
        count += 1
    else:
        if dict_code[key] != dict_lib[key]:
            print(key, dict_code[key], dict_lib[key])
            count += 1
print(count)
