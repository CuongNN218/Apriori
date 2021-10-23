import copy
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from collections import OrderedDict


def calculate_confidence(freq_set, consequence, all_freq):
    remain_set = set(freq_set) - set(consequence)
    if len(remain_set) == 1:
        b = all_freq[tuple(remain_set)]
    else:
        sorted_remain_set = sorted(list(remain_set))
        b = all_freq[tuple(sorted_remain_set)]
    conf = all_freq[tuple(freq_set)] / b
    return conf, remain_set


def generate_itemsets(previous_itemsets, k):
    itemsets = []
    for a in previous_itemsets:
        for b in previous_itemsets:
            if b == a:
                continue
            merged = []
            if k == 1 and a[k - 1] < b[k - 1]:
                merged = a + [b[k - 1]]
                if sorted(merged) in itemsets:
                    continue
            elif k >= 1:
                if a[:k-1] == b[:k-1] and a[k-1] < b[k-1]:
                    merged = a + [b[k-1]]
                    if sorted(merged) in itemsets:
                        continue
            if len(merged) > 0:
                check = True
                for element in merged:
                    sub_merge = copy.copy(merged)
                    sub_merge.remove(element)
                    if sub_merge not in previous_itemsets:
                        check = False
                if check:
                    itemsets.append(sorted(merged))
    return itemsets


def get_subset(candidates, transaction, freq_dict):
    sub_candidates = []
    for rule in candidates:
        if set(rule) <= set(transaction):
            sub_candidates.append(candidates)
            freq_dict[tuple(rule)] = freq_dict.get(tuple(rule), 0) + 1
    return sub_candidates


def prune_itemsets(freq_dict, min_support, all_freq):
    pruned_itemset = set()
    for k, v in freq_dict.items():
        if v >= min_support:
            pruned_itemset.add(k)
            all_freq[k] = v
    return list(map(list, pruned_itemset))


def generate_l1(items_freq, min_sup):
    # transaction should be dict, key: id, val: list items
    # generate 1- itemsets from input file
    items = []
    for k, v in items_freq.items():
        if v >= min_sup:
            items.append(k)
    items = sorted(items)
    l1 = [[x] for x in items]
    return l1


def preprocessing(input_file):
    # read input file return a dict of trans and dict of item's frequency.
    trans = {}
    items_freq = {}
    i = 0
    with open(input_file, 'r') as f:
        for line in f:
            tran_id, item_id = line.strip().split(' ')[:2]
            items_freq[int(item_id)] = items_freq.get(int(item_id), 0) + 1
            if int(tran_id) not in trans.keys():
                trans[int(tran_id)] = []
            trans[int(tran_id)].append(int(item_id))
    return trans, items_freq


def apiori(trans, items_freq, min_sup):
    start = time.time()
    l1 = generate_l1(items_freq, min_sup)
    previous_itemset = l1
    frequent_itemset = [l1]
    print("Num of 1 freq itemset: ", len(l1))
    k = 1
    all_itemsets_freq_dict = OrderedDict()
    while len(previous_itemset) > 0:
        candidates = generate_itemsets(previous_itemset, k)
        subset_freq = {}
        for tran_id, val in trans.items():
            sub_candidates = get_subset(candidates, val, subset_freq)

        pruned_itemset = prune_itemsets(subset_freq, min_sup, all_itemsets_freq_dict)
        pruned_itemset = sorted(pruned_itemset)
        if len(pruned_itemset) > 0:
            frequent_itemset.append(pruned_itemset)
        previous_itemset = pruned_itemset
        print(f'Num of {k+1} freq set: {len(pruned_itemset)}')
        k += 1
    p_time = round(time.time() - start, 2)
    for k, v in items_freq.items():
        if [k] in l1:
            all_itemsets_freq_dict[tuple([k])] = v
    return frequent_itemset, all_itemsets_freq_dict, p_time


def generation_rules(all_freq_itemset, all_freq, min_conf):
    rules = []
    start = time.time()
    if min_conf == -1:
        for k_itemset in all_freq_itemset[1:]:
            for f_k in k_itemset:
                rules.append([sorted(f_k), [], all_freq[tuple(f_k)], -1])
        p_time = time.time() - start
        return rules, p_time

    for k_itemsets in all_freq_itemset[1:]:
        for f_k in k_itemsets:
            h_1 = [[x] for x in f_k]
            for x in f_k:
                conf, lhs = calculate_confidence(f_k, [x], all_freq)
                if conf >= min_conf:
                    rules.append([sorted(lhs), [x], all_freq[tuple(f_k)], round(conf, 3)])
            ap_genrules(f_k, h_1, min_conf, all_freq, rules)

    p_time = round(100 * (time.time() - start), 3)
    print(f'Nums of rules: {len(rules)}')
    return rules, p_time


def ap_genrules(f_k, h_prev, min_conf, all_freq, rules):
    if len(h_prev) == 0:
        return
    k = len(f_k)
    m = len(h_prev[0])
    if k > m + 1:
        h_next = generate_itemsets(h_prev, m)
        curr_set = set()
        remove_set = set()
        for itemset in h_next:
            curr_set.add(tuple(itemset))
            conf, lhs = calculate_confidence(f_k, itemset, all_freq)
            if conf >= min_conf:
                rules.append([sorted(lhs), itemset, all_freq[tuple(f_k)], round(conf, 3)])
            else:
                remove_set.add(tuple(itemset))
        h_remain = curr_set - remove_set
        h_remain = list(map(list, h_remain))
        ap_genrules(f_k, h_remain, min_conf, all_freq, rules)


def post_processing(generated_rules, output_file, min_conf):
    f = open(output_file, 'w')
    if min_conf > 0:
        for rule in generated_rules:
            if len(rule[0]) > 1:
                str_rule = map(str, rule[0])
                lhs = ','.join(str_rule)
            else:
                lhs = str(rule[0][0])
            if len(rule[1]) > 1:
                str_rule = map(str, rule[1])
                rhs = ','.join(str_rule)
            else:
                rhs = str(rule[1][0])
            line = '{' + lhs + '}|{' + rhs + '}|' + str(rule[2]) + f'|{rule[3]}\n'
            f.write(line)
    else:
        for rule in generated_rules:
            str_rule = map(str, rule[0])
            lhs = ','.join(str_rule)
            line = '{' + lhs + '}|{}|' + str(rule[2]) + '|-1\n'
            f.write(line)
    f.close()


def plot_bar_chart(val_1, val_2, plot_title, y_title, fig_name, args):
    n_groups = len(val_1)
    fig, ax_1 = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.45
    opacity = 0.8

    color_1 = 'tab:red'
    color_2 = 'tab:blue'
    rects1 = ax_1.bar(index,
                      val_1,
                      bar_width,
                      alpha=opacity,
                      color=color_1)
    ax_2 = ax_1.twinx()
    rects2 = ax_2.bar(index + bar_width,
                      val_2,
                      bar_width,
                      alpha=opacity,
                      color=color_2)

    categories = tuple(args.min_sup)

    def autolabel(rects, values, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.,
                    1.005 * height,
                    f'{val}',
                    ha='center',
                    va='bottom')

    autolabel(rects1, val_1, ax_1)
    autolabel(rects2, val_2, ax_2)
    ax_1.set_xlabel('Minimum Support Counting')
    if 'time' in y_title.lower():
        y_title_1 = 'time of frequent itemset generation (s)'
        y_title_2 = 'time of rule generation (ms)'
        ax_1.set_ylabel(y_title_1).set_color(color_1)
        ax_2.set_ylabel(y_title_2).set_color(color_2)
    else:
        ax_1.set_ylabel(y_title + ' of frequent itemsets').set_color(color_1)
        ax_2.set_ylabel(y_title + ' of rules').set_color(color_2)
    ax_1.set_ylim([0, max(val_1) * 1.1])
    ax_2.set_ylim([0, max(val_2) * 1.2])
    plt.title(plot_title)
    plt.xticks(index + bar_width / 2, categories)
    plt.legend([rects1, rects2], ['Frequent Itemset', 'Rule'])
    plt.tight_layout()
    plt.savefig(fig_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_sup', type=int, nargs='+', default=2, help="Min support count")
    parser.add_argument('--min_conf', type=float, default=0.5, help="Min confident")
    parser.add_argument('--input_file', type=str, help="Name to input file")
    parser.add_argument('--output_file', type=str, default="text.txt", help="Name of output file")
    args = parser.parse_args()

    trans, items_freq = preprocessing(args.input_file)

    os.makedirs(args.output_file, exist_ok=True)

    count_freq_set_list = []
    time_freq_list = []
    count_rule_list = []
    time_rule_list = []
    for min_sup in args.min_sup:
        print("Generating rules for: ", min_sup)
        freq_itemset, freq_itemset_count, p_time_itemset = apiori(trans, items_freq, min_sup)
        count_freq_set = 0
        for level in freq_itemset:
            count_freq_set += len(level)
        count_freq_set_list.append(count_freq_set)
        time_freq_list.append(p_time_itemset)

        rules, p_time_rules = generation_rules(freq_itemset, freq_itemset_count, args.min_conf)
        count_rules = len(rules)
        count_rule_list.append(count_rules)
        time_rule_list.append(p_time_rules)

        post_processing(rules,
                        os.path.join(args.output_file, f'rule_ms_{min_sup}_mc_{args.min_conf}.txt'),
                        args.min_conf)
    # plot time need to generate frequent sets and rules
    plot_bar_chart(time_freq_list,
                   time_rule_list,
                   f'Processing time to generate rule and frequent set with min conf: {args.min_conf}',
                   "Time",
                   os.path.join(args.output_file, f'time_mc_{args.min_conf}.png'),
                   args)

    # plot number of frequent sets and rules

    plot_bar_chart(count_freq_set_list,
                   count_rule_list,
                   f'Number of frequent sets and rules with min conf: {args.min_conf}',
                   'The number',
                   os.path.join(args.output_file, f'Counting_mc_{args.min_conf}.png'),
                   args)





