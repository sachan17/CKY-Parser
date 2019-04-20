from nltk.corpus import treebank as tb
import re
import nltk
from tqdm import tqdm


# for manual cnf conversion
# def base_category(t):
#     if isinstance(t,nltk.tree.Tree):
#         m = re.match("^(-[^-]+-|[^-=]+)",t1.label())
#         if m == None:
#             print(t1.label())
#         return(m.group(1))
#     else:
#         return(t1)
#
# def rule(t):
#     parent = t.label()
#     children = []
#     for each in t:
#         if type(each) == nltk.tree.Tree:
#             children.append(each.label())
#         else:
#             children.append(each)
#     return parent, children
#
# def break_rule(p, c):
#     r = []
#     for i in range(len(c)-2):
#         new_p = '@' + p + '_' + c[i]
#         r.append([p, [c[i], new_p]])
#         p = new_p
#     r.append([p, c[i+1:]])
#     return r

def grammer(ts):
    productions = []
    for tree in ts:
        tree.collapse_unary(collapsePOS = False)
        tree.chomsky_normal_form(horzMarkov = 2)
        productions += tree.productions()
    return productions


def pcfg(data):
    productions = grammer(data)
    fd_parents = nltk.FreqDist()
    fd_cfg = nltk.FreqDist(productions)
    fd_parents_count = nltk.FreqDist()
    vocab = set()
    term_parents = set()

    for prod in fd_cfg:
        if prod.lhs() not in fd_parents_count:
            fd_parents_count[prod.lhs()] = 0
            fd_parents[prod.lhs()] = 0
        fd_parents_count[prod.lhs()] += 1
        fd_parents[prod.lhs()] += fd_cfg[prod]

        if type(prod.rhs()[0]) == str:
            vocab.add(prod.rhs()[0])
            term_parents.add(prod.lhs())

    # for r in tqdm(fd_cfg):
        # fd_cfg[r] = fd_cfg[r]/fd_parents[r.lhs()]

    return fd_cfg, fd_parents, vocab, term_parents, fd_parents_count


if __name__ == '__main__':
    files = tb.fileids()
    data = list(tb.parsed_sents(files))

    # 80:20 split
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    print('Data Loaded ::: {} : {}'.format(len(train_data), len(test_data)))
    grammar, non_terms, vocab, term_parents, fd_parents_count = pcfg(train_data)
    # print(len(list(non_terms))
