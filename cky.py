from nltk.corpus import treebank as tb
from nltk.tree import Tree
import pcfg
from tqdm import tqdm
from nltk.grammar import Production, Nonterminal
from copy import copy
import nltk
from nltk import casual_tokenize
import sys

SMOOTHING_PARAMETER = 0.5

def cky_parsing(words, grammar, non_terms, vocab, term_parents, fd_parents_count):
    # adding unknown productions
    print('Parsing')
    for w in words:
        if w not in vocab:
            for tp in term_parents:
                grammar[Production(tp, [w])] = 0

    # smoothing
    for prod in grammar:
        grammar[prod] = (grammar[prod] + SMOOTHING_PARAMETER) / (non_terms[prod.lhs()] + fd_parents_count[prod.lhs()] + 1)

    # words = tb.parsed_sents('wsj_0001.mrg')[0].leaves()
    score = {i:{j+i+1:{x:0 for x in non_terms} for j in range(len(words)-i)} for i in range(len(words))}
    back = {i:{j+i+1:{x:None for x in non_terms} for j in range(len(words)-i)} for i in range(len(words))}

    for i in range(len(words)):
        for A in non_terms:
            prod = Production(A ,[words[i]])
            if prod in grammar:
                score[i][i+1][A] = grammar[prod]
                back[i][i+1][A] = [words[i]]
        added = True
        while added:
            added = False
            for g in grammar.keys():
                A, B = g.lhs(), g.rhs()
                if len(B) > 1 or B[0] not in non_terms:
                    continue
                B = B[0]
                if score[i][i+1][B] > 0:
                    prob = grammar[g] * score[i][i+1][B]
                    if prob > score[i][i+1][A]:
                        score[i][i+1][A] = prob
                        back[i][i+1][A] = [B]
                        added = True

    for span in tqdm(range(2, len(words)+1)):
        for begin in range(0, len(words)-span+1):
            end = begin + span
            for split in range(begin+1, end):
                for g in grammar.keys():
                    A, B = g.lhs(), g.rhs()
                    if len(B) == 1:
                        continue
                    B, C = B
                    prob = score[begin][split][B] * score[split][end][C] * grammar[g]
                    if prob > score[begin][end][A]:
                        score[begin][end][A] = prob
                        back[begin][end][A] = [split, B, C]
            added = True
            while added:
                added = False
                for g in grammar.keys():
                    A, B = g.lhs(), g.rhs()
                    if len(B) > 1 or B[0] not in non_terms:
                        continue
                    B = B[0]
                    prob = grammar[g] * score[begin][end][B]
                    if prob > score[begin][end][A]:
                        score[begin][end][A] = prob
                        back[begin][end][A] = [B]
                        added = True
    return score, back


def build_tree(tree, begin, end, back, non_terms):
    if tree.label() not in non_terms:
        return tree.label()
    children = back[begin][end][tree.label()]
    if len(children) == 3:
        split, left, right = children
        left_sub_tree = build_tree(Tree(left, []), begin, split, back, non_terms)
        right_sub_tree = build_tree(Tree(right, []), split, end, back, non_terms)
        tree.append(left_sub_tree)
        tree.append(right_sub_tree)
    else:
        direct = build_tree(Tree(children[0], []), begin, end, back, non_terms)
        tree.append(direct)
    return tree

def clean_tree(t):
    if type(t) == str:
        return
    t.set_label(str(t.label()))
    for each in t:
        clean_tree(each)

def brackets(t, br, words):
    if type(t) != nltk.tree.Tree:
        return
    br.append((str(t.label()), words.index(t.leaves()[0]), words.index(t.leaves()[-1])))
    for each in t:
        brackets(each, br, words)

def evaluate(words, predicted, actual):
    candidate = []
    gold = []
    brackets(predicted, candidate, words)
    brackets(actual, gold, words)
    correct = []
    precision, recall, f1_score = 0, 0, 0
    for each in candidate:
        if each in gold:
            correct.append(each)
    if len(candidate) > 0:
        precision = len(correct)/len(candidate)
    if len(gold) > 0:
        recall = len(correct)/len(gold)
    if (precision + recall) > 0:
        f1_score = (2*precision*recall)/(precision + recall)
    return precision, recall, f1_score

def get_start(score, l):
    max_p = 0
    root = None
    for each in score[0][l]:
        if score[0][l][each] > max_p:
            root = each
    return Tree(root, [])

def train():
    files = tb.fileids()
    data = list(tb.parsed_sents(files))

    # 80:20 split
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    P_grammar, P_non_terms, P_vocab, P_term_parents, P_parents_count = pcfg.pcfg(train_data)

    total_precision = 0
    toal_recall = 0
    total_f1_score = 0
    i = 0
    for test in test_data:
        print('Test', i)
        i+=1
        try:
            words = test.leaves()
            scores, backs = cky_parsing(words, copy(P_grammar), copy(P_non_terms), copy(P_vocab), copy(P_term_parents), copy(P_parents_count))
            start = Tree(Nonterminal('S'), [])
            if scores[0][len(words)][Nonterminal('S')] == 0:
                start = get_start(scores, len(words))
            predicted_tree = build_tree(start, 0, len(words), backs, P_non_terms)
            clean_tree(predicted_tree)
            predicted_tree.un_chomsky_normal_form()
            precision, recall, f1_score = evaluate(words, predicted_tree, test)
            print(precision, recall, f1_score)
            total_precision += precision
            toal_recall += recall
            total_f1_score += f1_score
        except:
            print('***************Failed', i-1)
            continue

    total_precision /= len(test_data)
    toal_recall /= len(test_data)
    total_f1_score /= len(test_data)

    print('Precision', total_precision)
    print('Recall', toal_recall)
    print('F1_score', total_f1_score)

def parse(sent):
    files = tb.fileids()
    data = list(tb.parsed_sents(files))

    P_grammar, P_non_terms, P_vocab, P_term_parents, P_parents_count = pcfg.pcfg(data)

    words = casual_tokenize(str(sent))
    scores, backs = cky_parsing(words, copy(P_grammar), copy(P_non_terms), copy(P_vocab), copy(P_term_parents), copy(P_parents_count))
    start = Tree(Nonterminal('S'), [])
    if scores[0][len(words)][Nonterminal('S')] == 0:
        start = get_start(scores, len(words))
    predicted_tree = build_tree(start, 0, len(words), backs, P_non_terms)
    clean_tree(predicted_tree)
    predicted_tree.un_chomsky_normal_form()
    print('Parsed Tree')
    print(predicted_tree)


if __name__ == '__main__':
    if sys.argv[1] == '-train':
        train()
    else:
        parse(' '.join(sys.argv[1:]))
