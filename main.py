from nltk.corpus import brown
from sklearn.model_selection import train_test_split
import numpy as np
import re

tag_start = "START"
tag_unknown_words = "UN-KN"
tag_stop = "STOP"


def load_data():
    tagged_sentences = brown.tagged_sents(categories='news')
    # brown_news_words = brown.tagged_words(categories='news')
    # X = [p[0] for p in brown_news_words]
    # y = [p[1] for p in brown_news_words]
    X, y = [], []
    for sent in tagged_sentences:
        new_sample, new_tag = [], []
        for item in sent:
            tag = item[1]
            # take only prefix of complex tags
            if ("-" in item[1] or "+" in item[1] or "*" in item[1] or "$" in item[1]) and len(item[1]) >= 2:
                tag = re.split("\+|-", item[1])[0]
            new_sample.append(item[0])
            new_tag.append(tag)
        X.append(new_sample)
        y.append(new_tag)
    return train_test_split(np.array(X), np.array(y), test_size=0.1, shuffle=False)


def get_all_words_and_tags(X_train, y_train, X_test, y_test):
    words_train = [word for sample in X_train for word in sample]
    tags_train = [tag for tags in y_train for tag in tags]
    words_test = [word for sample in X_test for word in sample]
    tags_test = [tag for tags in y_test for tag in tags]
    return words_train, tags_train, words_test, tags_test


def question_b(words_train, tags_train, words_test, tags_test):
    # Part i
    cache = {word: {} for word in words_train}
    for w, t in zip(words_train, tags_train):
        cache_word = cache[w]
        cache_word[t] = cache_word.get(t, 0) + 1
    # most likely tag for each word in the training set
    most_likely_tags = {w: max(cache[w], key=cache[w].get) for w in words_train}

    # Part ii
    words_tagged_correctly_known = 0
    words_tagged_correctly_unknown = 0
    for w, t in zip(words_test, tags_test):
        if w in most_likely_tags and most_likely_tags[w] == t:
            words_tagged_correctly_known += 1
        elif w not in most_likely_tags:
            if 'NN' == t:
                words_tagged_correctly_unknown += 1
    unknown_words, unknown_words_tags, known_words, known_words_tags = know_and_unknown_test(words_test, tags_test)
    return 1 - (words_tagged_correctly_known / len(known_words)), 1 - (
            words_tagged_correctly_unknown / len(unknown_words)), \
           1 - ((words_tagged_correctly_known + words_tagged_correctly_unknown) / len(words_test))


def compute_emission(words_train, words_test, tags_train, known_words, add_one=False):
    # train
    cache = {tag: {} for tag in tags_train}
    for w, t in zip(words_train, tags_train):
        cache_tag = cache[t]
        cache_tag[w] = cache_tag.get(w, 0) + 1

    tags = list(set(tags_train))
    emission = {}
    for w in words_test:
        for t in tags:
            if add_one:
                emission[(w, t)] = (cache[t].get(w, 0) + 1) / (
                            sum(cache[t].values()) + len(set(words_test + words_train)))
            else:
                if w not in known_words and t == 'NN':
                    emission[(w, t)] = 1
                else:
                    emission[(w, t)] = cache[t].get(w, 0) / sum(cache[t].values())
    return emission


def compute_transition(all_tags):
    all_tags = [tags + [tag_stop] for tags in all_tags]
    cache = {}
    for tags in all_tags:
        prev = tag_start
        for tag in tags:
            if prev not in cache:
                cache[prev] = {tag: 1}
            else:
                cache_tag = cache[prev]
                cache_tag[tag] = cache_tag.get(tag, 0) + 1
            prev = tag
    transition = {}
    for tags in all_tags:
        prev = tag_start
        for tag in tags:
            transition[(prev, tag)] = cache[prev][tag] / sum(cache[prev].values())
            prev = tag
    return transition


def calc_q(u, v, x_k, transition, emission):
    return transition.get((u, v), 0) * emission.get((x_k, v))


def viterbi_algorithm(sent, tags, emission, transition):
    # dp table for words * tags (including unknown tags)
    K, N = len(tags), len(sent)

    dp = [{v: [0, None] for v in tags} for i in range(N + 1)]

    # init base layer
    dp[0][tag_start] = [1, None]

    # algorithm
    for k in range(1, N + 1):
        for v in tags:
            max_u = max(list(dp[k - 1].keys()),
                        key=lambda u: dp[k - 1][u][0] * calc_q(u, v, sent[k - 1], transition, emission))
            dp_k_v = dp[k - 1][max_u][0] * calc_q(max_u, v, sent[k - 1], transition, emission)
            if dp_k_v > 0:
                dp[k][v][0] = dp_k_v
                dp[k][v][1] = max_u
            else:
                dp[k][v][1] = max(list(dp[k - 1].keys()), key=lambda u: dp[k - 1][u][0] * transition.get((u, v), 0))

    output = [None for i in range(N)]
    output[N - 1] = max(tags, key=lambda u: dp[N][u][0] * transition.get((u, tag_stop), 0))

    for k in range(max(N - 2, -1), -1, -1):
        output[k] = dp[k + 2][output[k + 1]][1]

    return output


def question_c(X_train, y_train, X_test, y_test,
               words_train, tags_train, words_test, tags_test, known_words, unknown_words):
    # part i: emission
    emission = compute_emission(words_train, words_test, tags_train, known_words)

    all_tags = list(set(tags_train))
    # part i: transition
    transition = compute_transition(y_train)

    # part iii
    sents_known_tagged_correctly = 0
    sents_unknown_tagged_correctly = 0

    for sample, tags in zip(X_test, y_test):
        output = viterbi_algorithm(sample, all_tags, emission, transition)
        for i, w in enumerate(sample):
            if w in known_words:
                if output[i] == tags[i]:
                    sents_known_tagged_correctly += 1
            else:
                if output[i] == tags[i]:
                    sents_unknown_tagged_correctly += 1
    return 1 - (sents_known_tagged_correctly / len(known_words)), 1 - (
            sents_unknown_tagged_correctly / len(unknown_words)), \
           1 - ((sents_known_tagged_correctly + sents_unknown_tagged_correctly) / len(words_test))


def question_d(X_train, y_train, X_test, y_test,
               words_train, tags_train, words_test, tags_test, known_words):
    emission = compute_emission(words_train, words_test, tags_train, known_words, True)
    transition = compute_transition(y_train)

    all_tags = list(set(tags_train))

    sents_known_tagged_correctly = 0
    sents_unknown_tagged_correctly = 0

    for sample, tags in zip(X_test, y_test):
        output = viterbi_algorithm(sample, all_tags, emission, transition)
        for i, w in enumerate(sample):
            if w in known_words:
                if output[i] == tags[i]:
                    sents_known_tagged_correctly += 1
            else:
                if output[i] == tags[i]:
                    sents_unknown_tagged_correctly += 1
    return 1 - (sents_known_tagged_correctly / len(known_words)), 1 - (
            sents_unknown_tagged_correctly / len(unknown_words)), \
           1 - ((sents_known_tagged_correctly + sents_unknown_tagged_correctly) / len(words_test))


def get_pseudoword(word):
    if re.match("(\d*\.?\d+|\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\S)", word):
        return 'NUMBER'
    elif len(word) >= 3 and word[0].isUpper() and word[-2] == '\'' and word[-1].isLower():
        return 'NAME_WITH_POSSESSION_S'
    elif re.match("\$(\d*\.?\d+|\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\S)", word):
        return 'SUM_OF_MONEY'
    elif word[0].isUpper():
        return 'initCap'
    elif len(word) >= 2 and word[-2:] == "ed":
        return 'ed_SUFFIX'
    elif len(word) >= 2 and word[-2:] == "ly":
        return 'ly_SUFFIX'
    elif len(word) >= 3 and word[-3:] == "ing":
        return 'ing_SUFFIX'
    elif len(word) >= 3 and word[-3:] == "est":
        return 'est_SUFFIX'

def question_e(words_train, tags_train, words_test, tags_test):
    # Part i
    pseudo_words, pseudo_words_tags, known_words, known_words_tags = know_and_unknown_test(words_test, tags_test)
    # Add words with low frequency
    freq = {}
    for w in words_train:
        freq[w] = freq.get(w, 0) + 1
    for word, tag in zip(words_train, tags_train):
        if freq[word] < 5:
            pseudo_words.append(word)
            pseudo_words_tags.append(tag)
    return pseudo_words


def know_and_unknown_test(words_test, tags_test):
    unknown_words, unknown_words_tags = [], []
    known_words, known_words_tags = [], []
    set_words_train = set(words_train)
    for word, tag in zip(words_test, tags_test):
        if word not in set_words_train:
            unknown_words.append(word)
            unknown_words_tags.append(tag)
        else:
            known_words.append(word)
            known_words_tags.append(tag)
    return unknown_words, unknown_words_tags, known_words, known_words_tags


if __name__ == '__main__':
    # For me: what's going on with the unknown tags?

    X_train, X_test, y_train, y_test = load_data()
    words_train, tags_train, words_test, tags_test = get_all_words_and_tags(X_train, y_train, X_test, y_test)

    # error_rate_known, error_rate_unknown, error_rate_total = question_b(words_train, tags_train, words_test, tags_test)
    # Result Qb
    # print(error_rate_known, error_rate_unknown, error_rate_total)

    unknown_words, unknown_words_tags, known_words, known_words_tags = know_and_unknown_test(words_test, tags_test)

    # error_rate_known, error_rate_unknown, error_rate_total = question_c(X_train, y_train, X_test, y_test,
    #                           words_train, tags_train, words_test, tags_test, known_words, unknown_words)
    # # Result Qc
    # print(error_rate_known, error_rate_unknown, error_rate_total)

    # error_rate_known, error_rate_unknown, error_rate_total = question_d(X_train, y_train, X_test, y_test,
    #                           words_train, tags_train, words_test, tags_test, known_words)
    # Result Qd
    # print(error_rate_known, error_rate_unknown, error_rate_total)

    pseudo_words = question_e(words_train, tags_train, words_test, tags_test)
    for w in set(pseudo_words):
        print(w)
