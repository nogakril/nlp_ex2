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
    words_tagged_correctly = 0
    for w, t in zip(words_test, tags_test):
        if most_likely_tags.get(w, 'NN') == t:
            words_tagged_correctly += 1
    return 1 - (words_tagged_correctly / len(words_test))


def compute_emission(words, tags, add_one=False):
    cache = {tag: {} for tag in tags_train}
    for w, t in zip(words, tags):
        cache_tag = cache[t]
        cache_tag[w] = cache_tag.get(w, 0) + 1

    emission = {}
    for w, t in zip(words, tags):
        # if add_one:
        #     emission[(w, t)] = (cache[t][w] + 1)/ (sum(cache[t].values()) + len(words_train))
        # else:
        emission[(w, t)] = cache[t][w] / sum(cache[t].values())
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


def viterbi_algorithm(sent, tags, emission, transition):
    # dp table for words * tags (including unknown tags)
    tags = tags + [tag_unknown_words]
    dp = [[[0, None] for i in range(len(tags))] for i in range(len(sent) + 1)]

    # init base layer
    for k in range(len(tags)):
        t = transition.get((tag_start, tags[k]), 1)
        e = emission.get((sent[0], tags[k]), 1)
        dp[0][k][0] = t * e

    # algorithm
    for i in range(1, len(sent)):
        for tag in range(len(tags)):
            # [probability, back pointer]
            e = emission.get((sent[i], tags[tag]), 1)
            for prev_tag in range(len(tags)):
                t = transition.get((tags[prev_tag], tags[tag]), 1)
                val = dp[i - 1][prev_tag][0] * t * e
                if val >= dp[i][tag][0]:
                    dp[i][tag][0] = val
                    dp[i][tag][1] = prev_tag
    output = []
    last_tag_idx, max_val = 0, 0
    for val, i in dp[-1]:
        if val > max_val:
            last_tag_idx, max_val = i, val
    for i in range(len(sent) - 1, -1, -1):
        output.append(tags[last_tag_idx])
        last_tag_idx = dp[i][last_tag_idx][1]

    return output[::-1]


def question_c(X_train, y_train, X_test, y_test,
               words_train, tags_train, words_test, tags_test):
    # part i: emission
    emission = compute_emission(words_train, tags_train)

    # part i: transition
    transition = compute_transition(y_train)

    # part iii
    sents_tagged_correctly = 0
    for sample, tags in zip(X_test, y_test):
        output = viterbi_algorithm(sample, tags, emission, transition)
        if output == tags:
            sents_tagged_correctly += 1
    return 1 - (sents_tagged_correctly / len(X_test))


def question_e(words_train, tags_train, words_test, tags_test):
    # Part i
    pseudo_words, pseudo_words_tags = [], []
    # Add words with low frequency
    freq = {}
    for w in words_train:
        freq[w] = freq.get(w, 0) + 1
    for word, tag in zip(words_train, tags_train):
        if freq[word] < 10:
            pseudo_words.append(word)
            pseudo_words_tags.append(tag)
    # add words not in train set
    set_words_train = set(words_train)
    for word, tag in zip(words_test, tags_test):
        if word not in set_words_train:
            pseudo_words.append(word)
            pseudo_words_tags.append(tag)





if __name__ == '__main__':
    # For me: what's going on with the unknown tags?

    X_train, X_test, y_train, y_test = load_data()
    words_train, tags_train, words_test, tags_test = get_all_words_and_tags(X_train, y_train, X_test, y_test)

    error_rate_b = question_b(words_train, tags_train, words_test, tags_test)
    # Result Qb
    print(error_rate_b)

    # error_rate_c = question_c(X_train, y_train, X_test, y_test,
    #           words_train, tags_train, words_test, tags_test)
    # # Result Qc
    # print(error_rate_c)

    question_e(words_train, tags_train, words_test, tags_test)
