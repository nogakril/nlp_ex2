from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import re
import matplotlib.pyplot as plt

tag_start = "START"
tag_unknown_words = "UN-KN"
tag_stop = "STOP"


def load_data():
    """
    Load corpus of tagged news samples and split it to train-test datasets
    :return:
    """
    tagged_sentences = brown.tagged_sents(categories='news')
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
    """
    Flattens train and test dataset to arrays that contains all words and tags in all samples
    :param X_train: an arrays of samples which are also arrays
    :param y_train: an arrays of samples which are also arrays
    :param X_test: an arrays of samples which are also arrays
    :param y_test: an arrays of samples which are also arrays
    :return: flatten array for each of the parameters
    """
    words_train_flatten = [word for sample in X_train for word in sample]
    tags_train_flatten = [tag for tags in y_train for tag in tags]
    words_test_flatten = [word for sample in X_test for word in sample]
    tags_test_flatten = [tag for tags in y_test for tag in tags]
    return words_train_flatten, tags_train_flatten, words_test_flatten, tags_test_flatten


def know_and_unknown_test(words_train, words_test):
    """
    Split all test words into known an unknown arrays
    :param words_train: an array contains all words in the train dataset
    :param words_test: an array contains all words in the test dataset
    :return:
    """
    unknown_words, known_words = [], []
    set_words_train = set(words_train)
    for word in words_test:
        if word not in set_words_train:
            unknown_words.append(word)
        else:
            known_words.append(word)
    return unknown_words, known_words


def calc_error_rate(tags_true,
                    tags_pred,
                    known_words_tagged_correctly,
                    unknown_words_tagged_correctly,
                    known_words,
                    unknown_words,
                    words_test):
    """
    Calculates the error-rate for known, unknown and all words, an produces a confusion matrics
    :param tags_true: true tags of the the test set
    :param tags_pred: predicted tags of the test set
    :param known_words_tagged_correctly: how many known words tagged correctly
    :param unknown_words_tagged_correctly: how many unknown words tagged correctly
    :param known_words: all known words in the test set
    :param unknown_words: all unknown words in the test set
    :param words_test: all words in the test set
    :return:
    """
    conf_mat = ConfusionMatrixDisplay.from_predictions(tags_true, tags_pred)
    error_rate_known = 1 - (known_words_tagged_correctly / len(known_words))
    error_rate_unknown = 1 - (max(unknown_words_tagged_correctly, 1) / max(len(unknown_words), 1))
    error_rate_total = 1 - ((known_words_tagged_correctly + unknown_words_tagged_correctly) / len(words_test))
    return conf_mat, error_rate_known, error_rate_unknown, error_rate_total


def compute_emission(words_train, words_test, tags_train, known_words, add_one=False):
    """
    Compute emission function for each word-tag pair using maximum likelihood estimation or Add-one smoothing
    :param words_train: all words in train set flatten
    :param words_test:  all words in test set flatten
    :param tags_train: all tags in train set flatten
    :param known_words: all known words in the test set
    :param add_one: bool defualt=False if to use Add-one smoothing instead of maximum likelihood estimation
    :return: dict of word-tag paris as keys and emissions as values
    """
    # train
    cache = {tag: {} for tag in tags_train}
    for w, t in zip(words_train, tags_train):
        cache_tag = cache[t]
        cache_tag[w] = cache_tag.get(w, 0) + 1

    tags = list(set(tags_train))
    emission = {}
    V = len(set(words_test + words_train))
    for w in words_test:
        for t in tags:
            if add_one:
                emission[(w, t)] = (cache[t].get(w, 0) + 1) / (sum(cache[t].values()) + V)
            else:
                if w not in known_words and t == 'NN':
                    emission[(w, t)] = 1
                else:
                    emission[(w, t)] = cache[t].get(w, 0) / sum(cache[t].values())
    return emission


def compute_transition(y_train):
    """
    Compute transitions probabilities over all tags in the train set
    :param y_train: array of samples of tags
    :return: dict of tag_prev-tag paris as keys and transition probability from tag_prev to tag as values
    """
    y_train = [tags + [tag_stop] for tags in y_train]
    cache = {}
    for tags in y_train:
        prev = tag_start
        for tag in tags:
            if prev not in cache:
                cache[prev] = {tag: 1}
            else:
                cache_tag = cache[prev]
                cache_tag[tag] = cache_tag.get(tag, 0) + 1
            prev = tag
    transition = {}
    for tags in y_train:
        prev = tag_start
        for tag in tags:
            transition[(prev, tag)] = cache[prev][tag] / sum(cache[prev].values())
            prev = tag
    return transition


def calc_q(u, v, x_k, transition, emission):
    """
    calculates the bigram HMM tagger probability of tag v
    :param u: tag we're transitioning from
    :param v: tag we're transitioning to
    :param x_k: word we're transitioning to
    :param transition: transitions dict
    :param emission: emissions dict
    :return: the probability of transitioning to v given u and x_k
    """
    return transition.get((u, v), 0) * emission.get((x_k, v))


def viterbi_algorithm(sent, tags, emission, transition):
    """
    Implementation of inference version of Bigram Viterbi algorithm
    :param sent: sample of words
    :param tags: set of all known tags
    :param emission: emission dict
    :param transition: transition dict
    :return: predicted tags
    """
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


def run_viterbi_calc_errors(y_train,
                            words_train,
                            tags_train,
                            words_test,
                            tags_test,
                            X_test,
                            y_test,
                            known_words,
                            unknown_words, laplace_flag=False):
    """
    Computes emissions and transitions over the data, then runs the Viterbi algorithm and calculates the error rates
    :param y_train: array of samples of tags in train set
    :param words_train: words in train set flatten
    :param tags_train: tags in train set flatten
    :param words_test: words in test set flatten
    :param tags_test: tags in test set flatten
    :param X_test: array of samples of words in test set
    :param y_test: array of samples of tags in test set
    :param known_words: all known words in the test set
    :param unknown_words: all unknown words in the test set
    :param laplace_flag: bool default=False if to compute emission with Add-one smoothing
    :return: confusion matrix and errors rates
    """
    # part i: emission
    emission = compute_emission(words_train, words_test, tags_train, known_words, laplace_flag)

    # part i: transition
    transition = compute_transition(y_train)

    # part iii
    all_tags = list(set(tags_train))

    known_words_tagged_correctly = 0
    unknown_words_tagged_correctly = 0
    tags_pred = []

    for sample, tags in zip(X_test, y_test):
        output = viterbi_algorithm(sample, all_tags, emission, transition)
        tags_pred.extend(output)
        for i, w in enumerate(sample):
            if w in known_words:
                if output[i] == tags[i]:
                    known_words_tagged_correctly += 1
            else:
                if output[i] == tags[i]:
                    unknown_words_tagged_correctly += 1
    return calc_error_rate(tags_test, tags_pred, known_words_tagged_correctly, unknown_words_tagged_correctly,
                           known_words, unknown_words, words_test)


def get_pseudo_word(word):
    """
    Find a pseud-word for a given word
    :param word:
    :return:
    """
    if re.match("(\d*\.?\d+|\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\S)", word):
        return 'NUMBER'
    elif len(word) >= 3 and word[0].isupper() and word[-2] == '\'' and word[-1].islower():
        return 'NAME_WITH_POSSESSION_S'
    elif re.match("\$(\d*\.?\d+|\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?!\S)", word):
        return 'SUM_OF_MONEY'
    elif word[0].isupper():
        return 'initCap'
    elif len(word) >= 2 and word[-2:] == "ed":
        return 'ed_SUFFIX'
    elif len(word) >= 2 and word[-2:] == "ly":
        return 'ly_SUFFIX'
    elif len(word) >= 3 and word[-3:] == "ing":
        return 'ing_SUFFIX'
    elif len(word) >= 3 and word[-3:] == "est":
        return 'est_SUFFIX'
    else:
        return 'OTHER'


def question_b(words_train,
               tags_train,
               words_test,
               tags_test,
               unknown_words,
               known_words):
    """
     Using the training set, computes for each word the tag that maximizes p(tag|word), based
     on the maximum likelihood estimation. The most likely tag of all the unknown
     words is “NN”.
     Computes the error rate for known, unknown, and all words as well the confusion matrix
    """
    # Part i
    cache = {word: {} for word in words_train}
    for w, t in zip(words_train, tags_train):
        cache_word = cache[w]
        cache_word[t] = cache_word.get(t, 0) + 1
    # most likely tag for each word in the training set
    most_likely_tags = {w: max(cache[w], key=cache[w].get) for w in words_train}

    # Part ii
    known_words_tagged_correctly = 0
    unknown_words_tagged_correctly = 0
    tags_pred = []

    for w, t in zip(words_test, tags_test):
        tags_pred.append(most_likely_tags.get(w, 'NN'))
        if w in most_likely_tags and most_likely_tags[w] == t:
            known_words_tagged_correctly += 1
        elif w not in most_likely_tags:
            if 'NN' == t:
                unknown_words_tagged_correctly += 1
    return calc_error_rate(tags_test, tags_pred, known_words_tagged_correctly, unknown_words_tagged_correctly,
                           known_words, unknown_words, words_test)


def question_c(y_train,
               words_train,
               tags_train,
               words_test,
               tags_test,
               X_test,
               y_test,
               known_words,
               unknown_words):
    return run_viterbi_calc_errors(y_train, words_train, tags_train, words_test,
                                   tags_test, X_test, y_test, known_words, unknown_words)


def question_d(y_train,
               words_train,
               tags_train,
               words_test,
               tags_test,
               X_test,
               y_test,
               known_words,
               unknown_words):
    laplace_flag = True
    return run_viterbi_calc_errors(y_train, words_train, tags_train, words_test,
                                   tags_test, X_test, y_test, known_words, unknown_words, laplace_flag)


def question_e(y_train, words_train, tags_train, words_test,
               tags_test, X_test, y_test, unknown_words, laplace_flag=False):
    # Find frequency of all words in train
    freq = {}
    for w in words_train:
        freq[w] = freq.get(w, 0) + 1

    # Part i - for all low frequency and unknown words get a pseudo-word, and replace then in the datasets
    for i, word in enumerate(words_train):
        if freq[word] < 5:
            words_train[i] = get_pseudo_word(word)
    for i, word in enumerate(words_test):
        if word in unknown_words:
            words_test[i] = get_pseudo_word(word)
        elif freq[word] < 5:
            words_test[i] = get_pseudo_word(word)
    for sent in X_test:
        for i, word in enumerate(sent):
            if word in unknown_words:
                sent[i] = get_pseudo_word(word)
            elif freq[word] < 5:
                sent[i] = get_pseudo_word(word)

    # Part ii-iii: Use the new datasets with maximum likelihood estimation or Add-one smoothing
    unknown_words, known_words = know_and_unknown_test(words_train, words_test)
    return run_viterbi_calc_errors(y_train, words_train, tags_train, words_test, tags_test, X_test,
                                   y_test, known_words, unknown_words, laplace_flag)


def main():
    # Question a: load data and split to train-test datasets
    X_train, X_test, y_train, y_test = load_data()
    words_train, tags_train, words_test, tags_test = get_all_words_and_tags(X_train, y_train, X_test, y_test)
    unknown_words, known_words, = know_and_unknown_test(words_train, words_test)

    # Question b: Calc maximum likelihood estimation
    conf_mat_b, error_rate_known_b, error_rate_unknown_b, error_rate_total_b = question_b(words_train, tags_train,
                                                                                          words_test,
                                                                                          tags_test, unknown_words,
                                                                                          known_words)
    # Result Qb
    print("=========================================")
    print(f"Qb - Known words error rate: {error_rate_known_b}")
    print(f"Qb - Unknown words error rate: {error_rate_unknown_b}")
    print(f"Qb - All words error rate: {error_rate_total_b}")
    print("=========================================\n")
    plt.show()

    # Question c: Implementation of bigram HMM tagger using Viterbi algorithm
    conf_mat_c, error_rate_known_c, error_rate_unknown_c, error_rate_total_c = question_c(y_train, words_train,
                                                                                          tags_train, words_test,
                                                                                          tags_test,
                                                                                          X_test, y_test, known_words,
                                                                                          unknown_words)
    # Result Qc
    print("=========================================")
    print(f"Qc - Known words error rate: {error_rate_known_c}")
    print(f"Qc - Unknown words error rate: {error_rate_unknown_c}")
    print(f"Qc - All words error rate: {error_rate_total_c}")
    print("=========================================\n")
    plt.show()

    # Question d: Compute emission probabilities using Laplace smoothing
    conf_mat_d, error_rate_known_d, error_rate_unknown_d, error_rate_total_d = question_d(y_train, words_train,
                                                                                          tags_train, words_test,
                                                                                          tags_test,
                                                                                          X_test, y_test, known_words,
                                                                                          unknown_words)
    # Result Qd
    print("=========================================")
    print(f"Qd - Known words error rate: {error_rate_known_d}")
    print(f"Qd - Unknown words error rate: {error_rate_unknown_d}")
    print(f"Qd - All words error rate: {error_rate_total_d}")
    print("=========================================\n")
    plt.show()

    # Question e: Using pseudo-words instead of unknown and low frequency words

    # Use pseudo-words and maximum likelihood estimation
    conf_mat_e_ii, error_rate_known_e_ii, error_rate_unknown_e_ii, error_rate_total_e_ii = question_e(y_train,
                                                                                                      words_train,
                                                                                                      tags_train,
                                                                                                      words_test,
                                                                                                      tags_test,
                                                                                                      X_test, y_test,
                                                                                                      unknown_words)
    # Result Qe_ii
    print("=========================================")
    print(f"Qe_ii - Known words error rate: {error_rate_known_e_ii}")
    print(f"Qe_ii - Unknown words error rate: {error_rate_unknown_e_ii}")
    print(f"Qe_ii - All words error rate: {error_rate_total_e_ii}")
    print("=========================================\n")
    plt.show()

    # Use pseudo-words and Add-one smoothing
    laplace_flag = True
    conf_mat_e_iii, error_rate_known_e_iii, error_rate_unknown_e_iii, error_rate_total_e_iii = question_e(y_train,
                                                                                                          words_train,
                                                                                                          tags_train,
                                                                                                          words_test,
                                                                                                          tags_test,
                                                                                                          X_test,
                                                                                                          y_test,
                                                                                                          unknown_words,
                                                                                                          laplace_flag)
    # Result Qe_iii
    print("=========================================")
    print(f"Qe_iii - Known words error rate: {error_rate_known_e_iii}")
    print(f"Qe_iii - Unknown words error rate: {error_rate_unknown_e_iii}")
    print(f"Qe_iii - All words error rate: {error_rate_total_e_iii}")
    print("=========================================\n")
    plt.show()


if __name__ == '__main__':
    main()
