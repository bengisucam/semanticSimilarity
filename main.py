from datasets import load_dataset
from sklearn import preprocessing
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import time
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def remove_stop_words_and_punctuations(sentence):  # stop wordsleri silmeye gerek var mı?
    # remove punctuation
    sentence = [word.lower() for word in sentence if word.isalpha()]
    # remove stop words
    stop_words = set(stopwords.words('english'))
    for word in sentence:
        if word in stop_words:
            sentence.remove(word)
    return sentence


def return_nouns(sentence):
    # noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    tokenized = nltk.word_tokenize(sentence)
    without_stop_words = remove_stop_words_and_punctuations(tokenized)
    return [word for (word, pos) in nltk.pos_tag(without_stop_words) if pos.startswith('N')]


def return_verbs(sentence):
    # verb_tags = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    tokenized = nltk.word_tokenize(sentence)
    without_stop_words = remove_stop_words_and_punctuations(tokenized)
    return [word for (word, pos) in nltk.pos_tag(without_stop_words) if pos.startswith('V')]


def return_adjectives(sentence):
    # verb_tags = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
    tokenized = nltk.word_tokenize(sentence)
    without_stop_words = remove_stop_words_and_punctuations(tokenized)
    return [word for (word, pos) in nltk.pos_tag(without_stop_words) if pos.startswith('J')]


def calculate_average_depth(max_depth, min_depth):
    return (max_depth + min_depth) / 2


def calculcate_similarity_score(depth_node1, depth_node2, depth_common, max_depth):
    first_numerator = depth_node1 - depth_common
    second_numerator = depth_node2 - depth_common
    sim = (depth_common + 1) / ((first_numerator + 1) * (second_numerator + 1) * max_depth)
    return sim


def find_lowest_hypernym_and_calculate_distance(wn_w1, wn_w2):
    # average_node1_depth = calculate_average_depth(wn_w1.max_depth(), wn_w1.min_depth())
    # average_node2_depth = calculate_average_depth(wn_w2.max_depth(), wn_w2.max_depth())
    node1_depth, node2_depth = wn_w1.max_depth(), wn_w2.max_depth()
    hypernym_set = wn_w1.lowest_common_hypernyms(wn_w2)
    if len(hypernym_set) != 0:
        hypernym = hypernym_set[0]
        # average_hypernym_depth = calculate_average_depth(hypernym.max_depth(), hypernym.min_depth())
        hypernym_depth = hypernym.max_depth()

    else:
        # hypernym = wn.synset('entity.n.01')
        hypernym_depth = 0
    if hypernym_depth > node1_depth or hypernym_depth > node2_depth:
        print("here")
    return node1_depth, node2_depth, hypernym_depth


def calculate_similarity(first_word_list, second_word_list):
    similarity_scores = []
    if len(first_word_list) == 0 or len(second_word_list) == 0:
        return 0
    for w1 in first_word_list:
        partial_scores = []
        wn_w1_list = wn.synsets(w1)
        if len(wn_w1_list) == 0:
            partial_scores.append(0)
            continue
        for w2 in second_word_list:
            if w1 == w2:
                partial_scores.append(1)
            else:
                wn_w2_list = wn.synsets(w2)
                if len(wn_w2_list) == 0:
                    partial_scores.append(0)
                    continue
                depth_node1, depth_node2, hypernym = find_lowest_hypernym_and_calculate_distance(wn_w1_list[0],
                                                                                                 wn_w2_list[0])
                similarity_n1_n2 = calculcate_similarity_score(depth_node1, depth_node2, hypernym, max_depth=20)
                partial_scores.append(similarity_n1_n2)

        similarity_scores.append(max(partial_scores))
    if len(similarity_scores) == 0:
        return 0
    return sum(similarity_scores) / len(similarity_scores)


def calculate_pearson_correlation(x_list, y_list):
    # Apply the pearsonr()
    corr, _ = pearsonr(x_list, y_list)
    return corr


def plot_mean_squared_error(df):
    sns.regplot(data=df, x=df['our_normalized_similarity_score'], y=df['normalized_similarity_score'], scatter=False)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    start_time = time.time()
    dataset = load_dataset("stsb_multi_mt", "en", split="train")
    sim_scores = dataset["similarity_score"]

    normalized_sim_scores = preprocessing.normalize([sim_scores], norm='max', axis=1)
    normalized_sim_scores = normalized_sim_scores[0].reshape(len(normalized_sim_scores[0]), ).tolist()
    dataset = dataset.add_column(name="normalized_similarity_score", column=normalized_sim_scores)

    pair_similarity_scores = []
    for pairs in dataset:
        noun_list_first = return_nouns(pairs["sentence1"])
        noun_list_second = return_nouns(pairs["sentence2"])
        pair_similarity_noun = calculate_similarity(noun_list_first, noun_list_second)
        #pair_similarity_scores.append(pair_similarity_noun)

        verb_list_first = return_verbs(pairs["sentence1"])
        verb_list_second = return_verbs(pairs["sentence2"])
        pair_similarity_verb = calculate_similarity(verb_list_first, verb_list_second)
        # average_pair_similarity = (pair_similarity_noun + pair_similarity_verb) / 2
        # pair_similarity_scores.append(average_pair_similarity)
        #
        adj_list_first = return_adjectives(pairs["sentence1"])
        adj_list_second = return_adjectives(pairs["sentence2"])
        pair_similarity_adj = calculate_similarity(adj_list_first, adj_list_second)
        average_pair_similarity = (pair_similarity_noun + pair_similarity_verb + pair_similarity_adj) / 3
        pair_similarity_scores.append(average_pair_similarity)

    dataset = dataset.add_column(name="our_normalized_similarity_score", column=pair_similarity_scores)
    pearson_correlation = calculate_pearson_correlation(pair_similarity_scores,
                                                        dataset["normalized_similarity_score"])
    print('Pearsons correlation: %.3f' % pearson_correlation)
    print("--- %s seconds ---" % (time.time() - start_time))

    # when the threshold is 50%
    similar_pairs = [pair for pair in dataset if pair["normalized_similarity_score"] >= 0.50]
    nonsimilar_pairs = [pair for pair in dataset if pair["normalized_similarity_score"] < 0.50]
    our_similar_pairs_count = len([pair for pair in similar_pairs if pair["our_normalized_similarity_score"] >= 0.50])
    our_nonsimilar_pairs_count = len(
        [pair for pair in nonsimilar_pairs if pair["our_normalized_similarity_score"] < 0.50])
    print("Found Similar Pairs (100-50%) Percentage: ", our_similar_pairs_count * 100 / len(similar_pairs))
    print("Found NonSimilar Pairs (50-0%)Percentage: ", our_nonsimilar_pairs_count * 100 / len(nonsimilar_pairs))

    # when the threshold is 25%
    similar_pairs_count_1 = len([pair for pair in dataset if pair["normalized_similarity_score"] >= 0.75])
    similar_pairs_count_2 = len([pair for pair in dataset if pair["normalized_similarity_score"] >= 0.50 and pair[
        "normalized_similarity_score"] < 0.75])
    nonsimilar_pairs_count_1 = len([pair for pair in dataset if pair["normalized_similarity_score"] < 0.50 and pair[
        "normalized_similarity_score"] >= 0.25])
    nonsimilar_pairs_count_2 = len([pair for pair in dataset if pair["normalized_similarity_score"] < 0.25])
    our_similar_pairs_count_1 = len([pair for pair in similar_pairs if pair["our_normalized_similarity_score"] >= 0.75])
    our_similar_pairs_count_2 = len([pair for pair in similar_pairs if
                                     pair["our_normalized_similarity_score"] >= 0.50 and pair[
                                         "our_normalized_similarity_score"] < 0.75])
    our_nonsimilar_pairs_count_1 = len([pair for pair in nonsimilar_pairs if
                                        pair["our_normalized_similarity_score"] < 0.50 and pair[
                                            "our_normalized_similarity_score"] >= 0.25])
    our_nonsimilar_pairs_count_2 = len(
        [pair for pair in nonsimilar_pairs if pair["our_normalized_similarity_score"] < 0.25])
    print("Found Similar Pairs (100-75%) Percentage: ", our_similar_pairs_count_1 * 100 / similar_pairs_count_1)
    print("Found Similar Pairs (75-50% )Percentage: ", our_similar_pairs_count_2 * 100 / similar_pairs_count_2)
    print("Found NonSimilar Pairs (50-25%) Percentage: ", our_nonsimilar_pairs_count_1 * 100 / nonsimilar_pairs_count_1)
    print("Found NonSimilar Pairs (25-0%) Percentage: ", our_nonsimilar_pairs_count_2 * 100 / nonsimilar_pairs_count_2)

    plot_mean_squared_error(dataset)
