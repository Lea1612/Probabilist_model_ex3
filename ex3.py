# Lea Fanny Setruk	Vladimir Balagula	345226179	323792770

import math
from collections import Counter, defaultdict

NB_CLUSTERS = 9
LAMBDA = 1  # ??


#
def get_topics(topics_set_file_name):
    """
    Get the topics list
    :param topics_set_file_name:
    :return:
    """
    return [line.strip() for line in open(topics_set_file_name).readlines()
            if line not in ["\n", ""]]


def words_freq_without_rare(word_frequency):
    """
    Calculate the frequency of the words when rare words are deleted.
    We get a new dictionary of words_frequency without the rare words.
    :param word_frequency:
    :return:
    """
    return {word: freq for word, freq in word_frequency.items() if freq > 3}


def delete_rare_words(word_frequency, articles):
    """
    Delete the rare words from the articles.
    :param word_frequency:
    :param articles:
    :return:
    """
    return {id_art: [w for w in article if w in word_frequency]
            for id_art, article in articles.items()}


def frequency_word_in_article(articles):
    """
    n_t_k in class : frequency of word k in doc t.
    Returns the dictionary of word_frequency in a specific article for every acrticles
    :param articles:
    :return:
    """
    return {id_art: Counter(article) for id_art, article in articles.items()}


# Make an articles dictionary with index same index of headers dictionary. In order to couple topic with article
# Returns the word_frequency without rare words the articles without rare words 
# the word frequency in each article
def articles_topics_without_rare(dev_set_file_name):
    articles_dict = {}
    headers_dict = {}
    word_frequency = {}
    header_id = 0
    article_id = 0

    with open(dev_set_file_name) as f:
        for line in f:
            # print(f'line:{line}')
            if "<TRAIN" in line or "<TEST" in line:
                headers_dict[header_id] = line.replace("<", "").replace(">", "").split("\t")
                header_id += 1

            elif line != "\n":
                line_article = line.strip().split(' ')
                # print(line_article)
                articles_dict[article_id] = line_article
                article_id += 1
                word_frequency = Counter(line_article)

    # print(f'art:{articles_dict}')
    # print(f'word_freq:{word_frequency}')
    # print(f'head:{headers_dict}')

    word_frequency = words_freq_without_rare(word_frequency)
    articles_dict = delete_rare_words(word_frequency, articles_dict)
    frequency_in_article = frequency_word_in_article(articles_dict)

    # print(word_frequency)
    # print(frequency_in_article)

    return word_frequency, articles_dict, frequency_in_article


# Calculation of likelihood like written on the helper doc. We need in EM to calculate m and z
def likelihood_calculation(m, z, k):
    # m = max(z_i)
    # z = z_i = ln(alpha_i...)
    # k = parameter (=10?)
    likelihood = 0

    for t in range(len(m)):
        # sum_exp_z = sum(exp(z_i))
        sum_exp_z = 0

        for i in range(len(z[t])):
            # z_m = z_i^t - m^t
            z_m = z[t][i] - m[t]

            if z_m >= -k:
                sum_exp_z += math.exp(z_m)

        likelihood += math.log(sum_exp_z) + m[t]

    return likelihood


class EM(object):
    def __init__(self, articles_dict):
        self.clusters = defaultdict(list)
        self.weights = defaultdict(dict)
        self._init_probability(articles_dict)

    def _init_probability(self,articles_dict):
        self.clusters_init(articles_dict)
        self.m_step()

    def clusters_init(self, articles_dict):

        for i in range(len(articles_dict)):
            cluster = i % NB_CLUSTERS
            self.clusters[cluster].append(i)

        for c_id, art_lst in self.clusters.items():
            for art in art_lst:
                self.weights[art] = defaultdict(lambda: 0)
                self.weights[art][c_id] = 1


    def lidstone_smooth(self, word_frequency, train_set_size, vocabulary_size, lambda_val):
        return (word_frequency + lambda_val) / (train_set_size + lambda_val * vocabulary_size)

    def m_step(self, article_list, words):
        relation_cluster = []
        probs = defaultdict(dict)
        for cl_id in self.clusters:
            relation_cluster.append(sum([self.weights[art][cl_id]*sum(article_list[art].values())
                                    for art in article_list]))

        for word in words:
            m_num = 0
            for cl_id in self.clusters:
                [for art_id]
                for art_id in article_list:
                    if word in article_list[art_id] and self.weights[art_id][cl_id]!=0:
                        m_num += self.weights[art_id][cl_id]*article_list[art_id][word]





def e_step():
    # m = max(z_i)
    # z = z_i = ln(alpha_i...)
    m = []
    z = []
    w = {}


# function for perplexity
# e step
# m step
# confusion matrix
# Table with clusters
# Histograms (9 histos of topics 1 for each cluster) X: 9 topics, Y: nb of articles from that topic in this cluster
# Function to calculate accuracy
# Report

def main():
    dev_set_file_name = "dataset/develop.txt"  # sys.argv[1]
    topics_set_file_name = "dataset/topics.txt"  # sys.argv[2]
    topics = get_topics(topics_set_file_name)
    word_frequency, articles_dict, frequency_in_article = articles_topics_without_rare(dev_set_file_name)
    EM(articles_dict)
    print("X")


if __name__ == "__main__":
    main()
