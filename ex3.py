# Lea Fanny Setruk	Vladimir Balagula	345226179	323792770

import math
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

NB_CLUSTERS = 9
LAMBDA = 1  # ??
ALPHA_THRESHOLD = 0.000001
EM_THRESHOLD = 10
K_PARAM = 10


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
                
                for word in line_article:
                    if word not in word_frequency:
                        word_frequency.setdefault(word, 1)
                    else:
                        word_frequency[word] += 1

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
def likelihood_calculation(m, z):
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

            if z_m >= -K_PARAM:
                sum_exp_z += math.exp(z_m)

        likelihood += np.log(sum_exp_z) + m[t]

    return likelihood

def perplexity_calculation(current_likelihood, words):
        return math.pow(2, (-1 / sum(words.values()) * current_likelihood))


def plot_graph(epoch, y, label):
    x = [i for i in range(epoch)]

    plt.plot(x, y, label=label)
    plt.xlabel("Iterations")
    plt.ylabel(label)
    # plt.xlim(0, epoch)
    # plt.ylim(min(y), max(y))
    plt.legend()
    plt.savefig(label + ".png")


class EM(object):
    def __init__(self, articles_dict, words):
        self.clusters = defaultdict(list)
        self.weights = defaultdict(dict)
        self.articles_dict = articles_dict
        self.words = words
        self._init_alpha_probability()

    def _init_alpha_probability(self):
        self.clusters_init()
        alpha, probs = self.m_step()
        return(alpha, probs)

    def clusters_init(self):
        for i in range(len(self.articles_dict)):
            cluster = i % NB_CLUSTERS
            self.clusters[cluster].append(i)

        for c_id, art_lst in self.clusters.items():
            for art in art_lst:
                self.weights[art] = defaultdict(lambda: 0)
                self.weights[art][c_id] = 1


    def lidstone_smooth(self, word_frequency, train_set_size, vocabulary_size):
        return (word_frequency + LAMBDA) / (train_set_size + LAMBDA * vocabulary_size)

    def m_step(self):
        relation_cluster = []
        alpha = [0] * len(self.clusters)
        probs = defaultdict(dict)
        for cl_id in self.clusters:
            relation_cluster.append(sum([self.weights[art][cl_id]*sum(self.articles_dict[art].values())
                                    for art in self.articles_dict]))

        for word in self.words:
            m_num = 0
            probs[word] = {}
            for cl_id in self.clusters:
                for art_id in self.articles_dict:
                    if word in self.articles_dict[art_id] and self.weights[art_id][cl_id]!=0:
                        m_num += self.weights[art_id][cl_id]*self.articles_dict[art_id][word]
                probs[word][cl_id] = self.lidstone_smooth(m_num, relation_cluster[cl_id],len(self.words))
        
        for i in self.clusters:
            for art_id in self.articles_dict:
                alpha[i] += self.weights[art_id][i]
            alpha[i] /= len(self.articles_dict)

        for i in range(len(alpha)):
            if alpha[i] < ALPHA_THRESHOLD:
                alpha[i] = ALPHA_THRESHOLD
        
        alpha = [topic_index / sum(alpha) for topic_index in alpha]

        return alpha, probs

    def z_list_computation(self, article, probs, alpha):
        z_list = []
        for i in self.clusters:
                sum_ln = 0
                for word in article:
                    sum_ln += np.log(probs[word][i]) * article[word]
                z_list.append(np.log(alpha[i]) + sum_ln)
        return z_list, max(z_list)


    def e_step(self, alfa, proba):
        # m = max(z_i)
        # zi = ln(ai) + Sigma(Ntk) * Pik
        # m_list = []
        # z_list = []
        m = []
        z = []

        # alpha, probs = self._init_alpha_probability()

        for art_id, article in self.articles_dict.items():
            self.weights[art_id] = {}
            z_value_current_sum = 0

            # for i in self.clusters:
            #     sum_ln = 0
            #     for word in article:
            #         sum_ln += np.log(probs[word][i]) * article[word]
            #     z_list.append(np.log(alpha[i]) + sum_ln)
            # m_list = max(z_list)

            z_list, m_list = self.z_list_computation(article, proba, alfa)

            for i in self.clusters:
                if z_list[i] - m_list < -K_PARAM:
                    self.weights[art_id][i] = 0

                else:
                    self.weights[art_id][i] = math.exp(z_list[i] - m_list)
                    z_value_current_sum += self.weights[art_id][i]
            
            for i in self.clusters:
                self.weights[art_id][i] /= z_value_current_sum

            z.append(z_list)
            m.append(m_list)

        return z, m


    def run_em(self):
        likelihood_list = []
        perplexity_list = []
        current_likelihood = -10000000
        previous_likelihood = -10000101
        epoch = 0

        alpha, probs = self._init_alpha_probability()

        while current_likelihood - previous_likelihood > EM_THRESHOLD:
            z, m = self.e_step(alpha, probs)

            alpha, probs = self.m_step()
            previous_likelihood = current_likelihood

            current_likelihood = likelihood_calculation(m,z)
            current_perplexity = perplexity_calculation(current_likelihood, self.words)

            likelihood_list.append(current_likelihood)
            perplexity_list.append(current_perplexity)

            epoch += 1
        print('hey')
        print(likelihood_list)

        plot_graph(epoch, likelihood_list, "Likelihood")
        plot_graph(epoch, perplexity_list, "Perplexity")

        # return w


# confusion matrix
# Table with clusters
# Histograms (9 histos of topics 1 for each cluster) X: 9 topics, Y: nb of articles from that topic in this cluster
# Function to calculate accuracy
# Report

def main():
    dev_set_file_name = "dataset/develop.txt"  # sys.argv[1]
    topics_set_file_name = "dataset/topics.txt"  # sys.argv[2]
    topics = get_topics(topics_set_file_name)
    word_frequency, articles, frequency_in_article = articles_topics_without_rare(dev_set_file_name)
    em_algo = EM(frequency_in_article, word_frequency)
    em_algo.run_em()

    print("X")


if __name__ == "__main__":
    main()
