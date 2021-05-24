# Lea Fanny Setruk	Vladimir Balagula	

from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import operator
import matplotlib.pyplot as plt

NB_CLUSTERS = 9
LAMBDA = 1.35
ALPHA_THRESHOLD = 0.000001
EM_THRESHOLD = 10
K_PARAM = 15


def print_duration(func):
    """
    decorator to print time duration of attached function
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        print(f"{datetime.now()}: Started {func.__name__}")
        ret = func(*args, **kwargs)
        print(f"{datetime.now()}: Finished {func.__name__}")
        return ret

    return inner


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


def articles_topics_without_rare(dev_set_file_name):
    """
    Make an articles dictionary with index same index of headers dictionary. In order to couple topic with article
    Returns the word_frequency without rare words the articles without rare words
    the word frequency in each article
    """
    articles_dict = {}
    headers_dict = {}
    word_frequency = {}
    header_id = 0
    article_id = 0

    with open(dev_set_file_name) as f:
        for line in f:
            if "<TRAIN" in line or "<TEST" in line:
                headers_dict[header_id] = line.replace("<", "").replace(">", "").replace("\n", "").split("\t")[2:]
                header_id += 1

            elif line != "\n":
                line_article = line.strip().split(' ')
                articles_dict[article_id] = line_article
                article_id += 1

                for word in line_article:
                    if word not in word_frequency:
                        word_frequency.setdefault(word, 1)
                    else:
                        word_frequency[word] += 1

    word_frequency = words_freq_without_rare(word_frequency)
    articles_dict = delete_rare_words(word_frequency, articles_dict)
    frequency_in_article = frequency_word_in_article(articles_dict)

    return word_frequency, articles_dict, frequency_in_article, headers_dict


@print_duration
def likelihood_calculation(m, z):
    """
    Calculation of likelihood like written on the helper doc.
    We need in EM to calculate m and z
    """
    likelihood = 0

    for t in range(len(m)):
        sum_exp_z = 0

        for i in range(len(z[t])):
            z_m = z[t][i] - m[t]

            if z_m >= -K_PARAM:
                sum_exp_z += np.exp(z_m)

        likelihood += np.log(sum_exp_z) + m[t]

    return likelihood


@print_duration
def perplexity_calculation(current_likelihood, words):
    """
    Calculate perplexity using likelihood and words count
    """
    return 2**(-1 / sum(words.values()) * current_likelihood)


@print_duration
def plot_graph(epoch, y, label):
    """
    create and save figure of data
    """
    x = [i for i in range(epoch)]
    plt.figure()
    plt.plot(x, y, label=label)
    plt.xlabel("Iterations")
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label + ".png")


class EM(object):
    """
    EM algorithm
    """
    def __init__(self, articles_dict, words):
        self.clusters = defaultdict(list)
        self.weights = defaultdict(dict)
        self.articles_dict = articles_dict
        self.words = words

    def _init_alpha_probability(self):
        """
        first init of the weights and clusters
        """
        self.clusters_init()
        alpha, probs = self.m_step()
        return alpha, probs

    def clusters_init(self):
        """
        init cluster
        """
        for i in range(len(self.articles_dict)):
            cluster = i % NB_CLUSTERS
            self.clusters[cluster].append(i)

        for c_id, art_lst in self.clusters.items():
            for art in art_lst:
                self.weights[art] = defaultdict(lambda: 0)
                self.weights[art][c_id] = 1

    def lidstone_smooth(self, word_frequency, train_set_size, vocabulary_size):
        """
        calculate probability using lidstone smooth
        """
        return (word_frequency + LAMBDA) / (train_set_size + LAMBDA * vocabulary_size)

    @print_duration
    def m_step(self):
        """
        maximize step of em algorithm
        """
        relation_cluster = []
        alpha = [0] * len(self.clusters)
        probs = defaultdict(dict)
        for cl_id in self.clusters:
            relation_cluster.append(sum([self.weights[art][cl_id] * sum(self.articles_dict[art].values())
                                         for art in self.articles_dict]))

        for word in self.words:
            probs[word] = {}
            for cl_id in self.clusters:
                m_num = 0
                for art_id in self.articles_dict:
                    if word in self.articles_dict[art_id] and self.weights[art_id][cl_id] != 0:
                        m_num += self.weights[art_id][cl_id] * self.articles_dict[art_id][word]
                probs[word][cl_id] = self.lidstone_smooth(m_num, relation_cluster[cl_id], len(self.words))

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
        """
        z calculation
        """
        z_list = []
        for i in self.clusters:
            sum_ln = 0
            for word in article:
                sum_ln += np.log(probs[word][i]) * article[word]
            z_list.append(np.log(alpha[i]) + sum_ln)
        return z_list, max(z_list)

    @print_duration
    def e_step(self, alfa, proba):
        """
        expectation step of EM algorithm
        """
        m = []
        z = []

        for art_id, article in self.articles_dict.items():
            self.weights[art_id] = {}
            z_value_current_sum = 0

            z_list, m_list = self.z_list_computation(article, proba, alfa)

            for i in self.clusters:
                if z_list[i] - m_list < -K_PARAM:
                    self.weights[art_id][i] = 0

                else:
                    self.weights[art_id][i] = np.exp(z_list[i] - m_list)
                    z_value_current_sum += self.weights[art_id][i]

            for i in self.clusters:
                self.weights[art_id][i] /= z_value_current_sum

            z.append(z_list)
            m.append(m_list)

        return z, m

    def run(self):
        likelihood_list = []
        perplexity_list = []
        current_likelihood = -999999999
        previous_likelihood = -np.inf
        epoch = 0

        alpha, probs = self._init_alpha_probability()

        while current_likelihood - previous_likelihood > EM_THRESHOLD:
            previous_likelihood = current_likelihood
            z, m = self.e_step(alpha, probs)

            alpha, probs = self.m_step()
            current_likelihood = likelihood_calculation(m, z)
            print(f"{datetime.now()}: Current likelihood {current_likelihood}")
            current_perplexity = perplexity_calculation(current_likelihood, self.words)

            likelihood_list.append(current_likelihood)
            perplexity_list.append(current_perplexity)
            epoch += 1
        plot_graph(epoch, likelihood_list, "Likelihood lambda="+str(LAMBDA))
        plot_graph(epoch, perplexity_list, "Perplexity lambda="+str(LAMBDA))

    @print_duration
    def create_confusion_matrix(self, headers, topics):
        conf_mat = np.zeros((NB_CLUSTERS, len(topics)))
        for art_id in self.articles_dict:
            cl_id = max(self.weights[art_id].items(), key=operator.itemgetter(1))[0]
            for top in headers[art_id]:
                conf_mat[cl_id][topics.index(top)] += 1

        with open("confusion_matrix.csv", "w") as f:
            f.write("cluster\\topics," + ",".join([topic for topic in topics]) + "\n")
            for i in range(NB_CLUSTERS):
                f.write(str(i)+",")
                for j in range(len(topics)):
                    f.write(str(conf_mat[i][j])+",")
                f.write("\n")
        return conf_mat

    def calculate_accuracy(self, headers, topic_dict):
        cumulative = 0
        for art_id in self.articles_dict:
            cl_id = max(self.weights[art_id].items(), key=operator.itemgetter(1))[0]
            if topic_dict[cl_id] in headers[art_id]:
                cumulative += 1
        return cumulative/len(self.articles_dict)

# confusion matrix
# Table with clusters
# Histograms (9 histos of topics 1 for each cluster) X: 9 topics, Y: nb of articles from that topic in this cluster
# Function to calculate accuracy
# Report

def main():
    global LAMBDA
    dev_set_file_name = "dataset/develop.txt"  # sys.argv[1]
    topics_set_file_name = "dataset/topics.txt"  # sys.argv[2]
    topics = get_topics(topics_set_file_name)
    word_frequency, articles, frequency_in_article, headers = articles_topics_without_rare(dev_set_file_name)
    em_algo = EM(frequency_in_article, word_frequency)
    em_algo.run()
    confusion_matrix = em_algo.create_confusion_matrix(headers, topics)
    cluster_to_topic = {}
    for cluster_ind in range(NB_CLUSTERS):
        index = int(np.argmax(confusion_matrix[cluster_ind]))
        cluster_to_topic[cluster_ind] = topics[index]
    print(f"Accuracy for lambda {LAMBDA}: {em_algo.calculate_accuracy(headers, cluster_to_topic)}")


if __name__ == "__main__":
    main()
