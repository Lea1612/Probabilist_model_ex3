# Lea Fanny Setruk	Vladimir Balagula	345226179	323792770

import math
from collections import Counter

k = 10
nb_clusters = 9
lambda_val = 1 #??

# Get the topics list
def get_topics(topics_set_file_name):
    topics_list = []

    with open(topics_set_file_name) as f:

        for line in f:
            
            if line != "\n" and line != "":
                topics_list.append(line.strip())

    return topics_list


# Calculate the frequency of the words when rare words are deleted. 
# We get a new dictionary of words_frequency without the rare words.
def words_freq_without_rare(word_frequency):
    filtered_words_freq = {}

    for word, frequency in word_frequency.items():

        if frequency > 3:
            filtered_words_freq[word] = frequency

    return filtered_words_freq

# Delete the rare words from the articles.
def delete_rare_words(word_frequency, articles):
    filtered_articles = {}

    for id, article in articles.items():
        article_without_rare = []
        for word in article:
            if word in word_frequency:
                article_without_rare.append(word)
        filtered_articles[id] = article_without_rare

    return filtered_articles

# n_t_k in class : frequency of word k in doc t. 
# Returns the dictionary of word_frequency in a specific article for every acrticles
def frequency_word_in_article(articles):
    article_words_counter = {}

    for id, article in articles.items():
        article_words_counter[id] = Counter(article)

    return article_words_counter

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
def likelihood_calculation(m, z, k):
    # m = max(z_i)
    # z = z_i = ln(alpha_i...)
    # k = parameter (=10?)
    likelihood = 0

    for t in range(len(m)):
        # sum_exp_z = sum(exp(z_i))
        sum_exp_z = 0

        for i in range (len(z[t])):
            # z_m = z_i^t - m^t
            z_m = z[t][i] - m[t]

            if z_m >= -k:
                sum_exp_z += math.exp(z_m)

        likelihood += math.log(sum_exp_z) + m[t]

    return likelihood

def lidstone_smooth(word_frequency, train_set_size, vocabulary_size, lambda_val):
        return (word_frequency + lambda_val) / (train_set_size + lambda_val * vocabulary_size)

def clusters_init(articles_dict):
    clusters = {}

    for i in range(len(articles_dict)):
        cluster = (i+1) % nb_clusters

        # if cluster == 0 : if cluster = 9j modulo 9 : 9j-th article
        if cluster == 0:
            cluster = nb_clusters

        if cluster not in clusters:
            clusters[cluster] = []

        clusters[cluster].append(i)

    return clusters

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


if __name__ == "__main__":
    main()

    