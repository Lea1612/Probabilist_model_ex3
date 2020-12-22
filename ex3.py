# Lea Fanny Setruk	Vladimir Balagula	345226179	323792770

import math
from collections import Counter

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

    for word, frequency in word_frequency:

        if frequency > 3:
            filtered_words_freq[word] = frequency

    return filtered_words_freq

# Delete the rare words from the articles.
def delete_rare_words(word_frequency, articles):
    filtered_articles = {}

    for id, article in articles:
        article_without_rare = []
        for word in article:
            if word in word_frequency:
                article_without_rare.append(word)
        filtered_articles[id] = article_without_rare

    return filtered_articles

# n_t_k in class : frequency of word k in doc t. 
# Returns the dictionary of word_frequency in a specific article for every acrticles
def frequency_word_in_article(self, articles):
    article_words_counter = {}

    for id, article in articles:
        article_words_counter[id] = Counter(article)

    return article_words_counter

# Get the articles, the headers, the words frequencies, word frequencies in each article
def articles(dev_set_file_name):
    articles_dict = {}
    headers = {}
    word_frequency = {}

    with open(dev_set_file_name) as f:
        for line in f:
            if "<TRAIN" in line or "<TEST" in line:
                #headers ?
            # else :
            #   articles + word_frequency
            # word_frequency without rare words
            # word frequency in specific article
                pass
                
      
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
    # print(topics)


if __name__ == "__main__":
    main()

    