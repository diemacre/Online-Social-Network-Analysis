"""
Summarize data.
"""

import pickle
import random

def get_obj(name):
    """
    load stored data
    """
    
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def ave_num_clusters(clusters):
    tot = 0
    for cluster in clusters:
        tot += len(cluster.nodes())
    ave = tot / len(clusters)

    return ave


def main():
    text_file = open('summary.txt', 'w')
    text_file.write("Number of users collected:\n")
    text_file.write('\n')
    
    users = get_obj('user_info')
    text_file.write("There are %d initial users collected.\n" % (len(users)))
    text_file.write("All friends of these users are also collected for future analysis.\n")
    for user in users:
        text_file.write("%s has %d friends.\n" % (user['screen_name'], len(user['friends'])))
    text_file.write('\n')
    text_file.write('Number of messages collected:\n')
    text_file.write('\n')
    
    tweets = get_obj('tweets')
    text_file.write('For sentiment analysis, we collected %d tweets that mentioned Donnal Trump, which is the maximum number of tweets we could collect.\n' % len(tweets))
    text_file.write('\n')
    text_file.write('Number of communities discovered:\n')
    text_file.write('\n')
    text_file.write('We cluster all initial users and their friends in to different communities and exclude users that followed by less than two initial users and outliers.\n')
    
    clusters = get_obj('clusters')
    text_file.write("There are %d communities\n" % (len(clusters)))
    text_file.write('\n')
    a_n_clusters = ave_num_clusters(clusters)
    text_file.write('Average number of users per community: %d\n' % (a_n_clusters))
    text_file.write('\n')
    text_file.write('Number of instances per class found:\n')
    text_file.write('\n')
    text_file.write('There are three classes for sentiment analysis.\n')
    
    classified_results = get_obj('classified_results')
    text_file.write('The positive class has %d instances\n' % (len(classified_results['pos'])))
    text_file.write('The negative class has %d instances\n' % (len(classified_results['neg'])))
    text_file.write('The neutral class has %d instances\n' % (len(classified_results['neutral'])))
    text_file.write('\n')
    text_file.write('One example from each class:\n')
    text_file.write('\n')
    random.seed(4)
    text_file.write('Positive example:\n%s\n' % (classified_results['pos'].ix[random.sample(list(classified_results['pos'].index), 1)[0]]['Tweet']))
    random.seed(4)
    text_file.write('Negative example:\n%s\n' % (classified_results['neg'].ix[random.sample(list(classified_results['neg'].index), 1)[0]]['Tweet']))
    random.seed(4)
    text_file.write('Neutral example:\n%s\n' % (classified_results['neutral'].ix[random.sample(list(classified_results['neutral'].index), 1)[0]]['Tweet']))
    print('Write to summary.txt done.')

if __name__ == '__main__':
    main()
