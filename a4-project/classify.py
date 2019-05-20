"""
Classify data.
"""


import numpy as np
import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix

def read_sentiments_labeled(document):
    """
    Read afinn document to dict, word to score.
    """
    
    senti = dict()
    with open(document, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                senti[parts[0]] = int(parts[1])

    print('Read %d sentiment terms.\nE.g.: %s' % (len(senti), str(list(senti.items())[:10])))

    return senti


senti = read_sentiments_labeled('sentimentsSetLabeled.txt')

def get_tweets(name):
    """
    Load stored tweets.
    List of strings, one per tweet.
    """

    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this 86 fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this 86 fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    
    if keep_internal_punct:
        strings = re.findall(r"[\w'-]+", doc.lower())
        for st in strings:
          if st=="'" or st=='-':
            strings.remove(st)
    else:
        strings = re.sub(r'\W+', r' ', doc.lower()).split()
    #strings= removeStopWords(strings)
    tokens = np.array(strings)
    return tokens


pos_tweets = []
neg_tweets = []
neutral_tweets = []


stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def removeStopWords(strings):
  for token in strings:
    if token in stop_words:
      strings.remove(token)
  return strings


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for key in feats.keys():
        feats[key] = 0
    for word in tokens:
            feats['token='+word] += 1
    return

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    if len(tokens) < k:
        k = len(tokens)

    for i in range(0, len(tokens)-k+1):
      comb = list(combinations(tokens[i:i+k], 2))
      for item in comb:
        feats['token_pair='+item[0]+'__'+item[1]] += 1
    return

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['pos_words'] = 0
    feats['neg_words'] = 0
    feats['neutral_words'] =0

    for token in tokens:
        if token.lower() in senti.keys() and senti[token] < 0:
            feats['neg_words'] += 1
        elif token.lower() in senti.keys() and senti[token] > 0:
            feats['pos_words'] += 1
        elif token.lower() in senti.keys() and senti[token] == 0:
            feats['neural_words'] += 1
    return

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats = defaultdict(lambda: 0)
    vals = []

    for function in feature_fns:
        function(tokens, feats)
    for key, value in feats.items():
        vals.append((key, value))
    vals.sort()

    return vals

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    # params: tokens_list, feature_fns, min_freq, vocab=none
    freq_tokens = defaultdict(int)
    vals = []
    cols = []
    rows = []
    filt_tokens = []

    for i in range(0, len(tokens_list)):
      feats = featurize(tokens_list[i], feature_fns)

      for [key, value] in feats:
        if value > 0:
          freq_tokens[key] += 1

    for key, value in freq_tokens.items():
        if value >= min_freq:
            filt_tokens.append(key)
    filt_tokens.sort()
    aux = True
    if vocab == None:
      aux = False
      vocab = defaultdict(int)
      for i in range(0, len(filt_tokens)):
          vocab[filt_tokens[i]] = i

    for i in range(0, len(tokens_list)):  # for each string in the document
      feats = featurize(tokens_list[i], feature_fns)
      #add to totalfeats all diferent feats od each doc
      if aux:
        #if there is a token on training vocab that is not on the test, we add that token to the feasts of test
        # just in the last loop so it is faster (do not need to do it for each doc)
        if i == len(tokens_list)-1:
          for token in vocab.keys():
              if token not in filt_tokens:
                # value 0 beacuse it is not on the features of test
                feats.append((token, 0))
          feats.sort()
      #create the values for the csr_matrix
      for [token, value] in feats:
          if token in vocab.keys():
              vals.append(value)
              rows.append(i)
              cols.append(vocab[token])
    #print(csr_matrix((vals, (rows, cols))).toarray())

    X = csr_matrix((vals, (rows, cols)), dtype=np.int64)
    return X, vocab

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth == predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    accuracies = []
    cv = KFold(n_splits=k)

    for train, test in cv.split(X):
        clf.fit(X[train], labels[train])
        predicted = clf.predict(X[test])
        accuracy = accuracy_score(labels[test], predicted)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO

    result = []
    for punc in punct_vals:
        tokens = [tokenize(doc, punc) for doc in docs]
        for i in range(1, len(feature_fns)+1):
            combs = combinations(feature_fns, i)
            for comb in combs:
                for min_fr in min_freqs:
                    dic = {}
                    model = LogisticRegression()
                    matr = vectorize(tokens, comb, min_fr, vocab=None)[0]
                    accuracies = cross_validation_accuracy(
                        model, matr, labels, 5)
                    dic['punct'] = punc
                    dic['features'] = comb
                    dic['min_freq'] = min_fr
                    dic['accuracy'] = accuracies
                    #print(dic['punct'], dic['features'],dic['min_freq'], dic['accuracy'])
                    result.append(dic)

    results_sorted = sorted(result, key=lambda k: k['accuracy'], reverse=True)

    return results_sorted

def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    tuples = []

    s = set()
    # all keys from dictionary in results [punct, features, min_freq, accuracy]
    keys = list(results[0].keys())
    for dicc in results:
      for key in keys[:-1]:
        s.add(dicc[key])
    settings = list(s)

    for sett in settings:
      accu = 0
      n = 0
      key2 = ''
      for key in keys[:-1]:
        for dictionary in results:
          if dictionary[key] == sett:
            key2 = key
            accu += dictionary['accuracy']
            n += 1
      if hasattr(sett, '__len__'):
        sett2 = []
        for item in sett:
          sett2.append(item.__name__)
        sett = ' '.join(str(e) for e in sett2)
      tuples.append((accu/n, str(key2)+'='+str(sett)))
    return sorted(tuples, key=lambda x: x[0], reverse=True)

def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    new_docs = []
    for doc in docs:
        words = tokenize(doc, best_result['punct'])
        new_docs.append(words)

    X, vocab = vectorize(
        new_docs, best_result['features'], best_result['min_freq'])

    model = LogisticRegression()
    clf = model.fit(X, labels)
    return clf, vocab


def parse_test_data(best_result, vocab, test_data):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO

    new_docs = []
    docs, labels = read_data(test_data)  # read test data
    for doc in docs:
        words = tokenize(doc, best_result['punct'])
        new_docs.append(words)
    X, vocab = vectorize(new_docs, best_result['features'], best_result['min_freq'], vocab)

    return docs, labels, X


def parse_test_data2(best_result, vocab, tweets):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO

    new_docs = []
    for doc in tweets:
        words = tokenize(doc, best_result['punct'])
        new_docs.append(words)
    X, vocab = vectorize(
        new_docs, best_result['features'], best_result['min_freq'], vocab)

    return  X

def read_data(data):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    data.iloc[:, 0].values
    labels = data.iloc[:, 0].values
    docs = data.iloc[:, 1].values

    return docs, labels


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def divide_data(data):
    training_data, test_data = train_test_split(
        data, test_size=0.3, random_state=10)
    return training_data, test_data

def classify_tweets(tweets,label):
     all_Tweets=pd.DataFrame({'Tweet': tweets, 'Label': label})
     classify_results = {}
     classify_results['pos'] = all_Tweets.loc[all_Tweets['Label']==1]
     classify_results['neg'] = all_Tweets.loc[all_Tweets['Label'] == -1]
     classify_results['neutral'] = all_Tweets.loc[all_Tweets['Label'] == 0]

     return classify_results

def divide_data2(pos,neg,neutra):
    possitive=[]
    negative=[]
    neutral=[]
    for p in pos:
        possitive.append([1,p])
    for n in neg:
        negative.append([-1,n])
    for nn in neutra:
        neutral.append([0,nn])
    
    train_pos, test_pos = divide_data(possitive)
    train_neg, test_neg = divide_data(negative)
    train_neutral, test_neutral = divide_data(neutral)

    train= train_pos + train_neg + train_neutral
    test= test_pos + test_neg + test_neutral

    trainD = pd.DataFrame(train, columns=['Sentiment', 'SentimentText'])
    testD = pd.DataFrame(test, columns=['Sentiment', 'SentimentText'])
    return trainD, testD

    
def main():
    #data = pd.read_csv('data.csv', sep=',')
    #data = data.drop(columns=['ItemID'])

    pos = list(pd.read_csv('./data/processedPositive.csv', sep=',').columns.values)
    neg = list(pd.read_csv('./data/processedNegative.csv', sep=',').columns.values)
    neutral = list(pd.read_csv('./data/processedNeutral.csv',sep=',').columns.values)


    print('Read data for making classification model')
    
    #training_data, test_data = divide_data(data[:5000])

    training_data, test_data = divide_data2(pos,neg,neutral)


    feature_fns = [token_features, token_pair_features, lexicon_features]
    #transform the training into np.array for labels and documents
    docs, labels = read_data(training_data)

    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2, 5, 10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab, test_data)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('\ntesting accuracy=%f\n' % accuracy_score(test_labels, predictions))

    tweets = get_tweets('tweets')
    print('Got tweets.')

    # Evaluate dowloaded tweets set.
    X_test2 = parse_test_data2(best_result, vocab, tweets)
    predictions_tweets = clf.predict(X_test2)

    classified_results = classify_tweets(tweets, predictions_tweets)

    print('Got sentiment for all tweets.')
    print('There are %d positive tweets, %d negative tweets and %d neutral tweets.' %
          (len(classified_results['pos']), len(classified_results['neg']), len(classified_results['neutral'])))

    save_obj(classified_results, 'classified_results')
    print('Classified_results saved.')


if __name__ == '__main__':
    main()
