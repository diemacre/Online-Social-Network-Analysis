# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
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
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


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
        #strings = re.findall(r'[a-zA-Z0-9]+[^a-zA-Z0-9]?[a-zA-Z0-9]+', doc.lower())
        strings = re.findall(r"[\w'-]+", doc.lower())
        for st in strings:
          if st=="'" or st=='-':
            strings.remove(st)
        #strings = re.findall(r'\w+[^\w\s]?\w+', doc.lower())
        #strings = re.sub(r'[?|$|.|!|,|:|;|-|[|]|{|}]', r' ', doc.lower()).split()
    else:
        strings = re.sub(r'\W+', r' ', doc.lower()).split()
        #strings = re.findall(r'\w+', doc.lower())
    #strings= removeStopWords(strings)
    tokens = np.array(strings)
    return tokens
    '''
    doc = ''.join([x for x in doc if x in string.ascii_letters + string.digits+ '_' '\'- '])
    if not doc:
        return []
    tokens = []
    if keep_internal_punct:
        tokens = doc.lower().split()
    else:
        tokens = re.sub('\W+', ' ', doc).lower().split()
    return tokens
    '''
    
    
stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def removeStopWords(strings):
  for token in strings:
    if token in stop_words:
      strings.remove(token)
  return strings

'''
print('\ntokenize():')
print(tokenize(" Hi there! Isn't this' (_) - h-fun?", keep_internal_punct=False))
print(tokenize("Hi there! Isn't this' (_)  - h-fun? ", keep_internal_punct=True))
'''

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


'''
print('\ntoken_features():')
feats = defaultdict(lambda: 0)
token_features(['hi', 'there', 'hi'], feats)
print(sorted(feats.items()))
'''

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


def token_trios_features(tokens, feats, k=4):
    for i in range(0, len(tokens)-k+1):
      comb = list(combinations(tokens[i:i+k], 3))
      for item in comb:
        feats['token_pair='+item[0]+'__'+item[1]+'__'+item[2]] += 1
    return
'''
print('\ntoken_pair_features():')
feats = defaultdict(lambda: 0)
token_pair_features(np.array(['a', 'b', 'c','d']), feats)
print(sorted(feats.items()))
'''
neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

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
    #feats['stop_words'] = 0
    for token in tokens:
        if token.lower() in neg_words:
            feats['neg_words'] += 1
        elif token.lower() in pos_words:
            feats['pos_words'] += 1
        #elif token.lower() in stop_words:
        #    feats['stop_words'] += 1
    return

'''
print('\nlexicon_features():')
feats = defaultdict(lambda: 0)
lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
print(sorted(feats.items()))
'''
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

'''
print('\nfeaturize():')
feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
print(feats)
'''
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
    rows=[]
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
    aux=True
    if vocab==None:
      aux=False
      vocab = defaultdict(int)
      for i in range(0, len(filt_tokens)):
          vocab[filt_tokens[i]] = i


    for i in range(0, len(tokens_list)):  # for each string in the document
      feats = featurize(tokens_list[i], feature_fns) 
      #add to totalfeats all diferent feats od each doc
      if aux:
        #if there is a token on training vocab that is not on the test, we add that token to the feasts of test
        if i==len(tokens_list)-1:#just in the last loop so it is faster (do not need to do it for each doc)
          for token in vocab.keys():
              if token not in filt_tokens:
                feats.append((token, 0)) #value 0 beacuse it is not on the features of test
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
'''
print('\nvectorize():')
docs = ["Isn't this movie great?", "Horrible, horrible movie"]
tokens_list = [tokenize(d) for d in docs]
feature_fns = [token_features]
X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
print(type(X))
print(X.toarray())
print(sorted(vocab.items(), key=lambda x: x[1]))
'''

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


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
                    accuracies = cross_validation_accuracy(model, matr, labels, 5)
                    dic['punct'] = punc
                    dic['features'] = comb
                    dic['min_freq'] = min_fr
                    dic['accuracy'] = accuracies
                    #print(dic['punct'], dic['features'],dic['min_freq'], dic['accuracy'])
                    result.append(dic)

    results_sorted = sorted(result, key=lambda k: k['accuracy'], reverse=True)

    return results_sorted


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    x = [result['accuracy'] for result in results]
    x = x[::-1] #change orther to ascending
    y = range(0, len(x))

    plt.plot(y, x)
    plt.title("Accuracies Plot")
    plt.xlabel("setting")
    plt.ylabel("accuracy")

    plt.savefig('accuracies.png')

    return

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
    tuples=[]

    s = set()
    keys = list(results[0].keys()) #all keys from dictionary in results [punct, features, min_freq, accuracy]
    for dicc in results:
      for key in keys[:-1]:
        s.add(dicc[key])
    settings= list(s)

    for sett in settings:
      accu = 0
      n = 0
      key2 = ''
      for key in keys[:-1]:
        for dictionary in results:
          if dictionary[key]==sett:
            key2 = key
            accu+=dictionary['accuracy']
            n+=1
      if hasattr(sett, '__len__'):
        sett2=[]
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

    X, vocab = vectorize(new_docs, best_result['features'], best_result['min_freq'])

    model = LogisticRegression()
    clf = model.fit(X, labels)
    return clf, vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    coefs = []
    coef = clf.coef_[0]
    top_coef_terms = []
    l = n

    if label:
        top_coef_ind = np.argsort(coef)[::-1][:l]
    else:
        top_coef_ind = np.argsort(coef)[:l]

    for i in top_coef_ind:
        for key, value in vocab.items():
            if i == value:
                top_coef_terms.append(key)

    top_coef = coef[top_coef_ind]

    for i in range(len(top_coef)):
      coefs.append((top_coef_terms[i], top_coef[i]))
    return coefs


def parse_test_data(best_result, vocab):
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
    docs, labels = read_data(os.path.join('data', 'test')) #read test data
    for doc in docs:
        words = tokenize(doc, best_result['punct'])
        new_docs.append(words)
    X, vocab = vectorize(new_docs, best_result['features'], best_result['min_freq'], vocab)

    return docs, labels, X


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    #print(probabilities)
    #print('[0][1]', probabilities[0][1])
    #print('[0][1]', probabilities[0][1])
    p=[]
    for i in range(len(predictions)):
      if predictions[i]!=test_labels[i]:
        p.append((test_labels[i], predictions[i], max(probabilities[i]), test_docs[i]))
    p_sorted = sorted(p, key=lambda x: x[2], reverse=True)[:n]

    for doc in p_sorted:
      print('truth='+str(doc[0])+' predicted='+str(doc[1])+' proba='+str(doc[2]))
      print(str(doc[3])+'\n')
    


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    print('\n\n')
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('\ntesting accuracy=%f' % accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    

if __name__ == '__main__':
    main()
