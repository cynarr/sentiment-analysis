#!/usr/bin/env python
# This code implements sentiment classification using the Scikit-learn machine learning toolkit for Python:
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

# Sentiment classification of parallel sentence data in English (original), Finnish, French, or Italian (translations).

# Run this script to classify the data using:
# - Pre-compiled training and testing sets (90%/10%)
# - Stratified 10-fold cross-validation
# - Scikit-learn train_test_split (90%/10%)

# The data is classified using the following classifiers:
# - Multinomial Naïve Bayes
# - Logistic Regression
# - Linear SVC
# - Multilayer Perceptron

# Usage: python3 classify.py <LANG> <DIMENSIONS>
# Arguments:
# - <LANG>: en / fi / fr / it
# - <DIMENSIONS>: bin / multi
#    - bin: positive/negative
#    - multi: 8-class classification into classes: anger/anticipation/disgust/fear/joy/sadness/surprise/trust (Plutchik's Eight)

import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import re


def load_data(tagfile, goldfile):

    # Returns numpy arrays containing the labels of the training set
    # and the testing set (gold labels)

    tags = []
    with open(tagfile, 'r') as f:
        taglines = f.readlines()
        for l in taglines:
            tags.append(l)
    f.close()

    gold = []
    with open(goldfile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            gold.append(l)
    f.close()

    tags = np.array(tags)
    gold = np.array(gold)

    return tags, gold


def vectorize(trainfile, testfile):

    # Returns count-vectorized representation of the training and testing data

    vectorizer = CountVectorizer(analyzer='word')

    trainset = vectorizer.fit_transform(codecs.open(trainfile, 'r', 'utf8'))
    testset = vectorizer.transform(codecs.open(testfile, 'r', 'utf8'))

    return trainset, testset


def evaluate(predictions, test_tags):

    # Evaluate classification of pre-compiled test data
    # Uses Accuracy, F-measure, Precision, and Recall
    # Displays a confusion matrix

    print('Pre-compiled stratified training and testing data:')

    print('Accuracy: ', metrics.accuracy_score(test_tags, predictions))
    print(metrics.classification_report(test_tags, predictions))

    print('Confusion matrix for pre-compiled stratified testing data classification:')
    cm = confusion_matrix(test_tags, predictions)
    print(cm)


def stratified_cross_validate(model, dims, lang):

    # Function for classifying using stratified 10-fold cross-validation
    # Used for comparison and evaluation against own pre-compiled testset
    # Displays a confusion matrix

    print()
    print('Stratified 10-fold cross-validation:')

    vectorizer = CountVectorizer(analyzer='word')
    data = vectorizer.fit_transform(codecs.open(dims+'/crossval/'+lang+'/'+lang+'.txt', 'r', 'utf8'))

    cross_tags = []
    with open(dims+'/crossval/tags.txt', 'r') as f:
        taglines = f.readlines()
        for l in taglines:
            cross_tags.append(l)
    f.close()

    cross_tags = np.array(cross_tags)

    skf = StratifiedKFold(n_splits=10, shuffle=True)

    # Structure for storing evaluation scores in lists modified from https://gist.github.com/zacstewart/5978000
    prec = []
    rec = []
    f1 = []

    confusion = np.zeros(())

    if dims == 'bin':
        confusion = np.zeros((2, 2))
    elif dims == 'multi':
        confusion = np.zeros((8, 8))

    for train_index, test_index in skf.split(data, cross_tags):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = cross_tags[train_index], cross_tags[test_index]

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        y_tags = np.squeeze(model.predict(X_test))

        prec_score = precision_score(y_test, pred, average=None)
        rec_score = recall_score(y_test, pred, average=None)
        f_measure = f1_score(y_test, pred, average=None)

        prec.append(prec_score)
        rec.append(rec_score)
        f1.append(f_measure)

        cm = confusion_matrix(y_test, y_tags)
        confusion += cm

    # Avgs modified from https://github.com/sri-teja/chemical-NER/blob/master/kfold.py
    print('Average precision: ', sum(prec)/len(prec))
    print('Average recall: ', sum(rec)/len(rec))
    print('Average F1-score: ', sum(f1)/len(f1))

    print('Confusion matrix using stratified 10-fold cross-validation: ')
    print(confusion)

def traintestsplit(model, dims, lang):

    # Function for classifying using the in-built Scikit-learn train_test_split function
    # Used for comparison and evaluation against own pre-compiled testset

    vectorizer = CountVectorizer(analyzer='word')
    data = vectorizer.fit_transform(codecs.open(dims+'/traintest/'+lang+'/'+lang+'.txt', 'r', 'utf8'))

    testsplit_tags = []
    with open(dims+'/traintest/tags.txt', 'r') as f:
        taglines = f.readlines()
        for l in taglines:
            testsplit_tags.append(l)
    f.close()

    tags = np.array(testsplit_tags)

    X_train, X_test, y_train, y_test = train_test_split(data, tags, stratify=tags, test_size=0.1, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print()
    print('Accuracy using the Scikit-learn train_test_split function:', metrics.accuracy_score(y_test, predictions))
    print(metrics.classification_report(y_test, predictions))


def classify(lang, dims):

    # Function to classify using pre-compiled training and test sets

    tags, gold = load_data(dims+'/traintest/gold-train.txt', dims+'/traintest/gold-test.txt')
    trainset, testset = vectorize(dims+'/traintest/'+lang+'/'+lang+'-train.txt', dims+'/traintest/'+lang+'/'+lang+'-test.txt')

    # Classifier  models
    models = {'Multinomial Naïve Bayes': MultinomialNB(),
              'Logistic Regression': linear_model.LogisticRegression(random_state=0, solver='liblinear'),
              'Linear SVC': LinearSVC(random_state=0), 'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(100, 100, 100),
                        max_iter=500, alpha=0.0001, solver='adam', verbose=10, random_state=21, tol=0.000000001,
                        activation='relu', batch_size='auto')}

    for k, v in models.items():
        print('***', k, '***')
        if k is 'Multilayer Perceptron':
            scaler = StandardScaler(with_mean=False) # Scale dataset for MLP model
            scaler.fit(trainset)

            trainset = scaler.transform(trainset)
            testset = scaler.transform(testset)
        else:
            pass
        model = v.fit(trainset, tags)

        pred = model.predict(testset)
        evaluate(pred, gold)

        stratified_cross_validate(model, dims, lang) # Use stratified cross-validation
        traintestsplit(model, dims, lang) # Use the in-built Scikit-learn train_test_split function


def main():

    import argparse
    parser = ArgumentParser()

    # Code borrowed and modified from https://github.com/cynarr/sentimentator/blob/master/data_import.py
    def check_lang(l, pattern=re.compile(r'^[a-zA-Z]{2}$')):
        if not pattern.match(l):
            raise argparse.ArgumentTypeError('Use a lowercase two-character alphabetic language code. Available codes: en, fi, fr, it.')
        return l

    def check_dim(d):
        dims = ['multi', 'bin']
        if d not in dims:
            raise argparse.ArgumentTypeError('Use "multi" or "bin" for the classification type.')
        return d

    parser.add_argument('LANG', help='', type=check_lang)
    parser.add_argument('DIMS', help='', type=check_dim)

    args = parser.parse_args()

    classify(args.LANG, args.DIMS)


if __name__ == "__main__":
    main()
