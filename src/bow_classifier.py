"""A baseline Bag-of-Words text classification.
Usage: python3 classify.py <train.txt> <test.txt> [--svm] [--tfidf] [--bigrams] [--test_list]
train.txt and test.txt should contain one "document" per line,
first token should be the label.
The default is to use regularized Logistic Regression and relative frequencies.
Pass --svm to use Linear SVM instead.
Pass --tfidf to use tf-idf instead of relative frequencies.
Pass --bigrams to use bigrams instead of unigrams.
Pass --test subset list
"""
import sys
import getopt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def readcorpus(corpusfile, lab, use_list_test=False):
	documents = []
	labels = []
	cnt = 0
	dict_w = {'dom0': 0, 'dom1':1, 'dom2':2}
	with open(corpusfile, encoding='utf8') as inp, open(lab, encoding='utf8') as trg:
		for line, trg in zip(inp, trg):
			doc = line.strip() #.split()
			documents.append(doc)
			label = dict_w[trg[:-1]]
			labels.append(label)
	return documents, labels


def main():
	# Command line interface
	try:
		opts, args = getopt.gnu_getopt(
				sys.argv[1:], '', ['svm', 'tfidf', 'bigrams'])
		opts = dict(opts)
		train, train_lab, test, test_lab = args
	except (getopt.GetoptError, IndexError, ValueError) as err:
		print(err)
		print(__doc__)
		return

	# read train and test corpus
	Xtrain, Ytrain = readcorpus(train, train_lab)
	use_list_test = '--test_list' in opts
	Xtest, Ytest = readcorpus(test, test_lab, use_list_test)

	# Bag-of-Words extraction
	vec = TfidfVectorizer(
			use_idf='--tfidf' in opts,
			ngram_range=(2, 2) if '--bigrams' in opts else (1, 1),
			lowercase=True,
			max_features=100000,
			binary=False)

	# choose classifier
	if '--svm' in opts:
		# With LinearSVC you have to specify the regularization parameter C
		clf = LinearSVC(C=1.0)
	else:
		# LogisticRegressionCV automatically picks the best regularization
		# parameter using cross validation.
		clf = LogisticRegressionCV(
				cv=3,
				class_weight='balanced',
				max_iter=500)

	# combine the vectorizer with a classifier
	classifier = Pipeline([
			('vec', vec),
			('clf', clf)])

	# train the classifier
	classifier.fit(Xtrain[::50], Ytrain[::50])

	# make predictions on test set
	Yguess = classifier.predict(Xtest)

	# evaluate
	print('confusion matrix:\n', confusion_matrix(Ytest, Yguess))
	print(classification_report(Ytest, Yguess))


if __name__ == '__main__':
	main()
