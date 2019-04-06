from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron,SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from mylib.legal_reader import *

from sklearn.preprocessing import MultiLabelBinarizer
import score

def read_data(path, limit=-1):
    ldr = LegalDatasetReader()
    X_sents = []
    y = []

    num = 0
    for inst in ldr._read(path):
        # each inst has graf (TextField) and label (MultiLabelField)
        graf = inst["graf"]
        labels = inst["label"]
        tokens = [tok.text for tok in graf]
        X_sents.append(" ".join(tokens))
            
        y.append(labels.labels)
        
        num += 1
        if limit > -1 and num > limit:
            break
    print(path, "read {} examples".format(num))
    
    return X_sents,y
    

def prepare_data(trainpath, testpath, mlb):

    X_train_sents, y_train = read_data(trainpath)
    X_test_sents, y_test = read_data(testpath)

    # use the max_features here so that the feature space is shared across train and test.
    #vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', ngram_range=(1,3))
    vectorizer = CountVectorizer(stop_words='english', max_df=0.5, ngram_range=(1,3))

    X_train = vectorizer.fit_transform(X_train_sents)
    X_test = vectorizer.transform(X_test_sents)

    print(X_train.shape)
    print(X_test.shape)
    
    y_train_array = mlb.fit_transform(y_train)
    print("num classes", len(mlb.classes_))
    print(mlb.classes_)
    y_test_array = mlb.transform(y_test)

    return X_train,y_train_array,X_test,y_test_array,X_test_sents



if __name__ == "__main__":

    mlb = MultiLabelBinarizer()

    # also consider data/all_remove.train
    outfile = "results/linear.txt"
    X_train,y_train,X_test,y_test,test_sents = prepare_data("data/all.train", "data/validation/all_validation", mlb)
    
    #clf = LogisticRegression(verbose=1, solver="sag", class_weight={0:0.1})
    
    clf = SGDClassifier(verbose=1, n_jobs=10, loss="log", class_weight={0:0.1})
    
    multi_clf = MultiOutputClassifier(clf, n_jobs=1)
    multi_clf.fit(X_train, y_train)
    
    preds = multi_clf.predict(X_test)
    print(preds.shape)

    y_preds = mlb.inverse_transform(preds)
    print(y_preds)
    
    with open(outfile, "w") as out:
        for sent,pred in zip(test_sents, y_preds):
            pred = list(pred)
            if len(pred) == 0:
                pred = ["unmatched"]
            if len(pred) > 1 and "unmatched" in pred:
                pred.remove("unmatched")
            print("{}\t{}".format(sent,",".join(pred)), file=out)


    score.score("data/validation/all_validation", outfile)
    print("Results written to", outfile)
