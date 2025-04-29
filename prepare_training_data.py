import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()


def train_model(feature_train, label_train):
    # Arguments:
        # + feature_train: columns those are not target used to train
        # + label_train: column that is target used to train
    
    # Create model of Decision Tree
            # + criterion = 'entropy': Determine the function to evaluate the quality of splitting
            # + random_state=42: Ensures the same training results every time we run it again 
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    # Method training
    # -> It will be trained based on those informations
    
    clf.fit(feature_train, label_train)
    return clf

def train_all_models(datasets):
    # Argument:
        # + datasets: Data after splitting
    # Save trained model of decision trees
    clfs = []
    
    for (feature_train, _, label_train, _) in datasets:
        # call to train_model function to get decision tree
        clf = train_model(feature_train, label_train)
        # add it to a list
        clfs.append(clf)
    return clfs



