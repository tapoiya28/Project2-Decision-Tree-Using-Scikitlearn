import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def prepare_splits(feature, target, splits=[(0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]):
    # This function is used for 2.1 Preparing the datasets (You should read 2.1 to understand easily)
    # Argument:
        # feature: feature subset
        # + target: label subset
        # + splits: the proportions
    # Save data sets
    datasets = []

    # Take the proportions in "splits" for test and training sets required
    for train_size, test_size in splits:
        
        # feature_train: feature train
        # label_train: label train
        # feature_test: feature test
        # label_test: label test
        
        feature_train, feature_test, label_train, label_test = train_test_split(
            feature, target,
            train_size=train_size, 
            test_size=test_size,
            stratify=target, 
            shuffle=True, 
            random_state=42
        )
        
        # shuffle = true: shuffle the data
        # train_size and test_size: it is wut it is
        # stratify = true: make sure that the data will be splitted equally:
            # - Example: 70 for class 1 and 30 for class 2, we take the proportion 80/20. 
            # - Then it could be 70 class 1 + 10 class 2 for training (80) and 20 class 2 remaining for test *20)
            # -> Data for test only class 2 remains. 
        # random-state = 42: like seed in minecraft: 
            # - It stands for the initialization. If we set it to 42, it will be same the initialization. 
            # -> Random but same result bc same seed.
            
        datasets.append((feature_train, feature_test, label_train, label_test))
    return datasets

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

def evaluate_model(clf, feature_test, label_test, class_names, title=""):
    # Agrument:
        # + clf: trained model of decision tree
        # + feature_test: columns that are not target used to test
        # + label_test: column that is target used to test
        
    # Using the model input (clf) to predict label for feature_test
    label_pred = clf.predict(feature_test)
    
    # generate a report using classification_report
    # print classification_report
    print(classification_report(label_test, label_pred, target_names=class_names))

    # generate a report using confusion_matrix
    cm = confusion_matrix(label_test, label_pred)
    
    # display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # set the blue color for confusion_matrix
    disp.plot(cmap=plt.cm.Blues)
    # set the titlte for the confusion_matrix
    plt.title(f"Confusion Matrix {title}")
    # no set grid
    plt.grid(False)
    # show the chart
    plt.show()

def evaluate_all(clfs, datasets, class_names):
    # 2.3 Evaluating the decision tree classifiers
    
    #Argument:
        # + clf: list of models of decision tree
        # feature_names: list of feature names
        # class_names: Disease or No disease
    for i, (clf, (feature_train, feature_test, label_train, label_test)) in enumerate(zip(clfs, datasets)):
        # recalculate % train
        percent_train = len(feature_train) * 100 / (len(feature_train) + len(feature_test))
        # print the title
        print(f"📊 Evaluation Tree {i+1}: ({percent_train:.1f}% train)")
        # call the evaluate_model to draw confusion_matrix
        evaluate_model(clf, feature_test, label_test, class_names, title=f"(Split {i+1})")
