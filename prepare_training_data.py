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

def prepare_splits(feature, target, splits=[(0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (0.9, 0.1)]):
    """
    Prepares the data by encoding categorical features and splitting into train/test sets.
    
    Arguments:
    - feature: Feature set (DataFrame)
    - target: Target labels (Series or array)
    - splits: Proportions for train/test split (default: 40/60, 60/40, 80/20, 90/10)
    
    Returns:
    - datasets: List of tuples containing the training and testing data (features and labels)
    """
    
    # Encode categorical features
    for column in feature.select_dtypes(include=['object']).columns:  # Identify categorical columns
        feature[column] = label_encoder.fit_transform(feature[column])  # Apply LabelEncoder to each column

    # Initialize list to hold datasets for each split
    datasets = []

    # Split the data based on the proportions defined in 'splits'
    for train_size, test_size in splits:
        feature_train, feature_test, label_train, label_test = train_test_split(
            feature, 
            target,
            train_size=train_size, 
            test_size=test_size,
            stratify=target, 
            shuffle=True, 
            random_state=42
        )
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
