import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display

sns.set(style="whitegrid")

def visualize_tree(clf, feature_names, class_names):
    #Argument:
        # + clf: model decision tree
        # feature_names: list of feature names
        # class_names: Disease or No disease
        
    # Convert the decision tree after training to .dot (graph description text format). For drawing decision trees.
            # + clf: Decision tree model
            # + out_file = None: Instead of writting to file, it returns the description string
            # + filled=True: fill color
            # + rounded=True: rounded shapes
            # + special_characters: allow using the special_characters in labels or features
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, 
        rounded=True, 
        special_characters=True
    )
    # Initialize the graph by the above description string
    graph = graphviz.Source(dot_data)
    display(graph)



def visualize_all(clfs, feature_names, class_names):
    # 2.2 Building the decision tree classifiers

    #Argument:
        # + clf: list of models of decision tree
        # feature_names: list of feature names
        # class_names: Disease or No disease
    for i, clf in enumerate(clfs):
        print(f"🌲 Tree {i+1}")
        visualize_tree(clf, feature_names, class_names)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator




def analyze_accuracy_vs_depth(dataset, feature_names, class_names, max_depth_values=[None, 2, 3, 4, 5, 6, 7]):
    
    
    
    #2.4 The depth and accuracy of a decision tree

    
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    # Take the data 80/20
    X_train, X_test, y_train, y_test = dataset
    # Save the result
    results = []

    # For each depth [None, 2, 3, 4, 5, 6, 7]
    for depth in max_depth_values:
        
        # Set up for the decision tree but we add one more variable max_depth
        # Re-trained with limited depth requirement
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate by take the right predicts divide to total predicts
        acc = accuracy_score(y_test, y_pred)
        
        # Save the current depth
        results.append(("None" if depth is None else depth, acc))
        # Convert to string
        dot_data = export_graphviz(
            clf, out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True, rounded=True, special_characters=True
        )
        # Create graph based on string
        graph = graphviz.Source(dot_data)
        print(f"🌲 Decision Tree with max_depth = {depth}, Accuracy = {acc:.4f}")
        display(graph)

    # For report the accuracy_score (on the test set) of the decision tree classifier for each value of the max_depth parameter.
    return pd.DataFrame(results, columns=["max_depth", "accuracy"]) 



def plot_accuracy_vs_depth(results, title="Accuracy vs. Tree Depth"):
  
    
    
    df = results.copy() 
    # 
    df["numeric_depth"] = df["max_depth"].apply(lambda d: 10 if d=="None" or d is None else d)
    
   
    plt.figure()
    plt.plot(df["numeric_depth"], df["accuracy"], marker='o')

    xticks = df["numeric_depth"].tolist()
    xlabels = df["max_depth"].tolist()
    plt.xticks(xticks, xlabels)
    
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.show()