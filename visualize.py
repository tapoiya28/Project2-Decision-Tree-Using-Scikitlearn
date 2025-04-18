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
        clf, out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True, special_characters=True
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


def plot_class_distributions(datasets, splits, title):
    #Argument:
        # + datasets: For data after we split
        # + slipts: the proportions
        # + title: Name of the this data 
    
    # Draw the chart for test and train
    
    # For each pair of datasets: 
    for i, (__, __, label_train, label_test) in enumerate(datasets):
        
        # Create the figure with 2 charts in 1 line (1, 2), each chart has width: 10 inch, height 4 inch
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        # axs[0] mean the first chart (train)
        # x for the x axis. It takes lables from label_train (like 0,1)
        sns.countplot(x=label_train, ax=axs[0])
        
        # Set the name for the first chart
        axs[0].set_title(f"Train {int(splits[i][0]*100)}%")  
        
        # axs[1] mean the second chart (test)
        # It takes lables from y_tes
        sns.countplot(x=label_test, ax=axs[1]) 
        
        # Set the name for the second chart
        axs[1].set_title(f"Test {int(splits[i][1]*100)}%")
        
        for ax in axs:
            
            ax.set_xlabel("Target") # x axis presents Tagert (0,1)
            ax.set_ylabel("Count") # y axis presents Quantity 
            
            # True: show the grid to read easily
            # line: grid draw by "--"
            # alpha: opacity of the grid 
            ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.suptitle(title + f" {int(splits[i][0]*100)}/{int(splits[i][1]*100)}") 
        plt.tight_layout() # For the aesthetic
        plt.show()

def plot_original_distribution(target, title="Original Dataset"):
    # Draw the chart of the original data 
     
    sns.countplot(x=target,)   # x_axis count quantity of classes of target column (0, 1) 
    plt.title(title) # set title
    plt.xlabel("Label") # set x axis title
    plt.ylabel("Count") # set y axis title
    plt.grid(True, linestyle="--", alpha=0.5) # Same
    plt.show()

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