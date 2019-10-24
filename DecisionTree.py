import sys
import math
import pandas as pd



class Node(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None


# Calculate info content (entropy) of the test data
def info_entropy(df, predict_attr):
    # Dataframe and number of positive/negatives examples in the data
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    p = float(p_df.shape[0])
    n = float(n_df.shape[0])
    # Calculate entropy
    if p == 0 or n == 0:
        I = 0
    else:
        I = ((-1 * p) / (p + n)) * math.log(p / (p + n), 2) + ((-1 * n) / (p + n)) * math.log(n / (p + n), 2)
    return I


# Calculates the weighted average of the entropy after an attribute test
def remainder(df, df_subsets, predict_attr):
    # number of test data
    num_data = df.shape[0]
    remainder = float(0)
    for df_sub in df_subsets:
        if df_sub.shape[0] > 1:
            remainder += float(df_sub.shape[0] / num_data) * info_entropy(df_sub, predict_attr)
    return remainder


# Calculates the information gain from the attribute test based on a given threshold
# Note: thresholds can change for the same attribute over time
def info_gain(df, attribute, predict_attr, threshold):
    sub_1 = df[df[attribute] == threshold]
    sub_2 = df[df[attribute] != threshold]
    # Determine information content, and subract remainder of attributes from it
    ig = info_entropy(df, predict_attr) - remainder(df, [sub_1, sub_2], predict_attr)
    return ig


# Returns the number of positive and negative data
def num_class(df, predict_attr):
    p_df = df[df[predict_attr] == 1]
    n_df = df[df[predict_attr] == 0]
    print("\n\nFinding Positive and Negative Outcomes from Data Set...")
    print("Positive=",p_df.shape[0], "Negative=",n_df.shape[0])
    return p_df.shape[0], n_df.shape[0]


# Chooses the attribute and its threshold with the highest info gain
# from the set of attributes
def choose_attr(df, attributes, predict_attr):
    max_info_gain = float("-inf")
    best_attr = None
    threshold = 0
    print("\n\nFinding Best Attribute...")
    # Test each attribute (note attributes maybe be chosen more than once)
    for attr in attributes:
        # Convert dataframe column to a list and round each value
        values = df[attr].tolist()
        values = [float(x) for x in values]
        # Remove duplicate values by converting the list to a set, then sort the set
        values = set(values)
        values = list(values)
        values.sort()
        print("Checking attribute:",attr)
        for value in values:
            ig = info_gain(df, attr, predict_attr, value)
            #print("For Threshold of",value," IG=",ig)
            if ig > max_info_gain:
                max_info_gain = ig
                best_attr = attr
                threshold = value
                print("\nFound better IG...\nNew IG=",max_info_gain,"\nNew Best Attribute=",best_attr,"New Threshold=",threshold)
    print("\n\n\nFinal Best Attribut and Threshold:\n\nBest Attribute:",best_attr,"\nThreshold:",threshold)
    return best_attr, threshold


# Builds the Decision Tree based on training data, attributes to train on,
# and a prediction attribute
def build_tree(df, cols, predict_attr):
    # Get the number of positive and negative examples in the training data
    p, n = num_class(df, predict_attr)
    # If train data has all positive or all negative values
    # then we have reached the end of our tree
    if p == 0 or n == 0:
        # Create a leaf node indicating it's prediction
        leaf = Node(None, None)
        leaf.leaf = True
        if p > n:
            leaf.predict = 1
        else:
            leaf.predict = 0
        return leaf
    else:
        # Determine attribute and its threshold value with the highest
        # information gain
        best_attr, threshold = choose_attr(df, cols, predict_attr)
        # Create internal tree node based on attribute and it's threshold
        tree = Node(best_attr, threshold)
        sub_1 = df[df[best_attr] < threshold]
        sub_2 = df[df[best_attr] > threshold]
        # Recursively build left and right subtree
        tree.left = build_tree(sub_1, cols, predict_attr)
        tree.right = build_tree(sub_2, cols, predict_attr)
        return tree


# Given a instance of a training data, make a prediction of true or false
# based on the Decision Tree
# Assumes all data has been cleaned (i.e. no NULL data)
def predict(node, row_df):
    # If we are at a leaf node, return the prediction of the leaf node
    if node.leaf:
        return node.predict
    # Traverse left or right subtree based on instance's data
    if row_df[node.attr] == node.thres:
        return predict(node.left, row_df)
    elif row_df[node.attr] != node.thres:
        return predict(node.right, row_df)


# Given a set of data, make a prediction for each instance using the Decision Tree
def test_predictions(root, df):
    num_data = df.shape[0]
    num_correct = 0
    print("Testing...")
    for index, row in df.iterrows():
        prediction = predict(root, row)
        if prediction == row['Outcome']:
            num_correct += 1
    print("Correct Predictions=",num_correct)
    print("Wrong Prediction=",num_data-num_correct)
    return round(num_correct / num_data, 2)


def clean(csv_file_name):
    df = pd.read_csv(csv_file_name, header=None)
    df.columns = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',  'OperatingSystems',
                  'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Outcome']
    cols = df.columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df


def main():
    # An example use of 'build_tree' and 'predict'
    df_train = clean('TrainingSet')
    attributes = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'OperatingSystems',
                  'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
    root = build_tree(df_train, attributes, 'Outcome')

    print("Accuracy of test data")
    df_test = clean('TestingSet')
    print(str(test_predictions(root, df_test) * 100.0) + '%')




if __name__ == '__main__':
    main()
