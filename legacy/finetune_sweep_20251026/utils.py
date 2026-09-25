import torch
import numpy as np
import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import tensorflow as tf

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

setup_seed(30)

print("UTILS Recurrence")

"""
Compute all metrics
"""
### CONFUSION MATRIX AND METRICS #####
    # Results
def compute_all_metrics(y_test, predictions, dataset, time_, subdirectory, metric_results_, split='test'):
    one_hot_predictions = predictions.argmax(1)
    n_classes = len(np.unique(y_test))
    LABELS = get_class_names(dataset)
    cms = []

    print("")
    from sklearn import metrics
    precision = metrics.precision_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    accuracy = metrics.accuracy_score(y_test, one_hot_predictions)
    recall = metrics.recall_score(y_test, one_hot_predictions, average="weighted")
    f1_score = metrics.f1_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_test, one_hot_predictions) 
    kappa = cohen_kappa_score(y_test, one_hot_predictions)
    # train, train_bal, f1_score_train_we, _, kappa_train
    metric_results_.append(accuracy)
    metric_results_.append(balanced_accuracy_score)
    metric_results_.append(f1_score)
    metric_results_.append(f1_score)
    metric_results_.append(kappa)


    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("f1_score: {:.4f}".format(f1_score))
    print("balanced_accuracy_score: {:.4f}".format(balanced_accuracy_score))

    mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
    with open(subdirectory+"/metrics_"+split+"_"+dataset+"_"+time_+".txt", 'w') as file:
        file.write("Dataset: {}".format( dataset))
        file.write("\n")
        file.write("mAP score: {:.4f}\n".format(mAP))
        file.write("precision: {:.4f}\n".format(precision))
        file.write("recall: {:.4f}\n".format(recall))
        file.write("f1_score: {:.4f}\n".format(f1_score))
        file.write("balanced_accuracy_score: {:.4f}".format(balanced_accuracy_score))
        file.write("\n")
        file.write("\n")
        
        file.write("\nF1-score (None):")
        print("F1-score (None):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average=None)
        metr = np.round(metr, 4)
        print(metr)
        file.write(str(metr))
        
        file.write("\nF1-score (weighted):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="weighted")
        metr = np.round(metr, 4)
        file.write(str(metr))

        file.write("\nF1-score (macro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="macro")
        metr = np.round(metr, 4)
        file.write(str(metr))

        file.write("\nF1-score (micro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="micro")
        metr = np.round(metr, 4)
        file.write(str(metr))
        
        file.write("\nclassification_report:")
        report = classification_report(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes))
        file.write(report)

        # Compute sensitivity and specificity for each class
        # Step 1: Compute the confusion matrix
        cm = metrics.confusion_matrix(y_test, one_hot_predictions)
        metrics = {}

        # Initialize lists to store sensitivity and specificity for each class
        sensitivities = []
        specificities = []

        # Step 2: Calculate sensitivity and specificity for each class
        for i in range(len(cm)):
            TP = cm[i, i]  # True Positives for class i
            FN = np.sum(cm[i, :]) - TP  # False Negatives for class i
            FP = np.sum(cm[:, i]) - TP  # False Positives for class i
            TN = np.sum(cm) - (TP + FP + FN)  # True Negatives for class i

            # Sensitivity (Recall) for class i
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            # Reduce decimals digits
            sensitivity = np.round(sensitivity, 4)
            sensitivities.append(sensitivity)

            # Specificity for class i
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            specificity = np.round(specificity, 4)
            specificities.append(specificity)

            # Store the metrics in a dictionary
            metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

        # Step 3: Print results
        file.write("\n")    
        file.write(str(LABELS))
        file.write("\n")

        print("Métricas por clase (one-vs-all):")
        for cls, m in metrics.items():
            print(f"Clase {cls}: TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']}")
        file.write("\n")    
            
        file.write(f"\nSensitivity per class:, {sensitivities}")
        file.write(f"\nSpecificity per class:, {specificities}")
        print()
        print(f"Sensitivity per class:, {sensitivities}")
        print(f"Specificity per class:, {specificities}")
    
    confusion_matrix = cm
    cms.append(confusion_matrix)

    #print(confusion_matrix)
    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Plot Confusion Matrix:
    width = 6
    height = 6  
    axis_size = 10
    ticks_size = 8
    fontsize = 12
    title = "Confusion matrix of {} data. Balanced accuracy: {:.2f}%".format(split, balanced_accuracy_score*100)

    # Plotting the confusion matrix
    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # You can switch to grey if needed

    thresh = confusion_matrix.max() * 0.5

    # Iterate over data dimensions and create text annotations with custom coloring
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if i == j else "black",
                    fontsize=fontsize)

    # Adding labels, titles, and formatting the confusion matrix plot
    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, LABELS, rotation=45, fontsize=ticks_size)
    plt.yticks(tick_marks, LABELS, fontsize=ticks_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_size)
    plt.xlabel('Predicted label', fontsize=axis_size)
    plt.title(title, fontsize=axis_size)
    #plt.subplots_adjust(left=0.25, bottom=0.15, right=0.9, top=0.85)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    
    plt.savefig(subdirectory + '/CM_' + split + '_' + dataset+'_'+time_+'.png')
    

    # NORMALIZE
    path_fig = subdirectory + '/CMn_' + split + '_' + dataset+'_'+time_+'.png'
    normalised_confusion_matrix = normalised_confusion_matrix * 100

    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  # You can switch to grey if needed

    thresh = normalised_confusion_matrix.max() * 0.5

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, f"{normalised_confusion_matrix[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if i == j else "black",
                    fontsize=fontsize)

    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, LABELS, rotation=45, fontsize=ticks_size)
    plt.yticks(tick_marks, LABELS, fontsize=ticks_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_size)
    plt.xlabel('Predicted label', fontsize=axis_size)
    plt.title(title, fontsize=axis_size)
    # plt.subplots_adjust(left=0.25, bottom=0.15, right=0.9, top=0.85)
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    plt.savefig(path_fig)

    
    # ROC CURVE
    print("ROC curve")
    # Example: y_test and y_preds (1D tensors with values 0, 1, 2)
    #y_test = torch.tensor([0, 1, 2, 1, 0])  # Example true labels
    #y_preds = torch.tensor([0, 2, 1, 1, 0])  # Example predicted labels

    # Convert tensors to numpy arrays for scikit-learn compatibility
    y_preds = one_hot_predictions
    y_test_np = y_test #.numpy()
    y_preds_np = y_preds #.numpy()

    # np.save('y_test.npy', y_test)
    # np.save('y_preds.npy', y_preds)

    # One-hot encode the true labels and predicted labels (binarization)
    n_classes = 3
    y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2])
    y_preds_bin = label_binarize(y_preds_np, classes=[0, 1, 2])

    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    colors = ['blue', 'red', 'green']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label='ROC curve (area = {0:0.2f}) for class {1}'.format(roc_auc[i], i+1))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of proposed model')
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.25, bottom=0.15, right=0.9, top=0.85)

    path_fig = subdirectory + '/roc_' + dataset+'_'+time_+'.png'
    plt.savefig(path_fig)


    from sklearn.metrics import precision_recall_curve, average_precision_score
    # PCR CURVE
    # print("PCR curve")

    # #precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # # Save y_true and y_scores
    # # np.save('y_true.npy', y_true)
    # # np.save('y_scores.npy', y_scores)

    # # 1) Your true binary labels and model scores/probabilities:
    # #    y_true: array-like of shape (n_samples,)
    # #    y_scores: array-like of shape (n_samples,)
    # y_true   = y_test       # replace with your ground truth
    # y_scores = y_preds  # replace with your model’s scores

    # # Initialize lists to store precision, recall, and AP scores for each class
    # precision_list = []
    # recall_list = []
    # ap_list = []

    # # 2) Compute precision, recall, and thresholds for each class
    # for i in range(3):  # 3 classes in total
    #     precision, recall, thresholds = precision_recall_curve(y_true == i, y_scores[:, i])
    #     precision_list.append(precision)
    #     recall_list.append(recall)
        
    #     # Compute Average Precision (AP) for each class
    #     ap = average_precision_score(y_true == i, y_scores[:, i])
    #     ap_list.append(ap)

    # # 3) Plot the Precision-Recall curve for each class
    # plt.figure(figsize=(8, 6))

    # for i in range(3):  # For each class
    #     plt.plot(recall_list[i], precision_list[i], lw=2, label=f'Class {i+1} (AP = {ap_list[i]:.3f})')

    # # Customize the plot
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Multi-Class Precision–Recall Curve')
    # plt.legend(loc='lower left')
    # plt.grid(True)

    # # Save plot
    # path_fig = subdirectory + '/pcr_' + dataset + '_' + time_ + '.png'
    # print("Saved PCR curve in:", path_fig)
    # plt.tight_layout()
    # plt.savefig(path_fig)
    # plt.close()

"""
Balanced accuracy
"""
import tensorflow as tf

def balanced_accuracy(y_true, y_pred):
    # Convert one-hot encoded labels or categorical labels to integers if necessary
    y_true = tf.argmax(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
    y_pred = tf.argmax(y_pred, axis=-1) if len(y_pred.shape) > 1 else y_pred

    # Calculate the number of classes
    num_classes = tf.reduce_max(y_true) + 1
    
    # Ensure predictions are within the valid range
    y_pred = tf.clip_by_value(y_pred, 0, num_classes - 1)
    
    # Compute confusion matrix
    #cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=tf.reduce_max(y_true) + 1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

    # True positives are the diagonal elements, cast to float32
    true_positives = tf.cast(tf.linalg.diag_part(cm), tf.float32)
    
    # Compute the per-class count, ensuring it is float32
    per_class_count = tf.cast(tf.reduce_sum(cm, axis=1), tf.float32)
    
    # Calculate recall per class, avoiding division by zero
    recall_per_class = true_positives / (per_class_count + tf.keras.backend.epsilon())
    
    # Balanced accuracy is the mean of the recall values
    balanced_acc = tf.reduce_mean(recall_per_class)
    
    return balanced_acc





"""
Overlap intra-class and intra-user
"""
def overlap_data3(data, labels, data_test, shift=0.5): #(408620, 100, 6) (408620,)
    
    classes = np.unique(labels)
    # Get the indexes of each class
    indexes = {}
    indexes_users = {}
    users_unique = np.unique(data_test[0])
    users = np.array(data_test[0])

    labels_name = data_test[4]

    # Indexes of data_test[4]
    indexes_labels = {}
    for i, l in enumerate(labels_name):
        indexes_labels[l] = i
    
    indexes_labels_name = data_test.iloc[:, 4].index


    for u in users_unique:
        indexes_users[u] = np.where(users == u )[0]
    for c in classes:
        indexes[c] = np.where(labels == c )[0]      

    # Get the data of each user and class
    new_data_class = []
    new_labels = []
    new_labels_name = []
    for u in indexes_users.keys():
        for c in indexes.keys():
            # Common values between indexes_users and indexes
            index = np.intersect1d(indexes_users[u], indexes[c])
            samples = data[index]
            samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])
            #print("Clase:", c)
            new_data_class, n = overlap_class(new_data_class, samples, shift, window_size=data.shape[1])
            new_labels = new_labels + list(np.repeat(c, n))

            indexes_c = indexes[c]
            label_name = [labels_name[i] for i in indexes_c]
            idx_label_name = [indexes_labels_name[i] for i in indexes_c]
            if ( len(np.unique(label_name)) > 1):
                unique_labels = np.unique(label_name)

                print("Error: More than one label")
                exit()
            label_name = label_name[0]
            new_labels_name = new_labels_name + list(np.repeat(label_name, n))

    new_labels = np.array(new_labels)
    new_data_class = np.array(new_data_class)
    return new_data_class, new_labels, new_labels_name


"""
Overlap function
"""
def overlap_class( new_data_class, samples, overlap_shift, window_size):
    init = 0
    n = 0
    while(init <= (samples.shape[0]-window_size)):
        new_data_class.append(samples[init:init+window_size])
        #init = init + (window_size//2) # init + (window_size * overlap_shift) # ex. overlap_shift=0.8
        init = init + int(window_size - (window_size * overlap_shift))
        n = n + 1
    return new_data_class, n

"""
Overlap intra-class and intra-user
"""
def overlap_data(data, labels, users_data, shift=0.5): #(408620, 100, 6) (408620,)
    
    classes = np.unique(labels)
    # Get the indexes of each class
    indexes = {}
    indexes_users = {}
    users_unique = np.unique(users_data)
    users = np.array(users_data)

    for u in users_unique:
        indexes_users[u] = np.where(users == u )[0]
    for c in classes:
        indexes[c] = np.where(labels == c )[0]          

    # Get the data of each user and class
    new_data_class = []
    new_labels = []
    new_users = []
    for u in indexes_users.keys():
        for c in indexes.keys():
            # Common values between indexes_users and indexes
            index = np.intersect1d(indexes_users[u], indexes[c])
            samples = data[index]
            samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])  # (N, 9)
            #print("Clase:", c)
            new_data_class, n = overlap_class(new_data_class, samples, shift, window_size=data.shape[1])
            new_labels = new_labels + list(np.repeat(c, n))
            new_users = new_users + list(np.repeat(u, n))
            # Repeat useeeeers

    new_labels = np.array(new_labels)
    new_users = np.array(new_users)
    new_data_class = np.array(new_data_class)

    return new_data_class, new_labels, new_users



"""
Only y_labels and users
"""
def overlap_data2(data, labels, users_data, shift=0.5): #(408620, 100, 6) (408620,)
    
    classes = np.unique(labels)
    # Get the indexes of each class
    indexes = {}
    indexes_users = {}
    users_unique = np.unique(users_data)
    users = np.array(users_data)

    for u in users_unique:
        indexes_users[u] = np.where(users == u )[0]
    for c in classes:
        indexes[c] = np.where(labels == c )[0]          

    # Get the data of each user and class
    new_labels = []
    new_users = []
    for u in indexes_users.keys():
        for c in indexes.keys():
            # Common values between indexes_users and indexes
            index = np.intersect1d(indexes_users[u], indexes[c])
            samples = data[index]
            samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])  # (N, 9)
            #print("Clase:", c)
            #new_data_class, n = overlap_class(new_data_class, samples, shift, window_size=data.shape[1])
            init = 0
            n = 0
            window_size=data.shape[1]
            while(init <= (samples.shape[0]-window_size)):
                init = init + int(window_size - (window_size * shift))
                n = n + 1

            new_labels = new_labels + list(np.repeat(c, n))
            new_users = new_users + list(np.repeat(u, n))

    new_labels = np.array(new_labels)
    new_users = np.array(new_users)

    return new_labels, new_users



# Normalization with scaler function
def normalise_data(dict_arrays):
    # Data
    train_data = dict_arrays['x_train']
    val_data = dict_arrays['x_val']
    test_data = dict_arrays['x_test']

    # Normalization
    scaler = StandardScaler()

    # Fit the scaler on the training data only
    train_data_flat = train_data.reshape(-1, 9)  # Flatten to (num_samples*timesteps, 9)
    scaler.fit(train_data_flat)

    # Apply the normalization
    dict_arrays['x_train'] = scaler.transform(train_data_flat).reshape(train_data.shape)
    dict_arrays['x_val'] = scaler.transform(val_data.reshape(-1, 9)).reshape(val_data.shape)
    dict_arrays['x_test'] = scaler.transform(test_data.reshape(-1, 9)).reshape(test_data.shape)

    return dict_arrays


# Normalise
def normalise_all_zscore(data, mean, std):
    """
    Normalise data (Z-normalisation)    
    """

    return ((data - mean) / std)


# Get processed fold for 1 test user, 1 val user,  of any dataset
def get_processed_fold_valtestuser(df, dataset, subdirectory, sensors, fold, val_user, test_user):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    x_train, y_train, x_val, y_val, x_test, y_test, _, _, _ = split_data_val_valtestuser(df, file_users_split, val_user, test_user, test_size=0.2)

    if (sensors == 2):
        dict_arrays = get_data_arrays(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    elif (sensors == 3):
        dict_arrays = get_data_arrays_3sns(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    else:
        print("No sensors:")
        exit()

    train_X = dict_arrays['x_train']
    train_Y = dict_arrays['y_train']
    val_X = dict_arrays['x_val']
    val_Y = dict_arrays['y_val']
    test_X = dict_arrays['x_test']
    test_Y = dict_arrays['y_test']

    # %%
    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)
    
   
    #train_X, train_Y = overlap_data(train_X, train_Y)
    #val_X, val_Y = overlap_data(val_X, val_Y)
    #test_X, test_Y = overlap_data(test_X, test_Y)

    means = np.mean(train_X, axis=0)
    stds = np.std(train_X, axis=0)
    train_X = normalise(train_X, means, stds)

    #means = np.mean(val_X, axis=0)
    #stds = np.std(val_X, axis=0)
    val_X = normalise(val_X, means, stds)

    #means = np.mean(test_X, axis=0)
    #stds = np.std(test_X, axis=0)
    test_X = normalise(test_X, means, stds)

    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    
    return dict_arrays



# Get processed fold for 1 test user of any dataset
def get_processed_fold_testuser(df, dataset, subdirectory, sensors, fold, overlap_shift, test_user):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    x_train, y_train, x_val, y_val, x_test, y_test, _, _, _ = split_data_val_testuser(df, file_users_split, test_user, test_size=0.2)

    if (sensors == 2):
        dict_arrays = get_data_arrays(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    elif (sensors == 3):
        dict_arrays = get_data_arrays_3sns(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    else:
        print("No sensors:")
        exit()

    train_X = dict_arrays['x_train']
    train_Y = dict_arrays['y_train']
    val_X = dict_arrays['x_val']
    val_Y = dict_arrays['y_val']
    test_X = dict_arrays['x_test']
    test_Y = dict_arrays['y_test']

    # %%
    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)
    
   
    train_X, train_Y = overlap_data(train_X, train_Y, overlap_shift)
    val_X, val_Y = overlap_data(val_X, val_Y, overlap_shift)
    test_X, test_Y = overlap_data(test_X, test_Y, overlap_shift)

    means = np.mean(train_X, axis=0)
    stds = np.std(train_X, axis=0)
    train_X = normalise(train_X, means, stds)

    #means = np.mean(val_X, axis=0)
    #stds = np.std(val_X, axis=0)
    val_X = normalise(val_X, means, stds)

    #means = np.mean(test_X, axis=0)
    #stds = np.std(test_X, axis=0)
    test_X = normalise(test_X, means, stds)

    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    
    return dict_arrays

# Dictionary with pre-defined splits
dic_datasets = {
    'fixed_users' : {
            'train_users' : ['S1358', 'S1004','S1018','S1073','S1696','S1346','S1534','S1075','S1053',
                                'S1076','S1538','S1003','S1016','S1078','S1692','S1539','S1064','S1006',
                                'S1354','S1347','S1010','S1694','S1080','S1352','S1052','S1008','S1013',
                                'S1355','S1348','S1614','S1068','S1532','S1046','S1077','S1544','S1615',
                                'S1685','S1067','S1072','S1015','S1055','S1047','S1049','S1061','S1036',
                                'S1071','S1353','S1541','S1066','S1025','S1082','S1063','S1005','S1691',
                                'S1007','S1074','S1054','S1050','S1039','S1012','S1029','S1349'],
            'val_users' : ['S1345','S1533','S1351','S1084','S1356','S1060','S1695','S1687','S1062','S1350','S1693'],                
            'test_users' : ['S1002','S1001','S1081','S1009','S1056','S1672','S1069','S1079','S1057',
                                'S1500','S1359','S1537','S1686','S1070','S1542','S1543','S1536','S1044',
                                'S1048','S1045','S1065','S1535','S1540','S1684','S1360','S1562','S1688',
                                'S1659','S1689','S1011','S1357','S1690']
            },
    }


# Get processed fold
def get_processed_fold_deo(df, dataset, subdirectory, sensors, fold, 
                           train_users, val_users, test_users,
                           seg5, overlap = True, overlap_shift=0.5, fixed_splits=False):
    
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    dict_arrays = {}

    if (fixed_splits):
        # train_X = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/data_vivabem012_train.npy', allow_pickle=True)
        # train_Y = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/labels_vivabem012_train.npy', allow_pickle=True)
        # val_X = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/data_vivabem012_val.npy', allow_pickle=True)
        # val_Y = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/labels_vivabem012_val.npy', allow_pickle=True)
        # test_X = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/data_vivabem012_test.npy', allow_pickle=True)
        # test_Y = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/labels_vivabem012_test.npy', allow_pickle=True)

        # # Reshape to N,500, 9
        # train_X = np.reshape(train_X, (-1, 500, 9))
        # val_X = np.reshape(val_X, (-1, 500, 9))
        # test_X = np.reshape(test_X, (-1, 500, 9))

        # users_train = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_train.npy', allow_pickle=True)
        # users_val = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_val.npy', allow_pickle=True)
        # users_test = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_test.npy', allow_pickle=True)

        x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test, splits_txt = split_data_fixedusers(df, train_users, val_users, test_users)

        users_train = train_users
        users_val = val_users
        users_test = test_users

        train_X = x_train
        train_Y = y_train
        val_X = x_val
        val_Y = y_val
        test_X = x_test
        test_Y = y_test

        uuid_train = np.unique(users_train)
        uuid_val = np.unique(users_val)
        uuid_test = np.unique(users_test)

    else:
        x_train, y_train, x_val, y_val, x_test, y_test, y_train_all, y_val_all, y_test_all, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt = split_data_val(df, 
                                                                                                                                                      dataset, 
                                                                                                                                                      file_users_split, 
                                                                                                                                                      test_size=0.3,)

        #if (sensors == 2):
            # dict_arrays, users_train, users_val, users_test = get_data_arrays(x_train, y_train, data_train[0], 
            #                                     #x_val, y_val, data_val[0], 
            #                                     #x_test, y_test, data_test[0], 
            #                                     #dataset, seg5, shuffle_data = False, axis=True)
        #elif (sensors == 3):
        dict_arrays, users_train, users_val, users_test = get_data_arrays_3sns(x_train, y_train, data_train[0], 
                                                                                x_val, y_val, data_val[0],
                                                                                x_test, y_test, data_test[0], 
                                                                                dataset, seg5, shuffle_data = False, axis = True)
            
        #else:
            #print("No sensors:")
            #exit()

        train_X = dict_arrays['x_train']
        train_Y = dict_arrays['y_train']
        val_X = dict_arrays['x_val']
        val_Y = dict_arrays['y_val']
        test_X = dict_arrays['x_test']
        test_Y = dict_arrays['y_test']
    
    # Convert to float32
    y_test_all = np.array(y_test_all)
    y_val_all = np.array(y_val_all)
    y_train_all = np.array(y_train_all)

    y_test_all =  y_test_all.astype(np.float32)
    y_val_all =  y_val_all.astype(np.float32)
    y_train_all =  y_train_all.astype(np.float32)
    
    # Convert to float32
    train_X = train_X.astype(np.float32)
    train_Y = train_Y.astype(np.float32)
    val_X = val_X.astype(np.float32)
    val_Y = val_Y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_Y = test_Y.astype(np.float32)

    # Check and handle NaN and infinite values
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    train_Y = np.nan_to_num(train_Y, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_Y = np.nan_to_num(val_Y, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_Y = np.nan_to_num(test_Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Overlap data
    if ( overlap ):
        train_X, train_Y, train_users = overlap_data(train_X, train_Y, users_train, overlap_shift)
        val_X, val_Y, val_users = overlap_data(val_X, val_Y, users_val, overlap_shift)
        test_X, test_Y, test_users = overlap_data(test_X, test_Y, users_test,overlap_shift)
    
    if ( sensors == 2):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]
        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]
        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ, train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ, val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ, test_gyrX, test_gyrY, test_gyrZ), axis=2)

        #Save train_X wit np.save
        # np.save('dance_sweep_20users_2sns/train_X.npy', train_X)
        # np.save('dance_sweep_20users_2sns/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_2sns/val_X.npy', val_X)
        # np.save('dance_sweep_20users_2sns/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_2sns/test_X.npy', test_X)
        # np.save('dance_sweep_20users_2sns/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_2sns/users_train.npy', users_train)
        # np.save('dance_sweep_20users_2sns/users_val.npy', users_val)
        # np.save('dance_sweep_20users_2sns/users_test.npy', users_test)        

        print(" 2 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()

    elif ( sensors == 1):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)
    
    elif ( sensors == 0): # Gyroscope sensor

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_gyrX, test_gyrY, test_gyrZ), axis=2)

        # np.save('dance_sweep_20users_1sns_g/train_X.npy', train_X)
        # np.save('dance_sweep_20users_1sns_g/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_1sns_g/val_X.npy', val_X)
        # np.save('dance_sweep_20users_1sns_g/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_1sns_g/test_X.npy', test_X)
        # np.save('dance_sweep_20users_1sns_g/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_1sns_g/users_train.npy', users_train)
        # np.save('dance_sweep_20users_1sns_g/users_val.npy', users_val)
        # np.save('dance_sweep_20users_1sns_g/users_test.npy', users_test)        

        print(" 1 sensor gyr")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()
    
    elif ( sensors == -1): # Accelerometer sensor

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)

        # np.save('dance_sweep_20users_1sns_a/train_X.npy', train_X)
        # np.save('dance_sweep_20users_1sns_a/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_1sns_a/val_X.npy', val_X)
        # np.save('dance_sweep_20users_1sns_a/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_1sns_a/test_X.npy', test_X)
        # np.save('dance_sweep_20users_1sns_a/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_1sns_a/users_train.npy', users_train)
        # np.save('dance_sweep_20users_1sns_a/users_val.npy', users_val)
        # np.save('dance_sweep_20users_1sns_a/users_test.npy', users_test) 

        print(" 1 sensor acc")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)  
        exit()
    
    elif ( sensors == 3):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())

        # Save train_X wit np.save
        # np.save('dance_sweep_20users/train_X.npy', train_X)
        # np.save('dance_sweep_20users/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users/val_X.npy', val_X)
        # np.save('dance_sweep_20users/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users/test_X.npy', test_X)
        # np.save('dance_sweep_20users/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users/users_train.npy', users_train)
        # np.save('dance_sweep_20users/users_val.npy', users_val)
        # np.save('dance_sweep_20users/users_test.npy', users_test)         

        print(" 3 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)

    # %%
    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)   
    
    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    dict_arrays['uuid_train'] = uuid_train
    dict_arrays['uuid_val'] = uuid_val
    dict_arrays['uuid_test'] = uuid_test
    
    return dict_arrays

# Normalize by z-score per axis
def zscore_per_axis(df):
    """
    Apply z-score normalization per axis group (acc_x, acc_y, ..., mag_z)
    on a wide dataframe with repeated sensor axis order.
    """
    n_axes = 9  # acc_x ... mag_z
    n_repeats = df.shape[1] // n_axes
    
    normalized = df.copy()
    axis_names = ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z"]
    
    for i, axis in enumerate(axis_names):
        # Get all columns corresponding to this axis
        cols = [j for j in range(i, df.shape[1], n_axes)]
        
        # Extract values
        vals = df.iloc[:, cols]
        
        # z-score normalization: (x - mean) / std, computed across all columns of this axis
        mean = vals.values.mean()
        std  = vals.values.std()
        normalized.iloc[:, cols] = (vals - mean) / std
    
    return normalized

# Get processed fold
def get_processed_fold(df, dataset, modeltype, subdirectory, sensors, fold, seg5, normalize, overlap, overlap_shift, feat=9):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'

    if ( (dataset.startswith('de')) & (normalize == True)):
        # Normalise all df for an unsupervised task
        print("Normalizing ...")
        df_info = df.iloc[:,:5]
        df_data = df.iloc[:,5:]
        df_data = zscore_per_axis(df_data)

        #concatenate
        df = pd.concat([df_info, df_data], axis=1)


    x_train, y_train, x_val, y_val, x_test, y_test, y_train_all, y_val_all, y_test_all, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt = split_data_val(df, 
                                                                                                                                                  dataset, 
                                                                                                                                                  file_users_split, 
                                                                                                                                                  test_size=0.3)
    # Use all the activities on labels
    if (sensors == 2):
        dict_arrays = get_data_arrays(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    elif (sensors == 3):
        dict_arrays, users_train, users_val, users_test = get_data_arrays_3sns(x_train, y_train, y_train_all, data_train[0], 
                                                                x_val, y_val, y_val_all, data_val[0], 
                                                                x_test, y_test, y_test_all, data_test[0], 
                                                                dataset, seg5, feat=feat, shuffle_data = False, axis=True)
        
    else:
        print("No sensors:")
        exit()


    # Dictionary with pre-defined splits
    train_X = dict_arrays['x_train']
    train_Y = dict_arrays['y_train']
    val_X = dict_arrays['x_val']
    val_Y = dict_arrays['y_val']
    test_X = dict_arrays['x_test']
    test_Y = dict_arrays['y_test']
    y_test_all = dict_arrays['y_test_all']
    y_train_all = dict_arrays['y_train_all']
    y_val_all = dict_arrays['y_val_all']

    # # Convert to float32
    # y_test_all = np.array(y_test_all)
    # y_val_all = np.array(y_val_all)
    # y_train_all = np.array(y_train_all)



    # Convert to float32
    # train_X = train_X.astype(np.float32)
    # train_Y = train_Y.astype(np.float32)
    # val_X = val_X.astype(np.float32)
    # val_Y = val_Y.astype(np.float32)
    # test_X = test_X.astype(np.float32)
    # test_Y = test_Y.astype(np.float32)

    # Check and handle NaN and infinite values
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    train_Y = np.nan_to_num(train_Y, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_Y = np.nan_to_num(val_Y, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_Y = np.nan_to_num(test_Y, nan=0.0, posinf=0.0, neginf=0.0)

    y_test_all = np.nan_to_num(y_test_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_all = np.nan_to_num(y_val_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_all = np.nan_to_num(y_train_all, nan=0.0, posinf=0.0, neginf=0.0)

    # %%
    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)

    # Here the data has 13 classes    
    if ( overlap ):
        if (dataset.startswith('vivabem')):
            train_X, train_Y, train_users = overlap_data(train_X, train_Y, users_train, overlap_shift)
            val_X, val_Y, val_users = overlap_data(val_X, val_Y, users_val, overlap_shift)
            test_X, test_Y, test_users = overlap_data(test_X, test_Y, users_test, overlap_shift)
        
        elif (dataset.startswith('eat')):
            train_X, y_train_all, train_users = overlap_data(train_X, y_train_all, users_train, overlap_shift)
            val_X, y_val_all, val_users = overlap_data(val_X, y_val_all, users_val, overlap_shift)
            test_X, y_test_all, test_users = overlap_data(test_X, y_test_all, users_test, overlap_shift)

            test_Y = y_test_all.copy()
            val_Y = y_val_all.copy()
            train_Y = y_train_all.copy()

            test_Y = np.where(test_Y == 2, 0, np.where(test_Y == 4, 1, 2))
            val_Y = np.where(val_Y == 2, 0, np.where(val_Y == 4, 1, 2))
            train_Y = np.where(train_Y == 2, 0, np.where(train_Y == 4, 1, 2))
        
        print('Fin overlap 1')

    # means = np.mean(train_X, axis=0)
    # stds = np.std(train_X, axis=0)

    # train_X = normalise_all_zscore(train_X, means, stds)
    # val_X = normalise_all_zscore(val_X, means, stds)
    # test_X = normalise_all_zscore(test_X, means, stds)

    #train_X = normalise_all_zscore(train_X, means, stds)
    # Concatenate train and val to normalise
    # train_val_X = np.concatenate((train_X, val_X), axis=0)

    # means = np.mean(train_val_X, axis=0)
    # stds = np.std(train_val_X, axis=0)
    # train_val_X = normalise_all_zscore(train_val_X, means, stds)

    # train_X = train_val_X[:train_X.shape[0]]
    # val_X = train_val_X[train_X.shape[0]:]
    # test_X = normalise_all_zscore(test_X, means, stds)

    #means = np.mean(test_X, axis=0)
    #stds = np.std(test_X, axis=0)
    #test_X = normalise_all_zscore(test_X, means, stds)

    #means = np.mean(val_X, axis=0)
    #stds = np.std(val_X, axis=0)
    #val_X = normalise_all_zscore(val_X, means, stds)
    # Save new_labels_name as numpy array
    # np.save('dance_sweep_20users/new_labels_name.npy', new_labels_name)

    # Here we assign the class DRINK as 0, the class EAT as 1 and the rest as class 2
    # if dataset doesn't start with 'vivabem'
    # save the array test_X
    #np.save('dance_sweep_20users/test_X.npy', test_X)

    # 'CHAIR': 0, 'DANCE': 1, 'DRINK': 2, 'DRY': 3, 'EAT': 4, 'LYING': 5, 'PAPERS': 6, 
    # 'STAIRS': 7, 'STAND': 8, 'SWEEP': 9, 'TV': 10, 'TYPING': 11, 'WALK': 12    
    

    # if not dataset.startswith('vivabem'): # and not dataset.startswith('eat'):
    #     test_Y = np.where(test_Y == 2, 0, np.where(test_Y == 4, 1, 2))
    #     val_Y = np.where(val_Y == 2, 0, np.where(val_Y == 4, 1, 2))
    #     train_Y = np.where(train_Y == 2, 0, np.where(train_Y == 4, 1, 2))
    
    # np.save('Recurrence/dataset_deo_over_code_2/X_train.npy', train_X)
    # np.save('Recurrence/dataset_deo_over_code_2/y_train.npy', train_Y)
    # np.save('Recurrence/dataset_deo_over_code_2/y_train_all.npy', y_train_all)

    # np.save('Recurrence/dataset_deo_over_code_2/X_val.npy', val_X)
    # np.save('Recurrence/dataset_deo_over_code_2/y_val.npy', val_Y)
    # np.save('Recurrence/dataset_deo_over_code_2/y_val_all.npy', y_val_all)

    # np.save('Recurrence/dataset_deo_over_code_2/X_test.npy', test_X)
    # np.save('Recurrence/dataset_deo_over_code_2/y_test.npy', test_Y)
    # np.save('Recurrence/dataset_deo_over_code_2/y_test_all.npy', y_test_all)

    # # Save uuid_train, uuid_val, uuid_test in one txt file4
    # np.save('Recurrence/dataset_deo_over_code_2/uuid_train.npy', uuid_train)
    # np.save('Recurrence/dataset_deo_over_code_2/uuid_val.npy', uuid_val)
    # np.save('Recurrence/dataset_deo_over_code_2/uuid_test.npy', uuid_test)
    
    # # Save users_train, users_val, users_test
    # np.save('Recurrence/dataset_deo_over_code_2/users_test.npy', train_users)
    # np.save('Recurrence/dataset_deo_over_code_2/users_val.npy', val_users)
    # np.save('Recurrence/dataset_deo_over_code_2/users_test.npy', test_users)

    # %%
    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    dict_arrays['uuid_train'] = uuid_train
    dict_arrays['uuid_val'] = uuid_val
    dict_arrays['uuid_test'] = uuid_test
    # dict_arrays['train_users'] = train_users
    # dict_arrays['val_users'] = val_users
    # dict_arrays['test_users'] = test_users

    #np.save(test_Y, subdirectory)
    
    return dict_arrays


def get_processed_fold_unsup(df, dataset, modeltype, subdirectory, sensors, fold, seg5, normalize, overlap, overlap_shift, feat=9):
    
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    df_data = df

    if ( (dataset.startswith('de')) & (normalize == True)):
        # Normalise all df for an unsupervised task
        print("Normalizing ...")
        df_data = df.iloc[:,5:]
        df_data = zscore_per_axis(df_data)

        df_data = df_data.reset_index(drop=True)
        df_data = np.array(df_data).astype(np.float32)


    df_data =  df_data.astype(np.float32)
    df_data = df_data.reshape((df_data.shape[0], df_data.shape[1], 1))


    # Check and handle NaN and infinite values
    df_data = np.nan_to_num(df_data, nan=0.0, posinf=0.0, neginf=0.0)

    feat = 9
    d2 = df_data.shape[-2] // feat
    df_data = np.reshape(df_data, (-1, d2, feat))
    print(df_data.shape)
    
    return df_data


"""
Get an average confusion matrix
"""
def normalize_cm(cm, LABELS, 
                    title=None,
                    cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=LABELS, yticklabels=LABELS,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()    
    
    return plt


"""
Split by fixed users.
"""
def split_data_fixedusers( df, uuid_train, uuid_val, uuid_test, augmented=False, augmentation_config=False ): # per users
    
    # Filter for wrong users, it is for cola, for another datasets doesn't have changes
    num_activities = np.unique(df[2])
    dict2 = {}
    for i in range(len(num_activities)):
        dict2[num_activities[i]] = i
    df = df.replace({2:dict2})

    # print("uuid_train: ",uuid_train)
    # print("uuid_val: ",uuid_val)
    # print("uuid_test: ",uuid_test)

    # Save uuid_train, uuid_val, uuid_test in one txt file
    splits_txt = str(uuid_train) + "\n" + str(uuid_val) + "\n" + str(uuid_test[0]) + "\n"

    data_train = df[df[0].isin(uuid_train)]
    data_val = df[df[0].isin(uuid_val)]
    data_test = df[df[0].isin(uuid_test)]

    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    x_train = data_train.iloc[:,5:]  #uuid(0), timestamp(1), act id(2), met id(3), met name(4), data(5), ...
    y_train = data_train.iloc[:,2]

    x_val = data_val.iloc[:,5:]
    y_val = data_val.iloc[:,2]

    x_test = data_test.iloc[:,5:]
    y_test = data_test.iloc[:,2]
        
    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    x_train = np.array(x_train).astype(np.float32)
    x_val = np.array(x_val).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)

    y_train =  np.array(y_train).astype(np.float32)
    y_val  = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)

    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    return x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test, splits_txt


# Split data with only 1 user in test and 1 user in val
def split_data_val_valtestuser(df, file_users_split, val_user, test_user, test_size=0.2): # per users
    uuids = np.unique(df[0])

    uuid_train = df[df[0]!=test_user]
    uuid_train = uuid_train[uuid_train[0] != val_user][0].unique()
    uuid_val = [val_user]
    uuid_test = [test_user]
    
    print("uuid_train: ",uuid_train)
    print("uuid_val: ",uuid_val)
    print("uuid_test: ",uuid_test)
        
    data_train = df[df[0].isin(uuid_train)]
    data_val = df[df[0].isin(uuid_val)]
    data_test = df[df[0].isin(uuid_test)]


    # Save uuid_train, uuid_val, uuid_test in one txt file
    with open(file_users_split, 'w') as f:
        f.write("uuid_train: " + str(uuid_train))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")

        f.write("uuid_val: " + str(uuid_val))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")

        f.write("uuid_test: " + str(uuid_test))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")


    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    x_train = data_train.iloc[:,5:]  #uuid(0), timestamp(1), act id(2), met id(3), met name(4), data(5), ...
    y_train = data_train.iloc[:,2]

    x_val = data_val.iloc[:, 5:]
    y_val = data_val.iloc[:,2]

    x_test = data_test.iloc[:, 5:]
    y_test = data_test.iloc[:,2]

    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    x_train = np.array(x_train).astype(np.float32)
    x_val = np.array(x_val).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)

    y_train =  np.array(y_train).astype(np.float32)
    y_val  = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)


    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    return x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test



# Split data with only 1 user in test split
def split_data_val_testuser(df, file_users_split, test_user, test_size=0.2): # per users
    uuids = np.unique(df[0])

    uuid_train = df[df[0]!=test_user][0].unique()
    uuid_test = [test_user]

    uuid_train, uuid_val = train_test_split(uuid_train,
                                    train_size = 0.95)
    
    print("uuid_train: ",uuid_train)
    print("uuid_val: ",uuid_val)
    print("uuid_test: ",uuid_test)
        

    data_train = df[df[0].isin(uuid_train)]
    data_val = df[df[0].isin(uuid_val)]
    data_test = df[df[0].isin(uuid_test)]

        # Save uuid_train, uuid_val, uuid_test in one txt file
    with open(file_users_split, 'w') as f:
        f.write("uuid_train: " + str(uuid_train))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")

        f.write("uuid_val: " + str(uuid_val))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")

        f.write("uuid_test: " + str(uuid_test))
        f.write("\nactivities: " + str(np.unique(data_train[2])) + "\n\n")


    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    x_train = data_train.iloc[:,5:]  #uuid(0), timestamp(1), act id(2), met id(3), met name(4), data(5), ...
    y_train = data_train.iloc[:,2]

    x_val = data_val.iloc[:, 5:]
    y_val = data_val.iloc[:,2]

    x_test = data_test.iloc[:, 5:]
    y_test = data_test.iloc[:,2]

    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    x_train = np.array(x_train).astype(np.float32)
    x_val = np.array(x_val).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)

    y_train =  np.array(y_train).astype(np.float32)
    y_val  = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)


    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    return x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test


# Get fixed  fold
def get_processed_fixed(df, dataset, subdirectory, sensors, fold, seg5, overlap_shift, fixed_splits=False):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    dict_arrays = {}

    # Load eat val and test data 
    basepath = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/gatednet_ax2Sns.worktrees/4cf329b/"
    train_X = np.load(f'{basepath}dataset/eatdrinkanother_all/train_eatdrinkother_norm.npy', allow_pickle=True)
    val_X = np.load(f'{basepath}dataset/eatdrinkanother_all/val_eatdrinkother_norm.npy', allow_pickle=True)
    test_X = np.load(f'{basepath}dataset/eatdrinkanother_all/test_eatdrinkother_norm.npy', allow_pickle=True)

    # Load y values
    train_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_train_y.npy', allow_pickle=True)
    val_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_val_y.npy', allow_pickle=True)
    test_Y = np.load(f'{basepath}dataset/eatdrinkanother_all/eatdrinkother_test_y.npy', allow_pickle=True)

    # Reshape to N,500, 9
    train_X = np.reshape(train_X, (-1, 500, 9))
    val_X = np.reshape(val_X, (-1, 500, 9))
    test_X = np.reshape(test_X, (-1, 500, 9))
    
    users_train = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_train.npy', allow_pickle=True)
    users_val = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_val.npy', allow_pickle=True)
    users_test = np.load('/home/elian.riveros/dl-13-elian/notebooks/workspaces/eatdrinkanother_data/users_test.npy', allow_pickle=True)

    uuid_train = np.unique(users_train)
    uuid_val = np.unique(users_val)
    uuid_test = np.unique(users_test)

    # Convert to float32
    train_X = train_X.astype(np.float32)
    train_Y = train_Y.astype(np.float32)
    val_X = val_X.astype(np.float32)
    val_Y = val_Y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_Y = test_Y.astype(np.float32)

    # Check and handle NaN and infinite values
    train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    train_Y = np.nan_to_num(train_Y, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num(val_X, nan=0.0, posinf=0.0, neginf=0.0)
    val_Y = np.nan_to_num(val_Y, nan=0.0, posinf=0.0, neginf=0.0)
    test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)
    test_Y = np.nan_to_num(test_Y, nan=0.0, posinf=0.0, neginf=0.0)

    # Overlap data
    train_X, train_Y = overlap_data(train_X, train_Y, users_train, overlap_shift)
    val_X, val_Y = overlap_data(val_X, val_Y, users_val, overlap_shift)
    test_X, test_Y = overlap_data(test_X, test_Y, users_test,overlap_shift)
    
    if ( sensors == 2):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]
        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]
        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ, train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ, val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ, test_gyrX, test_gyrY, test_gyrZ), axis=2)

        #Save train_X wit np.save
        # np.save('dance_sweep_20users_2sns/train_X.npy', train_X)
        # np.save('dance_sweep_20users_2sns/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_2sns/val_X.npy', val_X)
        # np.save('dance_sweep_20users_2sns/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_2sns/test_X.npy', test_X)
        # np.save('dance_sweep_20users_2sns/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_2sns/users_train.npy', users_train)
        # np.save('dance_sweep_20users_2sns/users_val.npy', users_val)
        # np.save('dance_sweep_20users_2sns/users_test.npy', users_test)        

        print(" 2 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()

    elif ( sensors == 1):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)
    
    elif ( sensors == 0): # Gyroscope sensor

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_gyrX = train_X[:,:,3:4]
        train_gyrY = train_X[:,:,4:5]
        train_gyrZ = train_X[:,:,5:6]

        val_gyrX = val_X[:,:,3:4]
        val_gyrY = val_X[:,:,4:5]
        val_gyrZ = val_X[:,:,5:6]

        test_gyrX = test_X[:,:,3:4]
        test_gyrY = test_X[:,:,4:5]
        test_gyrZ = test_X[:,:,5:6]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_gyrX, train_gyrY, train_gyrZ), axis=2)
        val_X = np.concatenate((val_gyrX, val_gyrY, val_gyrZ), axis=2)
        test_X = np.concatenate((test_gyrX, test_gyrY, test_gyrZ), axis=2)

        # np.save('dance_sweep_20users_1sns_g/train_X.npy', train_X)
        # np.save('dance_sweep_20users_1sns_g/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_1sns_g/val_X.npy', val_X)
        # np.save('dance_sweep_20users_1sns_g/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_1sns_g/test_X.npy', test_X)
        # np.save('dance_sweep_20users_1sns_g/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_1sns_g/users_train.npy', users_train)
        # np.save('dance_sweep_20users_1sns_g/users_val.npy', users_val)
        # np.save('dance_sweep_20users_1sns_g/users_test.npy', users_test)        

        print(" 1 sensor gyr")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)
        exit()
    
    elif ( sensors == -1): # Accelerometer sensor

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())
        
        # Extract axis data
        train_accX = train_X[:,:,0:1]
        train_accY = train_X[:,:,1:2]
        train_accZ = train_X[:,:,2:3]

        val_accX = val_X[:,:,0:1]
        val_accY = val_X[:,:,1:2]
        val_accZ = val_X[:,:,2:3]

        test_accX = test_X[:,:,0:1]
        test_accY = test_X[:,:,1:2]
        test_accZ = test_X[:,:,2:3]

        # Conatenate axis data vertically with numpy library
        train_X = np.concatenate((train_accX, train_accY, train_accZ), axis=2)
        val_X = np.concatenate((val_accX, val_accY, val_accZ), axis=2)
        test_X = np.concatenate((test_accX, test_accY, test_accZ), axis=2)

        # np.save('dance_sweep_20users_1sns_a/train_X.npy', train_X)
        # np.save('dance_sweep_20users_1sns_a/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users_1sns_a/val_X.npy', val_X)
        # np.save('dance_sweep_20users_1sns_a/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users_1sns_a/test_X.npy', test_X)
        # np.save('dance_sweep_20users_1sns_a/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users_1sns_a/users_train.npy', users_train)
        # np.save('dance_sweep_20users_1sns_a/users_val.npy', users_val)
        # np.save('dance_sweep_20users_1sns_a/users_test.npy', users_test) 

        print(" 1 sensor acc")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)  
        exit()
    
    elif ( sensors == 3):

        #train_X = torch.FloatTensor(train_X).transpose(-1, -2)
        train_Y = torch.FloatTensor(train_Y.squeeze())

        #val_X = torch.FloatTensor(val_X.squeeze()).transpose(-1, -2)
        val_Y = torch.FloatTensor(val_Y.squeeze())

        #test_X = torch.FloatTensor(test_X.squeeze()).transpose(-1, -2)
        test_Y = torch.FloatTensor(test_Y.squeeze())

        # Save train_X wit np.save
        # np.save('dance_sweep_20users/train_X.npy', train_X)
        # np.save('dance_sweep_20users/train_Y.npy', train_Y)
        # np.save('dance_sweep_20users/val_X.npy', val_X)
        # np.save('dance_sweep_20users/val_Y.npy', val_Y)
        # np.save('dance_sweep_20users/test_X.npy', test_X)
        # np.save('dance_sweep_20users/test_Y.npy', test_Y)
        # np.save('dance_sweep_20users/users_train.npy', users_train)
        # np.save('dance_sweep_20users/users_val.npy', users_val)
        # np.save('dance_sweep_20users/users_test.npy', users_test)         

        print(" 3 sensors")
        print("Shapes:")
        print(train_X.shape, train_Y.shape)
        print(val_X.shape, val_Y.shape)
        print(test_X.shape, test_Y.shape)

    # %%
    print("Shapes:")
    print(train_X.shape, train_Y.shape)
    print(val_X.shape, val_Y.shape)
    print(test_X.shape, test_Y.shape)   
    
    dict_arrays['x_train'] = train_X
    dict_arrays['y_train'] = train_Y
    dict_arrays['x_val'] = val_X
    dict_arrays['y_val'] = val_Y
    dict_arrays['x_test'] = test_X
    dict_arrays['y_test'] = test_Y
    dict_arrays['uuid_train'] = uuid_train
    dict_arrays['uuid_val'] = uuid_val
    dict_arrays['uuid_test'] = uuid_test
    
    return dict_arrays

# Split data DEO
def split_data_val(df,  dataset, file_users_split, test_size=0.2): # per users

    if (dataset=='cola2-agm'):
        df = df[ (df[0]!='S1003') & (df[0]!='S1011') ]
    elif  (dataset=='vivabem12_lnd_ma'):
        df = df[ (df[0]!='S1047') & (df[0]!='S1056') ]
    elif (dataset=='vivabem12_lnd_mb'):
        df = df[ (df[0]!='S1002') & (df[0]!='S1003') & (df[0]!='S1056') ]
    elif (dataset=='vivabem12_lnd_ma_mb'):
        df = df[ (df[0]!='S1002') & (df[0]!='S1003') & (df[0]!='S1056') ]
    
    num_activities = np.unique(df[2])
    dict2 = {}
    for i in range(len(num_activities)):
        dict2[num_activities[i]] = i
    df = df.replace({2:dict2})
    
    uuids = np.unique(df[0])

    uuid_train, uuid_test = train_test_split(uuids,
                                        test_size = test_size)

    data_train = df[df[0].isin(uuid_train)]
    data_test = df[df[0].isin(uuid_test)]

    uuids_train = np.unique(data_train[0])

    uuid_train, uuid_val = train_test_split(uuids_train,
                                        train_size = 0.86,
                                        test_size = 0.14)

    # print("uuid_train: ",uuid_train)
    # print("uuid_val: ",uuid_val)
    # print("uuid_test: ",uuid_test)
        
    # Save uuid_train, uuid_val, uuid_test in one txt file
    splits_txt = str(uuid_train) + "\n" + str(uuid_val) + "\n" + str(uuid_test) + "\n"

    data_train = df[df[0].isin(uuid_train)]
    data_val = df[df[0].isin(uuid_val)]

        # Save uuid_train, uuid_val, uuid_test in one txt file
    with open(file_users_split, 'w') as f:
        f.write("uuid_train: " + str(uuid_train))
        f.write("activities: " + str(np.unique(data_train[2])) + "\n")

        f.write("uuid_val: " + str(uuid_val))
        f.write("activities: " + str(np.unique(data_train[2])) + "\n")

        f.write("uuid_test: " + str(uuid_test))
        f.write("activities: " + str(np.unique(data_train[2])) + "\n")


    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # Save the column 4 to a numpy file
    # saved = '/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/Recurrence/saved_models/20240909-185214/fullraws3_test_Y_vivabem012_drink0eat1another2.npy'
    # np.save(saved, data_test[4].to_numpy())
    
    #path = '/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/Recurrence/dataset_deo2/'
    #np.save(path+'X_test.npy', data_test.to_numpy())
    x_train = data_train.iloc[:,5:]  #uuid(0), timestamp(1), act id(2), met id(3), met name(4), data(5), ...
    y_train = data_train.iloc[:,2]
    y_train_all_str = data_train.iloc[:,4]

    x_val = data_val.iloc[:, 5:]
    y_val = data_val.iloc[:,2]
    y_val_all_str = data_val.iloc[:,4]

    x_test = data_test.iloc[:, 5:]
    y_test = data_test.iloc[:,2]
    y_test_all_str = data_test.iloc[:,4]

    y_train.reset_index(drop = True, inplace = True)
    y_val.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    x_train = np.array(x_train).astype(np.float32)
    x_val = np.array(x_val).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)

    y_train =  np.array(y_train).astype(np.float32)
    y_val  = np.array(y_val).astype(np.float32)
    y_test  = np.array(y_test).astype(np.float32)


    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    # Create a list with the corresponding IDs
    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_test_all_str))}
    y_test_all = [string_to_id[string] for string in y_test_all_str]

    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_val_all_str))}
    y_val_all = [string_to_id[string] for string in y_val_all_str]

    string_to_id = {string: idx for idx, string in enumerate(np.unique(y_train_all_str))}
    y_train_all = [string_to_id[string] for string in y_train_all_str]

    y_train_all =  np.array(y_train_all).astype(np.float32)
    y_val_all  = np.array(y_val_all).astype(np.float32)
    y_test_all  = np.array(y_test_all).astype(np.float32)

    # 'CHAIR': 0, 'DANCE': 1, 'DRINK': 2, 'DRY': 3, 'EAT': 4, 'LYING': 5, 'PAPERS': 6, 
    # 'STAIRS': 7, 'STAND': 8, 'SWEEP': 9, 'TV': 10, 'TYPING': 11, 'WALK': 12

    return x_train, y_train, x_val, y_val, x_test, y_test, y_train_all, y_val_all, y_test_all, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt
    #return x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test, splits_txt


"""
Resize dataset to 3-dimensional vectors.
Each vector represent 1-second of activity, this
is the input vector to the model.
"""
def resize_data(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    x_train = np.reshape(x_train, (-1, 100, 6))
    x_val = np.reshape(x_val, (-1, 100, 6))
    x_test = np.reshape(x_test, (-1, 100, 6))
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_val, 'y_val': y_val, 
                'x_test': x_test, 'y_test':y_test}
                
    return dict_arrays


"""
Resize dataset to 3-dimensional vectors.
Each vector represent 1-second of activity, this
is the input vector to the model.
"""
def resize_data_3sns(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    x_train = np.reshape(x_train, (-1, 100, 9))
    x_val = np.reshape(x_val, (-1, 100, 9))
    x_test = np.reshape(x_test, (-1, 100, 9))
    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_val, 'y_val': y_val, 
                'x_test': x_test, 'y_test':y_test}
                
    return dict_arrays


"""
Divide the 'y' vector to 5 times the size. For windows of 5sec.
"""
def resize_y(y_):
    c = np.ones([y_.shape[0], 5])
    c[:,0] = c[:,0] * y_

    c[:,1] = c[:,1] * y_
    c[:,2] = c[:,2] * y_
    c[:,3] = c[:,3] * y_
    c[:,4] = c[:,4] * y_

    c = c.reshape([y_.shape[0]*5, 1])
    c = c.squeeze() #//c.squeeze()
    return c
    

# Read full_raws
def read_full_raws(saved_file, dataset_name=None): 
    print("\nReading full_raws from file: ", saved_file)
    #full_raws = pd.read_csv(saved_file, header=0)
    full_raws = pd.read_csv(saved_file, header=None)
    if dataset_name.startswith('de_fake_padts'):
        full_raws.columns = range(full_raws.shape[1])
        full_raws = full_raws.iloc[1:].reset_index(drop=True)

    print("full_raws shape:", full_raws.shape)

    return full_raws


# Read full_raws
def read_full_raws_without_lying(saved_file): 
    print("\nReading full_raws from file: ", saved_file)
    vivabem12 = pd.read_csv(saved_file, header=None)
    vivabem12_tv = vivabem12[vivabem12[2]!=5]
    
    # Removing lying
    dict2 = {
        6:5,
        7:6,
        8:7,
        9:8,
        10:9,
        11:10,
        12:11}
    
    vivabem12_tv = vivabem12_tv.replace({2: dict2})
    
    print("full_raws shape:", vivabem12_tv.shape)
    print()
    return vivabem12_tv


# Read full_raws
def read_full_raws_without_tv(saved_file): 
    print("\nReading full_raws from file: ", saved_file)
    vivabem12 = pd.read_csv(saved_file, header=None)
    
    # Removing tv
    vivabem12_lying = vivabem12[vivabem12[2]!=10]
    a = vivabem12_lying[2]
    a[a == 11] = 10
    print(np.unique(a))
    
    a[a == 12] = 11
    print(np.unique(a))
          
    vivabem12_lying[2] = a
    print(np.unique(vivabem12_lying[2]))
    
    print("full_raws shape:", vivabem12_lying.shape)
    print()
    return vivabem12_lying


# Save data split
def save_csv_splitdata(data_train, data_val, data_test, file_data_train, file_data_test, file_data_val):
	print()
	print("Saving split data...")
	data_train.to_csv(file_data_train, header=None, index=False)
	data_val.to_csv(file_data_val, header=None, index=False)
	data_test.to_csv(file_data_test, header=None, index=False)
	print("shape data_train: ",data_train.shape)
	print("shape data_val: ",data_val.shape)
	print("shape data_test: ",data_test.shape)
	print()
	print("Split data saved in: ")
	print(file_data_train)
	print(file_data_val)
	print(file_data_test)

"""
Resize dataset to 3-dimensional vectors.
Each vector represent 1-second of activity, this
is the input vector to the model.
"""
def resize_data_axis3sns(dict_arrays):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    y_train_all = dict_arrays['y_train_all']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    y_val_all = dict_arrays['y_val_all']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']
    y_test_all = dict_arrays['y_test_all']    

    x_train = np.reshape(x_train, (-1, 100, 9))
    x_val = np.reshape(x_val, (-1, 100, 9))
    x_test = np.reshape(x_test, (-1, 100, 9))

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    y_train = resize_y(y_train)
    y_val = resize_y(y_val)
    y_test = resize_y(y_test)

    y_train_all = resize_y(y_train_all)
    y_val_all = resize_y(y_val_all)
    y_test_all = resize_y(y_test_all)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 'y_train_all': y_train_all,
                'x_val': x_val, 'y_val': y_val, 'y_val_all': y_val_all,
                'x_test': x_test, 'y_test':y_test, 'y_test_all': y_test_all}
                
    return dict_arrays

"""
Resize dataset to 3-dimensional vectors.
Each vector represent 5-second of activity, this
is the input vector to the model.
"""
def resize_data_axis3sns_5sec(dict_arrays, feat=9):
    x_train = dict_arrays['x_train']
    x_val = dict_arrays['x_val']
    x_test = dict_arrays['x_test']
    
    d2 = x_train.shape[-2] // feat
    if ( feat==18):
        d2 = 500
    x_train = np.reshape(x_train, (-1, d2, feat))
    x_val = np.reshape(x_val, (-1, d2, feat))
    x_test = np.reshape(x_test, (-1, d2, feat))

    dict_arrays['x_train'] = x_train
    dict_arrays['x_val'] = x_val
    dict_arrays['x_test'] = x_test
                
    return dict_arrays

        
"""
Divide the 'y' vector to 5 times the size. For windows of 5sec.
"""

def resize_users_lst(u_):
       
    c = np.empty([u_.shape[0], 5], dtype='object')
    c[:,0] = u_

    c[:,1] = u_
    c[:,2] = u_
    c[:,3] = u_
    c[:,4] = u_

    c = c.reshape([u_.shape[0]*5, 1])
    c = c.squeeze()
    return c

'''
Get data arrays and reshape to 1sec with 3 sensors
'''
def get_data_arrays_3sns(x_train, y_train, y_train_all, users_train, 
                        x_valid, y_valid, y_val_all, users_val, 
                        x_test, y_test, y_test_all, users_test, 
                        dataset, seg5, feat=9, shuffle_data=False, axis=False):
    
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train =  y_train.astype(np.float32)
    y_valid  = y_valid.astype(np.float32)
    y_test =  y_test.astype(np.float32)

    y_test_all =  y_test_all.astype(np.float32)
    y_val_all =  y_val_all.astype(np.float32)
    y_train_all =  y_train_all.astype(np.float32)
    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    # (N, 9000, 1)

    users_train = np.array(users_train)
    users_val = np.array(users_val)
    users_test = np.array(users_test)

    if shuffle_data == True:
        x_train , y_train = shuffle(x_train, y_train)
        x_valid , y_valid = shuffle(x_valid, y_valid)
        x_test , y_test = shuffle(x_test, y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 'y_train_all': y_train_all,
                'x_val': x_valid, 'y_val': y_valid, 'y_val_all': y_val_all,
                'x_test': x_test, 'y_test':y_test, 'y_test_all': y_test_all,}
    
    if seg5 == False:
        print("Resize to 1sec  ",dataset)
        if axis == True:
            dict_arrays = resize_data_axis3sns(dict_arrays)
        else:
            dict_arrays = resize_data_3sns(dict_arrays)

        users_train = resize_users_lst(users_train)
        users_val = resize_users_lst(users_val)
        users_test = resize_users_lst(users_test)
    
    else:
        dict_arrays = resize_data_axis3sns_5sec(dict_arrays, feat)

    return dict_arrays, users_train, users_val, users_test



def get_data_arrays(x_train, y_train, x_valid, y_valid, x_test, y_test, dataset, magni):

    print("Resize to 1sec  ",dataset)
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train =  y_train.astype(np.float32)
    y_valid  = y_valid.astype(np.float32)
    y_test =  y_test.astype(np.float32)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    x_train , y_train = shuffle(x_train, y_train)
    x_valid , y_valid = shuffle(x_valid, y_valid)
    x_test , y_test = shuffle(x_test, y_test)

    n_classes = len(np.unique(y_train))
    print("No of classes:", n_classes)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_valid, 'y_val': y_valid, 
                'x_test': x_test, 'y_test':y_test}
    
    dict_arrays = resize_data(dict_arrays)

    return dict_arrays




# BUILD METRICS
def build_metrics(_model, dict_arrays):
    
    model = _model

    full_raws_train_fea_np = dict_arrays['x_train']
    full_raws_y_train = dict_arrays['y_train']
    full_raws_val_fea_np = dict_arrays['x_val']
    full_raws_y_val = dict_arrays['y_val']
    full_raws_test_fea_np = dict_arrays['x_test']
    full_raws_y_test = dict_arrays['y_test']

    # TRAIN #
    preds_t = model.predict(full_raws_train_fea_np)
    preds_t_flat = np.argmax(preds_t, axis=1).reshape(-1)
    accuracy_train = float(np.sum(preds_t_flat==full_raws_y_train))/full_raws_y_train.shape[0]
    balanced_accuracy_train = balanced_accuracy_score(full_raws_y_train, preds_t_flat)
    f1_score_train_we = f1_score(full_raws_y_train,  preds_t_flat, average = 'weighted') #macro weighted
    f1_score_train_mic = f1_score(full_raws_y_train,  preds_t_flat, average = 'micro')
    kappa_train = cohen_kappa_score(full_raws_y_train,  preds_t_flat)

    # VAL #
    preds_t_val = model.predict(full_raws_val_fea_np)
    preds_t_flat_val = np.argmax(preds_t_val, axis=1).reshape(-1)
    accuracy_val = float(np.sum(preds_t_flat_val==full_raws_y_val))/full_raws_y_val.shape[0]
    balanced_accuracy_val = balanced_accuracy_score(full_raws_y_val, preds_t_flat_val)
    f1_score_val_we = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'weighted') #macro weighted
    f1_score_val_mic = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'micro')
    kappa_val = cohen_kappa_score(full_raws_y_val,  preds_t_flat_val)

    # TEST #
    preds_t_test = model.predict(full_raws_test_fea_np)
    preds_t_flat_test = np.argmax(preds_t_test, axis=1).reshape(-1)
    accuracy_test = float(np.sum(preds_t_flat_test==full_raws_y_test))/full_raws_y_test.shape[0]
    balanced_accuracy_test = balanced_accuracy_score(full_raws_y_test, preds_t_flat_test)
    f1_score_test_we = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'weighted') #macro weighted
    f1_score_test_mic = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'micro')
    kappa_test = cohen_kappa_score(full_raws_y_test,  preds_t_flat_test)

    return [[accuracy_train, balanced_accuracy_train, f1_score_train_we, f1_score_train_mic, kappa_train], 
            [accuracy_val, balanced_accuracy_val, f1_score_val_we, f1_score_val_mic, kappa_val],
             [accuracy_test, balanced_accuracy_test, f1_score_test_we, f1_score_test_mic, kappa_test]]

""" 
Plot normalized confusion matrix
"""
def plot_confusion_matrix( y_true, y_pred, LABELS,
                          normalize=True,
                          title=None,
                          path_save='cm.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    print("\nCreating the confusion matrix ...")
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Choose normalized values or not
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=LABELS, yticklabels=LABELS,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()    
    plt.savefig(path_save)
    plt.close()


# ONE HOT ENCODING
def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# TABLE 
def build_metrics_table( dict_arrays, metric_results, table, time_, modeltype, dataset, last_epoch_early_stopping, learning_rate, dropout_rate, n_batch, LSTM_layers, lstm_hidden_units, lstm_reg, clf_reg, clipvalue, obs, subdirectory):
    new_row = { 'time' : time_,
                'modeltype' : modeltype,
                'dataset': dataset,
                'uuid_val': dict_arrays['uuid_val'],
                'uuid_test': dict_arrays['uuid_test'],
                'n_epochs': last_epoch_early_stopping, 
                'lr' : str(learning_rate),
                'do' : str(dropout_rate),
                'ov' : "",
                'batch' : str(n_batch),
                'layers' : str(LSTM_layers),
                'h_units' : str(lstm_hidden_units),
                'lstm_reg' : str(lstm_reg),
                'clf_reg' : str(clf_reg),
                'clipvalue' : str(clipvalue),
                'train': round(metric_results[0][0], 4),
                'train_bal' : round(metric_results[0][1], 4),
                'f1_score_train_we': round(metric_results[0][2], 4),
                'kappa_train': round(metric_results[0][4], 4),
                'val': round(metric_results[1][0], 4),
                'val_bal': round(metric_results[1][1], 4),
                'f1_score_val_we': round(metric_results[1][2], 4),
                'kappa_val': round(metric_results[1][4], 4),
                'test': round(metric_results[2][0], 4),
                'test_bal': round(metric_results[2][1], 4),
                'f1_score_test_we': round(metric_results[2][2], 4),
                'kappa_test': round(metric_results[2][4], 4),
                'obs' : obs,
                'path': subdirectory
            }

	
    print(new_row)
    table = table.append(new_row, ignore_index=True)
    table_name = subdirectory+"/table_"+dataset+"_"+time_+".csv"
    print("Saving table in...", table_name)
    table.to_csv(table_name, sep=',', encoding='utf-8', index=False)


# COMPUTE METRICS
def compute_metrics(predictions, y_test, n_classes, LABELS, subdirectory, dataset, time_):
    local_time_ = time.strftime("%Y%m%d-%H%M%S")
    one_hot_predictions = predictions.argmax(1)
    precision = metrics.precision_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    recall = metrics.recall_score(y_test, one_hot_predictions, average="weighted")
    f1_score = metrics.f1_score(y_test, one_hot_predictions, average="weighted", zero_division=0)
    mAP=np.mean(np.asarray([(metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")) for c in range(n_classes)]))
    with open(subdirectory+"/metrics_"+dataset+"_"+time_+"_"+local_time_+".txt", 'w') as file:
        file.write("Dataset: {}".format( dataset))
        file.write("\n")
        file.write("mAP score: {:.4f}\n".format(mAP))
        file.write("precision: {:.4f}\n".format(precision))
        file.write("recall: {:.4f}\n".format(recall))
        file.write("f1_score: {:.4f}".format(f1_score))
        file.write("\n")
        for c in range(n_classes):
            metr = metrics.average_precision_score(one_hot(y_test, n_classes)[:,c], one_hot(one_hot_predictions, n_classes)[:,c], average="weighted")
            metr = round(metr, 4)
            file.write("\n")
            file.write(str(LABELS[c])+"\t" + str(metr))
    
    confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
    

    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]


    # SAVE THE MODEL
    width = 20
    height = 20

    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.Blues # Blues or grey
    )
    thresh = confusion_matrix.max() *.5
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if (confusion_matrix[i, j] > thresh) and (confusion_matrix[i, j] != 0) else "black") # > if Blues, < if grey
    plt.title("Confusion matrix {} (F1={:.4f} - mAP={:.4f}) \n(normalised to % of total test data)".format(dataset, metrics.f1_score(y_test, one_hot_predictions, average="weighted"), mAP))
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=45)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(subdirectory + '/CM_' + dataset+'_'+time_+'_'+local_time_+'.png')

    # Normalized confusion matrix
    path_fig = subdirectory + '/CMn_' + dataset+'_'+time_+'_'+local_time_+'.png'
    plot_confusion_matrix( y_test, one_hot_predictions, LABELS,
                        normalize=True,
                        title="Normalized Confusion Matrix",
                        path_save = path_fig,
                        cmap=plt.cm.Blues)

    return confusion_matrix      



def get_class_names(dataset):
    classes_name = []
    if dataset == 'cola' or dataset == 'cola2-minmax2' or dataset == 'cola2-minmax' or dataset == 'cola2-minmax3':
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'STAIRS', 'SWEEP'])
    
    elif dataset == 'cola2-agm':
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'STAIRS', 'SWEEP'])

    elif dataset == 'ext':
        classes_name = np.array(['LYING_DOWN', 'SITTING', 'STANDING_IN_PLACE', 'STANDING_AND_MOVING', 'WALKING', 'RUNNING', 'BICYCLING'])

    elif dataset == 'haruml' or dataset == 'haruml2' or dataset=='haruml-3sns' or dataset=='haruml-agm':
        classes_name = np.array(['DOWNSTAIRS', 'LYING', 'RUNNING', 'SITTING', 'STANDING', 'UPSTAIRS', 'WALKING'])

    elif dataset == 'basa' or dataset == 'basa6040max' or dataset == 'basa6040lit':
        classes_name = np.array(['Sitting', 'Lying', 'Standing', 'Ascending_stairs', 
                    'Descending_stairs', 'Walking_1', 'Walking_2', 'Jogging_1', 'Jogging_2'])
        classes_name = np.array(['Sitting', 'Lying', 'Standing', 'Ascending_stairs', 
                    'Descending_stairs', 'Walking_1', 'Jogging_1'])
        
    elif dataset == 'mhealth' or dataset == 'mhealth-agm' or dataset == 'mhealth-agm-cub':
        classes_name = np.array([ '0', '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11'])
        classes_name = np.array(['Standing still', 'Sitting and relaxing', 'Lying down',
                'Walking', 'Climbing stairs', 'Waist bends forward', 'Frontal elevation of arms',
                'Knees bending (crouching)', 'Cycling', 'Jogging', 'Running', 'Jump front & back'])
        
    elif dataset == 'pamap2' or dataset == 'pamap2-agm' or dataset == 'pamap2-inter'  or dataset == 'pamap2-agm-inter':
            classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8'])
            classes_name = np.array(['lying','sitting','standing','walking','running','cycling','Nordic walking','descending stairs','vacuum cleaning'])
    
    elif dataset == 'vivabem12' or dataset == 'vivabem12_ld_ma_mb' or dataset == 'vivabem12_rd_ma_mb' or dataset == 'vivabem12_ld_rd_ma_mb':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'LYING', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TV', 'TYPING', 'WALK'])
    
    elif dataset == 'vivabem12_lying':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'LYING', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TYPING', 'WALK'])
    
    elif  dataset == 'vivabem12_tv':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TV', 'TYPING', 'WALK'])

    elif dataset == 'vivabem12_5' or dataset == 'vivabem12_10'  or dataset == 'vivabem12_20':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRINK', 'DRY', 'EAT', 'LYING', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TV', 'TYPING', 'WALK'])
     
    
    elif dataset == 'vivabem12_lnd_ma' or dataset == 'vivabem12_lnd_mb' or dataset == 'vivabem12_lnd_ma_mb':
        classes_name = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
        classes_name = np.array(['CHAIR', 'DANCE', 'DRY', 'EAT', 'LYING', 'PAPERS', 'STAIRS', 'STAND', 'SWEEP', 'TV', 'TYPING', 'WALK'])
    
    elif dataset == 'eatdrinkanother' or dataset == 'eatdrinkanother_aug' or dataset == 'eatdrinkanother5' or dataset == 'eatdrinkanother_94u_5' or dataset == 'eatdrinkanother_94u' or dataset == 'eatdrinkanother_94u_10':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])

    elif dataset == 'deo_butax_500x9' or dataset == 'deo_buttco_500x9' or dataset == 'deo_converted_butax_fftax_500x9' or dataset == 'deo_converted_butco_fftco_500x9' or dataset == 'deo_converted_fftax_500x9' or dataset == 'deo_converted_fftco_500x9':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])
    
    elif dataset == 'deo_join_fftax_500x18' or dataset=='deo_join_butco_fftco_500x18' or dataset=='deo_join_fftco_500x18' or dataset=='deo_join_butax_fftax_500x18':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])
    
    elif dataset == 'deo_converted_butco_fftco_250x9' or dataset == 'deo_converted_butax_fftax_250x9':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])
    
    elif dataset == 'deo_join_butco_fftco_750x9' or dataset == 'deo_join_butax_fftax_750x9':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])
    
    elif dataset == 'deo_join_butax_fftax_1000x9' or dataset == 'deo_join_butco_fftco_1000x9':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])

    return classes_name

    

def get_raw_datasets(dataset, magni):
    basename = "../../../files/"
    basename = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/files/"
    basename2 = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/"
    file_data_train = None
    file_data_val = None
    file_data_test = None
    
    if dataset=='vivabem12' or dataset=='vivabem12_tv' or dataset=='vivabem12_lying':
        file_full_raws = basename+"fullraws_vivabem12_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws_vivabem12_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws_vivabem12_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws_vivabem12_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        
    elif dataset=='vivabem12_5':
        file_full_raws = basename+"fullraws_vivabem12_5ta_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws_vivabem12_5ta_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws_vivabem12_5ta_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws_vivabem12_5ta_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_10':
        file_full_raws = basename+"fullraws_vivabem12_10part_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws_vivabem12_10part_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws_vivabem12_10part_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws_vivabem12_10part_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_20':
        file_full_raws = basename+"fullraws_vivabem12_20_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws_vivabem12_20_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws_vivabem12_20_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws_vivabem12_20_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_lnd_ma':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_L_ND_MA_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_L_ND_MA_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_L_ND_MA_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_L_ND_MA_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_lnd_mb':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_L_ND_MB_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_L_ND_MB_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_L_ND_MB_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_L_ND_MB_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"

    elif dataset=='vivabem12_lnd_ma_mb':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_L_ND_MA_MB_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_L_ND_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_L_ND_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_L_ND_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_ld_rd_ma_mb':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_LD_RD_MA_MB_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_LD_RD_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_LD_RD_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_LD_RD_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_ld_ma_mb':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_L_D_MA_MB_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_L_D_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_L_D_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_13act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_L_D_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_13act.csv"
    
    elif dataset=='vivabem12_rd_ma_mb':
        file_full_raws = basename+"fullraws2_vivabem12_noaer_R_D_MA_MB_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_train = basename+"fullraws2_vivabem12_noaer_R_D_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_val = basename+"fullraws2_vivabem12_noaer_R_D_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_test = basename+"fullraws2_vivabem12_noaer_R_D_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_6act.csv"
    
    elif dataset=='vivabem12_1watch':
        file_full_raws = basename+"fullraws2_vivabem12_1watch_noaer_nodance_LD_RD_LND_RND_MA_MB_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_train = basename+"fullraws2_vivabem12_1watch_noaer_nodance_LD_RD_LND_RND_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_val = basename+"fullraws2_vivabem12_1watch_noaer_nodance_LD_RD_LND_RND_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_6act.csv"
        file_data_test = basename+"fullraws2_vivabem12_1watch_noaer_nodance_LD_RD_LND_RND_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_6act.csv"

    elif dataset=='vivabem12_2watch':
        file_full_raws = basename+"fullraws2_vivabem12_2watch_noaer_nodance_LD_RD_LND_RND_MA_MB_acc_gyr_mag__coord__5sec_100hz_8act.csv"
        file_data_train = basename+"fullraws2_vivabem12_2watch_noaer_nodance_LD_RD_LND_RND_MA_MB_train_acc_gyr_mag__coord__5sec_100hz_8act.csv"
        file_data_val = basename+"fullraws2_vivabem12_2watch_noaer_nodance_LD_RD_LND_RND_MA_MB_val_acc_gyr_mag__coord__5sec_100hz_8act.csv"
        file_data_test = basename+"fullraws2_vivabem12_2watch_noaer_nodance_LD_RD_LND_RND_MA_MB_test_acc_gyr_mag__coord__5sec_100hz_8act.csv"
        
    elif dataset=='eatdrinkanother':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+"fullraws3_vivabem012_drink0eat1another2_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_val = basename+"fullraws3_vivabem012_drink0eat1another2_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_test = basename+"fullraws3_vivabem012_drink0eat1another2_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset=='eatdrinkanother_94u':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_94u_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""

    elif dataset=='eatdrinkanother_94u_5':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_94u_5th_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""

    elif dataset=='eatdrinkanother_94u_10':
        file_full_raws = basename+"fullraws3_vi10abem012_drink0eat1another2_94u_10th_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""

    elif dataset=='eatdrinkanother_94u_20':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_94u_20th_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""
        
    elif dataset=='eatdrinkanother5':
        file_full_raws = basename+'fullraws3_eatdrinkanother5th_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv'
        file_data_train = basename+'fullraws3_eatdrinkanother5th_nowspr_noaer_train_acc_gyr_mag__coord__5sec_100hz.csv'
        file_data_val = basename+'fullraws3_eatdrinkanother5th_nowspr_noaer_val_acc_gyr_mag__coord__5sec_100hz.csv'
        file_data_test = basename+'fullraws3_eatdrinkanother5th_nowspr_noaer_test_acc_gyr_mag__coord__5sec_100hz.csv'    
        
    elif dataset=='eatdrinkanother_aug':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_aug_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+"fullraws3_vivabem012_drink0eat1another2_aug_train_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_val = basename+"fullraws3_vivabem012_drink0eat1another2_aug_val_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_test = basename+"fullraws3_vivabem012_drink0eat1another2_aug_test_acc_gyr_mag__coord__5sec_100hz.csv"
    
    elif dataset=='de_fake_padts_94u':
        basename = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har"
        file_full_raws = basename + "/PaD-TS/OUTPUT/drinkeat_20250925-100620/ddpm_fake_drinkeat_4505_20251006-191354.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""
    
    elif dataset=='de_fake_padts_100-9_94u':
        basename = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har"
        file_full_raws = basename + "/PaD-TS/OUTPUT/drinkeat_20250925-100620/ddpm_fake_drinkeat_100-9_4505.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""

    elif dataset=='de_d_e_fake_100-9':
        basename = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har"
        file_full_raws = basename + "/PaD-TS/OUTPUT/drinkeat_1009/ddpm_fake_drink_eat_1009_4505_.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""

    elif dataset=='deo_drinkeat_94u_train':
        basename = "/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main"
        file_full_raws = basename + "/PCF-GAN/data/DEO10/deo_drinkeat_94u_train_acc_gyr_mag_coord_5secs_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""
    
    elif dataset=='deo_drinkeat_94u_gen_train':
        file_full_raws = basename+ "fullraws3_vivabem012_drink0eat1_94u_gen_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        file_data_train = basename+""
        file_data_val = basename+""
        file_data_test = basename+""
    
    elif dataset=='deo_butax_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/df_but_byaxis.csv"
    
    elif dataset=='deo_buttco_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/df_but_bytime.csv"
    
    elif dataset=='deo_converted_butax_fftax_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/df_but_byaxis_fft_byaxis.csv"
    
    elif dataset=='deo_converted_butco_fftco_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/df_but_bytime_fft_bytime.csv"
    
    elif dataset=='deo_converted_fftax_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/deo_df_fft_byaxis.csv"

    elif dataset=='deo_converted_fftco_500x9':
        file_full_raws = basename2 + "fft_files/order2_500x9/deo_df_fft_comp.csv"
    
    elif dataset == 'deo_converted_fftax_250x9':
        file_full_raws = basename2 + "fft_files/order2_250x9/deo_df_fft_byaxis_signal.csv"

    elif dataset == 'deo_converted_fftco_250x9':
        file_full_raws = basename2 + "fft_files/order2_250x9/deo_df_fft_complete_signal2.csv"

    elif dataset == 'deo_converted_butco_fftco_250x9':
        file_full_raws = basename2 + "fft_files/order2_250x9/df_but_byaxis_fft_byaxis.csv"

    elif dataset == 'deo_converted_butax_fftax_250x9':
        file_full_raws = basename2 + "fft_files/order2_250x9/df_but_bytime_fft_bytime.csv"
    
    elif dataset == 'deo_join_fftax_750x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_byaxis_250x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_fftco_750x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_comp_250x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_butax_fftax_750x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_ax_fft_ax_250x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_butco_fftco_750x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_comp_fft_comp_250x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset=='deo_join_fftax_750x9_10000':        
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_10000_fft_byaxis_250x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_fftax_1000x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_byaxis_500x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_fftco_1000x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_comp_500x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
    
    elif dataset == 'deo_join_butax_fftax_1000x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_ax_fft_ax_500x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
    
    elif dataset == 'deo_join_butco_fftco_1000x9':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_comp_fft_comp_500x9_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_fftax_500x18':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_byaxis_500x9_18f_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
        
    elif dataset == 'deo_join_fftco_500x18':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_fft_comp_500x9_18f_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_butax_fftax_500x18':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_ax_fft_ax_500x9_18f_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"

    elif dataset == 'deo_join_butco_fftco_500x18':
        file_full_raws = basename2 + "files/fullraws3_vivabem012_drink0eat1another2_filter_comp_fft_comp_500x9_18f_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"


        
    if dataset=='cola':
        file_data_train = basename+"fullraws_cola_noAER_train_acc_gyr__coord__5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_cola_noAER_val_acc_gyr__coord__5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_cola_noAER_test_acc_gyr__coord__5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_cola_noAER_acc_gyr__coord__5sec_100hz_7act.csv"
        
    elif dataset=='cola2-agm':
        file_full_raws = basename+"fullraws_cola2_noaer_acc_gyr_mag__coord__5sec_100hz_7act.csv"
        file_data_train = basename+"fullraws_cola2_noaer_train_acc_gyr_mag__coord__5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_cola2_noaer_val_acc_gyr_mag__coord__5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_cola2_noaer_test_acc_gyr_mag__coord__5sec_100hz_7act.csv"

    elif dataset == 'cola2-minmax':
        print("COLA dataset scaled minmax")
        file_full_raws = basename+"fullraws_cola2_noAER_acc_gyr_minmax_coord__5sec_100hz_7act.csv"
        file_data_train = basename+"fullraws_cola2_noAER_train_acc_gyr_minmax_coord__5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_cola2_noAER_val_acc_gyr_minmax_coord__5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_cola2_noAER_test_acc_gyr_minmax_coord__5sec_100hz_7act.csv"

    elif dataset == 'cola2-minmax3':
        print("COLA dataset scaled minmax")
        file_full_raws = basename+"fullraws_cola2_noaer_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act.csv"
        file_data_train = basename+"fullraws_cola2_noaer_train_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_cola2_noaer_val_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_cola2_noaer_test_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act.csv"

    elif dataset == 'cola2-3':
        print("COLA dataset scaled minmax")
        file_full_raws = basename+"fullraws_cola2_noaer_acc_gyr_mag_coord__5sec_100hz_7act.csv"
        file_data_train = basename+"fullraws_cola2_noaer_train_acc_gyr_mag_coord__5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_cola2_noaer_val_acc_gyr_mag_coord__5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_cola2_noaer_test_acc_gyr_mag_coord__5sec_100hz_7act.csv"

    elif dataset == 'cola2-3_augx2_60':
        print("COLA dataset scaled minmax with 3 sensors. Augmented data in double proportion, to 6000 hz")
        file_full_raws = basename+"fullraws_aug60_cola2_noaer_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_train = basename+"fullraws_aug60_cola2_noaer_train_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_val = basename+"fullraws_aug60_cola2_noaer_val_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_test = basename+"fullraws_aug60_cola2_noaer_test_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"

    elif dataset == 'cola2-3_augx2_600':
        print("COLA dataset scaled minmax with 3 sensors. Augmented data in double proportion, to 600 hz")
        file_full_raws = basename+"fullraws_aug600_cola2_noaer_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_train = basename+"fullraws_aug600_cola2_noaer_train_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_val = basename+"fullraws_aug600_cola2_noaer_val_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_test = basename+"fullraws_aug600_cola2_noaer_test_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
    
    elif dataset == 'cola2-3_augx2_1800':
        print("COLA dataset scaled minmax with 3 sensors. Augmented data in double proportion, to 200 hz")
        file_full_raws = basename+"fullraws_aug1800_cola2_noaer_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_train = basename+"fullraws_aug1800_cola2_noaer_train_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_val = basename+"fullraws_aug1800_cola2_noaer_val_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"
        file_data_test = basename+"fullraws_aug1800_cola2_noaer_test_acc_gyr_mag_minmaxnorm_coord__5sec_100hz_7act_origaug.csv"

    elif dataset == 'cola2-2_augx2_60':
        print("COLA dataset scaled minmax with 2 sensors. Augmented data in double proportion, to 6000 hz")
        file_full_raws = basename+"fullraws_aug60_cola2_noAER_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_train = basename+"fullraws_aug60_cola2_noAER_train_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_val = basename+"fullraws_aug60_cola2_noAER_val_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_test = basename+"fullraws_aug60_cola2_noAER_test_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
    
    elif dataset == 'cola2-2_augx2_600':
        print("COLA dataset scaled minmax with 2 sensors. Augmented data in double proportion, to 600 hz")
        file_full_raws = basename+"fullraws_aug600_cola2_noAER_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_train = basename+"fullraws_aug600_cola2_noAER_train_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_val = basename+"fullraws_aug600_cola2_noAER_val_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_test = basename+"fullraws_aug600_cola2_noAER_test_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"

    elif dataset == 'cola2-2_augx2_1800':
        print("COLA dataset scaled minmax with 2 sensors. Augmented data in double proportion, to 200 hz")
        file_full_raws = basename+"fullraws_aug1800_cola2_noAER_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_train = basename+"fullraws_aug1800_cola2_noAER_train_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_val = basename+"fullraws_aug1800_cola2_noAER_val_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"
        file_data_test = basename+"fullraws_aug1800_cola2_noAER_test_acc_gyr_minmax_coord__5sec_100hz_7act_infoaugcpy_in1step_origaug.csv"

    elif dataset=='haruml':
        folder = "../haruml"
        file_full_raws = ""

        file_data_train=basename+"fullraws_har-uml20_train_accgyr_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_har-uml20_val_accgyr_5sec_100hz_7act.csv"
        file_data_test= basename+"fullraws_har-uml20_test_accgyr_5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_har-uml20__accgyr_5sec_10u_100hz.csv"

        if (magni==True):
            file_data_train=basename+"fullraws_har-uml20_train_accgyr_magni_2sec_100hz_7act.csv"
            file_data_val= basename+"fullraws_har-uml20_val_accgyr_magni_2sec_100hz_7act.csv"
            file_data_test= basename+"fullraws_har-uml20_test_accgyr_magni_2sec_100hz_7act.csv"
            file_full_raws = basename+"fullraws_har-uml20__accgyr_magni_2sec_10u_100hz.csv"

            file_data_train="files2/fullraws_har-uml20_train_accgyr_magni_5sec_100hz_7act.csv"
            file_data_val= "files2/fullraws_har-uml20_val_accgyr_magni_5sec_100hz_7act.csv"
            file_data_test= "files2/fullraws_har-uml20_test_accgyr_magni_5sec_100hz_7act.csv"
            file_full_raws = "files2/fullraws_har-uml20__accgyr_magni_5sec_10u_100hz.csv"

    elif dataset=='haruml2': # haruml-lit'
        file_full_raws = basename+"fullraws_har-uml20__accgyr_norm_coord_5sec_10u_100hz.csv"
        file_data_train=basename+"fullraws_har-uml20_train_accgyr_norm_coord_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_har-uml20_val_accgyr_norm_coord_5sec_100hz_7act.csv"
        file_data_test= basename+"fullraws_har-uml20_test_accgyr_norm_coord_5sec_100hz_7act.csv"

    elif dataset=='haruml-lit':
        file_data_train = basename+"fullraws_har-uml20_train_accgyr_norm_coord_5sec_100hz_7act.csv"
        file_data_val = basename+"fullraws_har-uml20_val_accgyr_norm_coord_5sec_100hz_7act.csv"
        file_data_test = basename+"fullraws_har-uml20_test_accgyr_norm_coord_5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_har-uml20__accgyr_norm_coord_5sec_10u_100hz.csv"

    elif dataset=='haruml-3sns':
        file_full_raws = basename+"fullraws_har-uml20__acc_gyr_mag_normlit_coord_5sec_10u_100hz.csv"
        file_data_train=basename+"fullraws_har-uml20_train_acc_gyr_mag_normlit_coord_5sec_100hz.csv"
        file_data_val= basename+"fullraws_har-uml20_val_acc_gyr_mag_normlit_coord_5sec_100hz.csv"
        file_data_test= basename+"fullraws_har-uml20_test_acc_gyr_mag_normlit_coord_5sec_100hz.csv"

    ### basa ###
    elif dataset=='basa':
        print("BASA dataset without magni, 7030")
        file_data_train=basename+"fullraws_basa_train_acc_gyr__norm_coord_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_basa_val_acc_gyr__norm_coord_5sec_100hz_7act.csv"

        file_data_val= basename+"fullraws_basa_val_acc_gyr__norm_coord_5sec_100hz_7act_usr5.csv"
        file_data_test= basename+"fullraws_basa_test_acc_gyr__norm_coord_5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_basa_acc_gyr__norm_coord_5sec_100hz_7act.csv"

    elif dataset=='basa6040max':
        print("BASA dataset without magni, 6040")
        file_data_train=basename+"fullraws_basa_train_acc_gyr6040_norm_coord_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_basa_val_acc_gyr6040_norm_coord_5sec_100hz_7act.csv"
	
        file_data_val= basename+"fullraws_basa_val_acc_gyr6040_norm_coord_5sec_100hz_7act_usr5.csv"
        file_data_test= basename+"fullraws_basa_test_acc_gyr6040_norm_coord_5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_basa_acc_gyr6040_norm_coord_5sec_100hz_7act.csv"

    elif dataset=='basa6040lit':
        print("BASA dataset without magni, 6040-max")
        file_data_train=basename+"fullraws_basa_train_acc_gyr6040-max__norm_coord_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_basa_val_acc_gyr6040-max__norm_coord_5sec_100hz_7act.csv"
        
        file_data_val= basename+"fullraws_basa_val_acc_gyr6040-max__norm_coord_5sec_100hz_7act_usr5.csv"
        file_data_test= basename+"fullraws_basa_test_acc_gyr6040-max__norm_coord_5sec_100hz_7act.csv"
        file_full_raws = basename+"fullraws_basa_acc_gyr6040-max__norm_coord_5sec_100hz_7act.csv"

    elif dataset=='mhealth':
        file_full_raws = basename+"fullraws_mhealth_acc_gyr_norm__coord_5sec_100hz_12act.csv"
        file_data_train=basename+"fullraws_mhealth_train_acc_gyr_norm__coord_5sec_100hz_12act.csv"
        file_data_val= basename+"fullraws_mhealth_val_acc_gyr_norm__coord_5sec_100hz_12act.csv"
        file_data_test= basename+"fullraws_mhealth_test_acc_gyr_norm__coord_5sec_100hz_12act.csv"
    
    elif dataset=='mhealth-agm':
        file_full_raws = basename+"fullraws_mhealth_acc_gyr_mag_norm__coord_5sec_100hz_12act.csv"
        file_data_train=basename+"fullraws_mhealth_train_acc_gyr_mag_norm__coord_5sec_100hz_12act.csv"
        file_data_val= basename+"fullraws_mhealth_val_acc_gyr_mag_norm__coord_5sec_100hz_12act.csv"
        file_data_test= basename+"fullraws_mhealth_test_acc_gyr_mag_norm__coord_5sec_100hz_12act.csv"
    
    elif dataset=='pamap2-agm': # pamap without normalization, without interpolation
        file_full_raws = basename+"fullraws_pamap_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_train=basename+"fullraws_pamap_train_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_val= basename+"fullraws_pamap_val_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_test= basename+"fullraws_pamap_test_acc_gyr_mag__coord_5sec_100hz_9act.csv"

    elif dataset=='pamap2-agm-inter': # pamap without normalization, only the necessary interpolation
        file_full_raws = basename+"fullraws_pamap_inter_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_train=basename+"fullraws_pamap_inter_train_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_val= basename+"fullraws_pamap_inter_val_acc_gyr_mag__coord_5sec_100hz_9act.csv"
        file_data_test= basename+"fullraws_pamap_inter_test_acc_gyr_mag__coord_5sec_100hz_9act.csv"
    
    # HarUml
    elif dataset=='haruml-agm':
        file_full_raws = basename+"fullraws_haruml20__acc_gyr_mag__coord_5sec_10u_100hz.csv"
        file_data_train=basename+"fullraws_haruml20_train_acc_gyr_mag__coord_5sec_100hz_7act.csv"
        file_data_val= basename+"fullraws_haruml20_val_acc_gyr_mag__coord_5sec_100hz_7act.csv"
        file_data_test= basename+"fullraws_haruml20_test_acc_gyr_mag__coord_5sec_100hz_7act.csv"

	### mHealth ###
    elif dataset=='mhealth-agm-cub':
        file_full_raws = basename+"fullraws_mhealth_acc_gyr_mag__intercub_coord_5sec_100hz_12act.csv"
        file_data_train=basename+"fullraws_mhealth_train_acc_gyr_mag__intercub_coord_5sec_100hz_12act.csv"
        file_data_val= basename+"fullraws_mhealth_val_acc_gyr_mag__intercub_coord_5sec_100hz_12act.csv"
        file_data_test= basename+"fullraws_mhealth_test_acc_gyr_mag__intercub_coord_5sec_100hz_12act.csv"
        
    return file_data_train, file_data_val, file_data_test, file_full_raws