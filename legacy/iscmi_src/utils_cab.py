import torch
import numpy as np
import pandas as pd
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


"""
Compute all metrics
"""
def compute_all_metrics(y_test, predictions, dataset, time_, subdirectory, metric_results_, split='test'):
    one_hot_predictions = predictions.argmax(1)
    n_classes = len(np.unique(y_test))
    LABELS = get_class_names(dataset)
    cms = []

    print("")
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
        
        file.write("\nF1-score per class (None):")
        print("F1-score per class (None):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average=None)
        metr = np.round(metr, 4)
        print(metr)
        file.write(str(LABELS))
        file.write(str(metr))
        
        file.write("\nF1-score per class (weighted):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="weighted")
        metr = np.round(metr, 4)
        file.write(str(LABELS))
        file.write(str(metr))

        file.write("\nF1-score per class (macro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="macro")
        metr = np.round(metr, 4)
        file.write(str(LABELS))
        file.write(str(metr))

        file.write("\nF1-score per class (micro):")
        metr = metrics.f1_score(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes), average="micro")
        metr = np.round(metr, 4)
        file.write(str(LABELS))
        file.write(str(metr))
        
        file.write("\nclassification_report:")
        report = classification_report(one_hot(y_test, n_classes), one_hot(one_hot_predictions, n_classes))
        file.write(report)

        # Compute sensitivity and specificity for each class
        # Step 1: Compute the confusion matrix
        cm = metrics.confusion_matrix(y_test, one_hot_predictions)

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

        # Step 3: Print results
        file.write("\n")    
        file.write(str(LABELS)) 
        file.write(f"\nSensitivity per class:, {sensitivities}")
        file.write(f"\nSpecificity per class:, {specificities}")
        print()
        print(f"Sensitivity per class:, {sensitivities}")
        print(f"Specificity per class:, {specificities}")
    
    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, one_hot_predictions)
    
    cms.append(confusion_matrix)
    
    # Compute normalized confusion matrix
    normalised_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Plot Confusion Matrix:
    width = 6
    height = 6
    axis_size = 10
    ticks_size = 8
    fontsize = 8
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
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    
    plt.savefig(subdirectory + '/CM_' + split + '_' + dataset+'_'+time_+'.png')
    

    # NORMALIZE
    path_fig = subdirectory + '/CMn_' + split + '_' + dataset+'_'+time_+'.png'
    normalised_confusion_matrix = normalised_confusion_matrix * 100

    plt.figure(figsize=(width, height))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
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
    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.88)

    plt.savefig(path_fig)

    
    # ROC CURVE
    # Convert tensors to numpy arrays for scikit-learn compatibility
    y_preds = one_hot_predictions
    y_test_np = y_test #.numpy()
    y_preds_np = y_preds #.numpy()

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
    
    
"""
Overlap functtion
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
    for u in indexes_users.keys():
        for c in indexes.keys():
            # Common values between indexes_users and indexes
            index = np.intersect1d(indexes_users[u], indexes[c])
            samples = data[index]
            samples = samples.reshape(samples.shape[0]*samples.shape[1], samples.shape[2])
            #print("Clase:", c)
            new_data_class, n = overlap_class(new_data_class, samples, shift, window_size=data.shape[1])
            new_labels = new_labels + list(np.repeat(c, n))

    new_labels = np.array(new_labels)
    new_data_class = np.array(new_data_class)
    return new_data_class, new_labels


# Overlapping
def overlap_class( new_data_class, samples, overlap_shift, window_size):
    init = 0
    n = 0
    while(init <= (samples.shape[0]-window_size)):
        new_data_class.append(samples[init:init+window_size])
        #init = init + (window_size//2) # init + (window_size * overlap_shift) # ex. overlap_shift=0.8
        init = init + int(window_size - (window_size * overlap_shift))
        n = n + 1
    return new_data_class, n

# normalise
def normalise_all_zscore(data, mean, std):
    """    Normalise data (Z-normalisation)    """

    return ((data - mean) / std)



# Get processed fold
def get_processed_fold(df, dataset, subdirectory, sensors, fold, seg5, overlap_shift, feat=9):
    file_users_split = subdirectory+'/'+dataset+'_'+str(fold)+'.txt'
    x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt = split_data_val(df, 
                                                                                                                                                  dataset, 
                                                                                                                                                  file_users_split, 
                                                                                                                                                  test_size=0.3)

    if (sensors == 2):
        dict_arrays = get_data_arrays(x_train, y_train, x_val, y_val, x_test, y_test, dataset, False)
    elif (sensors == 3):
        dict_arrays, users_train, users_val, users_test = get_data_arrays_3sns(x_train, y_train, data_train[0], 
                                                                x_val, y_val, data_val[0], 
                                                                x_test, y_test, data_test[0], 
                                                                dataset, seg5, feat=feat, shuffle_data = False, axis=True)
        
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
    
    train_X, train_Y = overlap_data(train_X, train_Y, users_train, overlap_shift)
    val_X, val_Y = overlap_data(val_X, val_Y, users_val, overlap_shift)
    test_X, test_Y = overlap_data(test_X, test_Y, users_test,overlap_shift)

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


# Split data
def split_data_val(df,  dataset, file_users_split, test_size=0.2): # per users
    
    num_activities = np.unique(df[2])
    dict2 = {}
    for i in range(len(num_activities)):
        dict2[num_activities[i]] = i
    df = df.replace({2:dict2})
    
    uuids = np.unique(df[0])

    # Fixed
    splits = dic_datasets["fixed_users"]
    uuid_train = splits['train_users']
    uuid_val = splits['val_users']
    uuid_test = splits['test_users']

    data_train = df[df[0].isin(uuid_train)]
    data_test = df[df[0].isin(uuid_test)]
    data_val = df[df[0].isin(uuid_val)]
    
    # Print
    print("uuid_train: ",uuid_train)
    print("uuid_val: ",uuid_val)
    print("uuid_test: ",uuid_test)
        
    # Save uuid_train, uuid_val, uuid_test in one txt file
    splits_txt = str(uuid_train) + "\n" + str(uuid_val) + "\n" + str(uuid_test) + "\n"


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

    return x_train, y_train, x_val, y_val, x_test, y_test, data_train, data_val, data_test, uuid_train, uuid_val, uuid_test, splits_txt
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
    c = c.squeeze()
    return c


# Read full_raws
def read_full_raws(saved_file): 
    print("\nReading full_raws from file: ", saved_file)
    #full_raws = pd.read_csv(saved_file, header=0)
    full_raws = pd.read_csv(saved_file, header=None)
    print("full_raws shape:", full_raws.shape)
    print()
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



"""
Resize dataset to 3-dimensional vectors.
Each vector represent 1-second of activity, this
is the input vector to the model.
"""
def resize_data_axis3sns(dict_arrays):
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
Resize dataset to 3-dimensional vectors.
Each vector represent 5-second of activity, this
is the input vector to the model.
"""
def resize_data_axis3sns_5sec(dict_arrays, feat=9):
    x_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    x_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    x_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    
    d2 = x_train.shape[-2] // feat
    if ( feat==18):
        d2 = 500
    x_train = np.reshape(x_train, (-1, d2, feat))
    x_val = np.reshape(x_val, (-1, d2, feat))
    x_test = np.reshape(x_test, (-1, d2, feat))

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    dict_arrays = {'x_train': x_train, 'y_train': y_train, 
                'x_val': x_val, 'y_val': y_val, 
                'x_test': x_test, 'y_test':y_test}
                
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
def get_data_arrays_3sns(x_train, y_train, users_train, 
                        x_valid, y_valid, users_val, 
                        x_test, y_test, users_test, 
                        dataset, seg5, feat=9, shuffle_data=False, axis=False):
    
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train =  y_train.astype(np.float32)
    y_valid  = y_valid.astype(np.float32)
    y_test =  y_test.astype(np.float32)
    
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_valid = x_valid.reshape((x_valid.shape[0], x_valid.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    users_train = np.array(users_train)
    users_val = np.array(users_val)
    users_test = np.array(users_test)

    if shuffle_data == True:
        x_train , y_train = shuffle(x_train, y_train)
        x_valid , y_valid = shuffle(x_valid, y_valid)
        x_test , y_test = shuffle(x_test, y_test)

    dict_arrays = {'x_train': x_train, 'y_train': y_train,
                'x_val': x_valid, 'y_val': y_valid, 
                'x_test': x_test, 'y_test':y_test}
    
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


"""
One hot encoding
"""
def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


"""
Get class names
"""
def get_class_names(dataset):
    
    if dataset == 'eatdrinkanother' or dataset == 'eatdrinkanother5':
        classes_name = np.array(['0', '1', '2'])
        classes_name = np.array(['DRINK', 'EAT', 'ANOTHER'])
        
    else:
        print("Dataset not recognized")
        exit()
        
    return classes_name


"""
Get raw data
"""
def get_raw_datasets(dataset, magni):
    basename = "data/"
    
    file_data_train = None
    file_data_val = None
    file_data_test = None
        
    if dataset=='eatdrinkanother':
        file_full_raws = basename+"fullraws3_vivabem012_drink0eat1another2_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv"
    
    elif dataset=='eatdrinkanother5':
        file_full_raws = basename+'fullraws3_eatdrinkanother5th_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv'
        
    else:
        print("Dataset not recognized")
        exit()
        
    return file_data_train, file_data_val, file_data_test, file_full_raws

