import matplotlib.pyplot as plt

def plot_training_history(history, subdirectory, time_, dataset):
    """
    Plot and save training history plots.

    Args:
        history: Keras history object
        subdirectory (str): Directory to save plots
        time_ (str): Timestamp
        dataset (str): Dataset name
    """
    acc_type = 'accuracy'
    val_type = 'val_accuracy'

    # Summarize history for accuracy complete
    plt.figure()
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.title(dataset + '- loss and accuracy')
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss & accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(f"{subdirectory}/loss-acc_comp{dataset}-{time_}.png")
    plt.close()

    # Summarize history for accuracy with limit in axis y
    plt.figure()
    plt.ylim(0, 2.0)
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.title(dataset + '- loss and accuracy')
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss & accuracy (--)')
    plt.xlabel('epoch')
    plt.ylim(0, 1)
    plt.legend(['train accuracy', 'val accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(f"{subdirectory}/loss-acc_{dataset}-{time_}.png")
    plt.close()

    # Plot for history for accuracy
    plt.figure()
    plt.ylim(0, 1)
    plt.title(dataset + ' - accuracy ')
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper right')
    plt.savefig(f"{subdirectory}/acc_{dataset}-{time_}.png")
    plt.close()

    # Plot for history for loss
    plt.figure()
    plt.ylim(0, 1.5)
    plt.title(dataset + ' - loss ')
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper right')
    plt.savefig(f"{subdirectory}/loss_{dataset}-{time_}.png")
    plt.close()