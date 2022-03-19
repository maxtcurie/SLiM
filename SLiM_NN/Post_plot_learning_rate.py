import matplotlib.pyplot as plt 

def plot_hist(history):
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    
    
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    plot_acc=True
    try:
        acc=history.history['accuracy']
        val_acc=history.history['val_accuracy']
    except:
        try:
            acc=history.history['acc']
            val_acc=history.history['val_acc']
        except:
            print('No Accuracy data')
            plot_acc=False

    if plot_acc:
        epochs=range(len(acc)) # Get number of epochs
        #------------------------------------------------
        # Plot training and validation accuracy per epoch
        #------------------------------------------------
        plt.plot(epochs, acc, 'r', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        plt.title('Training and validation accuracy')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        print("")

    

    plot_loss=True
    try:
        loss=history.history['loss']
        val_loss=history.history['val_loss']
    except:
        print('No loss data')
        plot_loss=False

    if plot_loss:
        epochs=range(len(loss)) # Get number of epochs
        #------------------------------------------------
        # Plot training and validation loss per epoch
        #------------------------------------------------
        plt.plot(epochs, loss, 'r', label="Training Loss")
        plt.plot(epochs, val_loss, 'b', label="Validation Loss")
        plt.title('Training and validation loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()  



    plot_mae=True
    try:
        mae=history.history['mean_absolute_error']
        val_mae=history.history['val_mean_absolute_error']
    except:
        try:
            mae=history.history['mae']
            val_mae=history.history['val_mae']
        except:
            print('No mean_absolute_error data')
            plot_mae=False


    if plot_mae:
        epochs=range(len(mae)) # Get number of epochs
        #------------------------------------------------
        # Plot training and validation loss per epoch
        #------------------------------------------------
        plt.plot(epochs, mae, 'r', label="Training Mean Absolute Error")
        plt.plot(epochs, val_mae, 'b', label="Validation Mean Absolute Error")
        plt.title('Training and validation Mean Absolute Error')
        plt.xlabel('epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.show()  

    
    