import numpy as np
import time
from tensorflow import keras
import util
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from keras.utils import np_utils
# Assuming 'util' is a custom module you have for subset selection
import util  
import matplotlib.pyplot as plt

plt.ion()
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

print("After Reshaping",X_train.shape)
print("After Reshaping",X_test.shape)

num_classes, smtk = 10, 0
Y_train_nocat = Y_train
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

print(Y_train.shape)
print(Y_test.shape)


# Parameters
batch_size = 32
subset, random = False, False  # greedy
subset_size = 0.3 if subset else 1.0
epochs = 15
reg = 1e-4
runs = 1
save_subset = False
num_clients = 5  # For Federated Learning
folder = f'/mnist'

print("Batch Size:", batch_size,"Subset Size:",subset_size,"Number of Clients:",num_clients)
# Define model architecture
def create_model():
    model = Sequential([
        Dense(100, input_shape=(784,), kernel_regularizer=l2(reg), activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Initialize arrays to track metrics
train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_time = np.zeros((runs, epochs))
grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
not_selected = np.zeros((runs, epochs))
times_selected = np.zeros((runs, len(X_train)))
best_acc = 0

if save_subset:
    print("Length of X_train", len(X_train))
    print("Subset Size is :",subset_size)
    B = int(subset_size * len(X_train))
    print(B)
    selected_ndx = np.zeros((runs, epochs, B))
    #print("Indices are",selected_ndx)
    print("Indices of Subsets",selected_ndx.shape)
    selected_wgt = np.zeros((runs, epochs, B))
    #print("Weights of Subset is",selected_wgt)
    print("Weights of Subsets",selected_wgt.shape)

for run in range(runs):
    model = create_model()
    X_subset = X_train
    Y_subset = Y_train
    W_subset = np.ones(len(X_subset))
    ordering_time,similarity_time, pre_time = 0, 0, 0
    loss_vec, acc_vec, time_vec = [], [], []
    for epoch in range(0, epochs):
        model = create_model()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        num_batches = int(np.ceil(X_subset.shape[0] / float(batch_size)))

        for index in range(num_batches):
            X_batch = X_subset[index * batch_size:(index + 1) * batch_size]
            #print(X_batch[0])
            #print(X_batch.shape) (32,784)
            Y_batch = Y_subset[index * batch_size:(index + 1) * batch_size]
            #print(Y_batch.shape)
            W_batch = W_subset[index * batch_size:(index + 1) * batch_size]
            #print(W_batch.shape)

            start = time.time()
            history = model.train_on_batch(X_batch, Y_batch, sample_weight=W_batch)
            train_time[run][epoch] += time.time() - start

        if subset:
            if random:
                # indices = np.random.randint(0, len(X_train), int(subset_size * len(X_train)))
                indices = np.arange(0, len(X_train))
                np.random.shuffle(indices)
                indices = indices[:int(subset_size * len(X_train))]
                W_subset = np.ones(len(indices))
            else:
                start = time.time()
                _logits = model.predict(X_train)
                pre_time = time.time() - start
                features = _logits - Y_train
                

                indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                    int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)

                W_subset = W_subset / np.sum(W_subset) * len(W_subset)  # todo

            if save_subset:
                selected_ndx[run, epoch], selected_wgt[run, epoch] = indices, W_subset

            grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = ordering_time, similarity_time, pre_time
            times_selected[run][indices] += 1
            not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
        else:
            pred_time = 0
            indices = np.arange(len(X_train))

        X_subset = X_train[indices, :]
        Y_subset = Y_train[indices]

        start = time.time()
        score = model.evaluate(X_test, Y_test, verbose=1)
        eval_time = time.time()-start

        start = time.time()
        score_loss = model.evaluate(X_train, Y_train, verbose=1)
        print(f'eval time on training: {time.time()-start}')

        test_loss[run][epoch], test_acc[run][epoch] = score[0], score[1]
        train_loss[run][epoch], train_acc[run][epoch] = score_loss[0], score_loss[1]
        best_acc = max(test_acc[run][epoch], best_acc)

        grd = 'random_wor' if random else 'grd_normw'
        print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, test_acc: {test_acc[run][epoch]}, '
              f'loss: {train_loss[run][epoch]}, best_prec1_gb: {best_acc}, not selected %:{not_selected[run][epoch]}')
        
        client_models = []

        # Divide data among clients
        client_data = np.array_split(X_subset, num_clients), np.array_split(Y_subset, num_clients)

        for client_id in range(num_clients):
            X_client, Y_client = client_data[0][client_id], client_data[1][client_id]
            # Clone the model architecture
            client_model = keras.models.clone_model(model)
            # Copy the weights from the global model
            client_model.set_weights(model.get_weights())
            # Compile the cloned model with the same configuration as the original model
            client_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # Now you can train the compiled client model
            start = time.time()
            client_model.fit(X_client, Y_client, batch_size=batch_size, epochs=1, verbose=0)
            train_time[run][epoch] += time.time() - start
            client_models.append(client_model)

        # Federated Averaging
        new_weights = [np.mean([client_model.get_weights()[i] for client_model in client_models], axis=0) for i in range(len(client_models[0].get_weights()))]
        model.set_weights(new_weights)

        # Evaluation
        train_loss[run][epoch], train_acc[run][epoch] = model.evaluate(X_subset, Y_subset, verbose=0)
        test_loss[run][epoch], test_acc[run][epoch] = model.evaluate(X_test, Y_test, verbose=0)
        best_acc = max(best_acc, test_acc[run][epoch])
        print(f'Test Acc: {test_acc[run][epoch]}, Best Acc: {best_acc}')
        plt.figure(figsize=(10, 6))
        for run in range(train_loss.shape[0]):  # Iterating through each run
            plt.plot(range(epochs), train_loss[run], label=f'Run {run+1}')

        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig('/home/drishya/drishya/craig/training_loss.png', dpi=300)  # Saves the plot as a PNG file with 300 DPI

        # Optionally display the plot
        # plt.show()

        print('Plot saved as "training_loss_per_epoch.png". Please check the current directory.')
        
        plt.figure(figsize=(10, 6))
        for run in range(test_acc.shape[0]):  # Iterating through each run
            plt.plot(range(epochs), test_acc[run], label=f'Run {run+1}')

        plt.title('Test Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.savefig('/home/drishya/drishya/craig/test_accuracy.png', dpi=300)  # Saves the plot as a PNG file with 300 DPI

        # Optionally display the plot
        # plt.show()

        print('Plot saved as "test_accuracy_per_epoch.png". Please check the current directory.')

print('Final Best Accuracy:', best_acc)
