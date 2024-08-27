from src.preprocessing import *
from src.paillier import *
from src.encoding import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import pickle
from src.nn import *
import numpy as np
import argparse
import logging


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

dir_path = os.getcwd()
model_encrypted_path = dir_path + "/saved/encrypted"
model_normal_path = dir_path + "/saved/unencrypted"
logger = logging.getLogger(__name__)

def training(train_x, train_y, test_x, test_y, model = 1, saved = [], private_key = None):
    # Define the neural network architectyure
    output_size = train_y.shape[1]
    learning_rate = 0.01
    input_size = len(train_x[0])
    hidden_size = len(train_x[0])
    X_train = train_x
    X_test = test_x
    y_train = train_y
    y_test = test_y

    # Create an instance of the neural network
    if model == 0 :
        nn = MLP(input_size, hidden_size, output_size, private_key)
        logger.info('Using Multi-layer Perceptron')
    else :
        logger.info('Using Recurrent Neural Network')
        nn = RNN(input_size, hidden_size, output_size, private_key)

    if len(saved) != 0:
        nn.W1 = saved[0]
        nn.W2 = saved[1]
        nn.b1 = saved[2]
        nn.b2 = saved[3]            

    # Training loop
    epochs = 5
    lowest_loss = float('inf')
    print("Beginning training")
    for epoch in range(epochs):
        total_loss = 0 
        correct_train = 0
        correct_test = 0
        prev_hidden = None
        with tqdm(total=len(X_train), desc=f"Epoch {epoch}", unit=" samples") as pbar:
            for i in range(len(X_train)):
                # Forward pass
                output, prev_hidden = nn.forward(X_train[i], prev_hidden = prev_hidden)
                # Backward pass
                nn.backward(y_train[i], output, learning_rate, prev_hidden = prev_hidden)
                # Accumulate loss
                total_loss += nn.loss
                pbar.update(1)  # Update progress bar
                if np.argmax(output) == np.argmax(y_train[i]):
                    correct_train += 1
                pbar.set_description(f"Epoch {epoch} - Accuracy: {correct_train/(i+1):.4f} - Loss: {nn.loss:.4f}")
        #Print average accuracy per epoch
        train_acc = (correct_train / len(X_train))*100
        print(f"Epoch {epoch}: Train Accuracy = {train_acc}%")

        # Print average loss per epoch
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Testing

        with tqdm(total=len(X_test), desc=f"Epoch {epoch}", unit=" samples") as tbar:
            for i in range(len(X_test)):
                output, prev_hidden = nn.forward(X_test[i], prev_hidden = prev_hidden)
                result = np.argmax(output)
                if result == np.argmax(y_test[i]):
                    correct_test += 1
                tbar.update(1)  # Update progress bar
                tbar.set_description(f"Epoch {epoch} - Accuracy: {correct_test/(i+1):.4f}")
            test_acc = (correct_test / len(X_test))*100
            print(f"Epoch {epoch}: Test Accuracy = {test_acc}%")

        if lowest_loss > avg_loss:
          pre = [nn.W1, nn.W2, nn.b1, nn.b2]
          if private_key != None:
            with open(f"{model_encrypted_path}/encrypted_{model}.pkl", 'wb') as file:
              pickle.dump(pre, file)
          else :
            with open(f"{model_normal_path}/unencrypted_{model}.pkl", 'wb') as file:
              pickle.dump(pre, file)
def main():
    parser = argparse.ArgumentParser(description='A brief description of your program.')
    parser.add_argument('--state', type=int, help='unencrypted (0 as arg) or encrypted (1 as arg)', required = False, default = 1)
    parser.add_argument('--model', type=int, help='MLP (0 as arg) or RNN (1 as arg)', required = False, default = 0)

    args = parser.parse_args()


    filename1 = dir_path + "/datasets/kdd/kddcup.data_10_percent_corrected"
    filename2 = dir_path + "/datasets/kdd/corrected"

    dataset1 = read_data(filename1)
    dataset2 = read_data(filename2)
    nomial(dataset1, dataset2)

    dataset1['label'] = initlabel(dataset1)
    dataset2['label'] = initlabel(dataset2)

    num_features = ["duration","protocol_type","service","flag", "src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
        ]
    dataset1[num_features] = dataset1[num_features].astype(float)
    dataset1[num_features] = MinMaxScaler().fit_transform(dataset1[num_features].values)
    dataset2[num_features] = dataset2[num_features].astype(float)
    dataset2[num_features] = MinMaxScaler().fit_transform(dataset2[num_features].values)

    print(dataset1.describe())

    print(dataset1['label'].value_counts())

    labels1 = dataset1['label'].copy()
    print(labels1.unique())

    labels1[labels1 == 'normal.'] = 0
    labels1[labels1 == 'dos'] = 1
    labels1[labels1 == 'u2r'] = 2
    labels1[labels1 == 'r2l'] = 3
    labels1[labels1 == 'probe'] = 4
    dataset1['label'] = labels1

    labels2 = dataset2['label'].copy()
    print(labels2.unique())

    labels2[labels2 == 'normal.'] = 0
    labels2[labels2 == 'dos'] = 1
    labels2[labels2 == 'u2r'] = 2
    labels2[labels2 == 'r2l'] = 3
    labels2[labels2 == 'probe'] = 4
    dataset2['label'] = labels2

    train,test= read_data_set(dataset1,dataset2)

    try :
        with open(model_encrypted_path + f"encrypted_{args.model}.pkl", 'rb') as file:
            saved = pickle.load(file)
    except:
        saved = []

    try :
        with open(model_normal_path + f"unencrypted_{args.model}.pkl", 'rb') as file:
            saved = pickle.load(file)
    except:
        saved = []



    np.random.seed(42)
    train_size = 10000
    test_size = 2000

    indices = np.arange(len(train._segments))
    np.random.shuffle(indices)
    shuffled_features = np.array(train._segments)[indices]
    shuffled_labels = np.array(train._labels)[indices]
    test._segments = np.array(test._segments)

    shuffled_features = shuffled_features.reshape(shuffled_features.shape[0], 41)
    train_x = shuffled_features[:train_size]
    train_y = shuffled_labels[:train_size]
    test._segments = test._segments.reshape(test._segments.shape[0], 41)
    test_x = test._segments[:test_size]
    test_y = test._labels[:test_size]
     
    if args.state == 1 :
        key_gen = PaillierKeyGen()
        public_key = key_gen.publickey()
        private_key = key_gen.privatekey(public_key)
        min = 0

        encrypted_train = multi_encrypt(train_x.tolist(), min, len(train_x), public_key, max_processes = 5)
        encrypted_test = multi_encrypt(test_x.tolist(), min, len(test_x), public_key, max_processes = 5)
        logger.info(f"Example unencrypted data : {train_x[0][1]}")
        logger.info(f"Example from encrypted data : {encrypted_train[0][1].ciphertext()}")
        exp = encrypted_train[0][1]*3
        logger.info(f"Example after decryption multiplied by 3 : {private_key.decrypt(exp)}")
        training(train_x = encrypted_train, train_y = train_y, test_x = encrypted_test, test_y = test_y, model = args.model, saved = saved, private_key = private_key)
    else :
        training(train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y, model = args.model, saved = saved, private_key = None)
    

if __name__ == "__main__":
    main() 
