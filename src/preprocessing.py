import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, segments, labels, reshape = True):
          """Construct a Dataset
          one_hot arg is used only if fake_data is True. 'dtype' can be either unit9 or float32
          """
          '''
          dtype = dtypes.as_dtype(dtype).base_dtype
          if dtype not in (dtypes.uint8, dtypes.float32):
              raise TypeError('Invalid')
          '''
          self._num_examples = len(segments)
          self._segments = segments
          self._labels = labels
          self._epochs_completed = 0
          self._index_in_epoch = 0

    @property
    def segments(self):
        return self._segments

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next batch-size examples from this dataset"""

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed +=1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._segments = [self._segments[i] for i in perm]
            self._labels = [self._labels[i] for i in perm]

            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._segments[start:end], self._labels[start:end]



    def windows(self, data, size):
        start = 0
        while start < data.count():
            yield start, start + size
            start += size

def segment_signal(data, window_size = 1):

        segments = np.empty((0, window_size, 41))
        labels = np.empty((0))
        num_features = ["duration", "protocol_type","service","flag", "src_bytes", "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
        ]
        segments = np.asarray(data[num_features].copy())
        labels = data["label"]

        return segments, labels

def read_data(filename):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_srv_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        dataset = pd.read_csv(filename, header = None, names = col_names)
        return dataset

def normalize(dataset):
        mu = np.mean(dataset, axis = 0)
        sigma = np.std(dataset, axis = 0)
        return (dataset - mu)/sigma

def read_data_set(dataset1, dataset2, reshape = True):
        segments1, labels1 = segment_signal(dataset1)
        #labels1 = np.asarray(pd.get_dummies(labels1), dtype = np.int8)

        segments2, labels2 = segment_signal(dataset2)
        #labels2 = np.asarray(pd.get_dummies(labels2), dtype = np.int8)
        labels = np.asarray(pd.get_dummies(labels1._append([labels2])), dtype = np.int8)
        labels1 = labels[:len(labels1)]
        labels2 = labels[len(labels1):]
        train_x = segments1.reshape(len(segments1), 1, 1 ,41)
        train_y = labels1

        test_x = segments2.reshape(len(segments2), 1, 1 ,41)
        test_y = labels2

        train = Dataset(train_x.tolist(), train_y, reshape = reshape)
        test = Dataset(test_x.tolist(), test_y, reshape = reshape)
        return train,test

def initlabel(dataset):
    labels = dataset['label'].copy()
    labels[labels == 'back.'] = 'dos'
    labels[labels == 'buffer_overflow.'] = 'u2r'
    labels[labels == 'ftp_write.'] =  'r2l'
    labels[labels == 'guess_passwd.'] = 'r2l'
    labels[labels == 'imap.'] = 'r2l'
    labels[labels == 'ipsweep.'] = 'probe'
    labels[labels == 'land.'] = 'dos'
    labels[labels == 'loadmodule.'] = 'u2r'
    labels[labels == 'multihop.'] = 'r2l'
    labels[labels == 'neptune.'] = 'dos'
    labels[labels == 'nmap.'] = 'probe'
    labels[labels == 'perl.'] = 'u2r'
    labels[labels == 'phf.'] =  'r2l'
    labels[labels == 'pod.'] =  'dos'
    labels[labels == 'portsweep.'] = 'probe'
    labels[labels == 'rootkit.'] = 'u2r'
    labels[labels == 'satan.'] = 'probe'
    labels[labels == 'smurf.'] = 'dos'
    labels[labels == 'spy.'] = 'r2l'
    labels[labels == 'teardrop.'] = 'dos'
    labels[labels == 'warezclient.'] = 'r2l'
    labels[labels == 'warezmaster.'] = 'r2l'
    labels[labels == 'apache2.'] = 'dos'
    labels[labels == 'mailbomb.'] = 'dos'
    labels[labels == 'processtable.'] = 'dos'
    labels[labels == 'udpstorm.'] = 'dos'
    labels[labels == 'mscan.'] = 'probe'
    labels[labels == 'saint.'] = 'probe'
    labels[labels == 'ps.'] = 'u2r'
    labels[labels == 'sqlattack.'] = 'u2r'
    labels[labels == 'xterm.'] = 'u2r'
    labels[labels == 'named.'] = 'r2l'
    labels[labels == 'sendmail.'] = 'r2l'
    labels[labels == 'snmpgetattack.'] = 'r2l'
    labels[labels == 'snmpguess.'] = 'r2l'
    labels[labels == 'worm.'] = 'r2l'
    labels[labels == 'xlock.'] = 'r2l'
    labels[labels == 'xsnoop.'] = 'r2l'
    labels[labels == 'httptunnel.'] = 'r2l'
    return labels

def nomial(dataset1, dataset2):
    dataset = dataset1._append([dataset2])
    protocol1 = dataset1['protocol_type'].copy()
    protocol2 = dataset2['protocol_type'].copy()
    protocol_type = dataset['protocol_type'].unique()
    for i in range(len(protocol_type)):
        protocol1[protocol1 == protocol_type[i]] = i
        protocol2[protocol2 == protocol_type[i]] = i
    dataset1['protocol_type'] = protocol1
    dataset2['protocol_type'] = protocol2

    service1 = dataset1['service'].copy()
    service2 = dataset2['service'].copy()
    service_type = dataset['service'].unique()
    for i in range(len(service_type)):
        service1[service1 == service_type[i]] = i
        service2[service2 == service_type[i]] = i
    dataset1['service'] = service1
    dataset2['service'] = service2

    flag1 = dataset1['flag'].copy()
    flag2 = dataset2['flag'].copy()
    flag_type = dataset['flag'].unique()
    for i in range(len(flag_type)):
        flag1[flag1 == flag_type[i]] = i
        flag2[flag2 == flag_type[i]] = i

    dataset1['flag'] = flag1
    dataset2['flag'] = flag2
