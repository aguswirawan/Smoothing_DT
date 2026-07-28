import sys
import numpy as np
import scipy.io as sio
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_features(band_index):
    feature_index = np.empty(0)
    for i in band_index:
        band = np.array(range((i-1)*14,i*14))
        feature_index = np.append(feature_index,band)
    feature_index = list(map(int,feature_index))
    return feature_index

def get_vector_deviation(vector1, vector2):
    return vector1 - vector2

def get_dataset_deviation(trial_data, base_data, data_seconds_list):
    new_dataset = np.empty([0, 56])
    second_now = 0
    for i, seconds in enumerate(data_seconds_list[0]):
        for j in range(int(seconds)):
            new_record = get_vector_deviation(trial_data[j + second_now], base_data[i]).reshape(1, 56)
            new_dataset = np.vstack([new_dataset, new_record])
        second_now += int(seconds)
    return new_dataset

def preprocess_six_labels(label_a, label_b, label_c):
    if label_a == 1 and label_b == 1 and label_c == 1:
        return 0
    elif label_a == 0 and label_b == 0 and label_c == 0:
        return 1
    elif label_a == 1 and label_b == 1 and label_c == 0:
        return 2
    elif label_a == 0 and label_b == 0 and label_c == 1:
        return 3
    elif label_a == 1 and label_b == 0 and label_c == 0:
        return 4
    elif label_a == 0 and label_b == 1 and label_c == 1:
        return 5

if __name__ == '__main__':
    args = sys.argv[:]
    with_or_without = "with"
    result = np.empty([0,15])
    
    # Inisialisasi DataFrame untuk menyimpan metrik per label
    metrics_df_per_label = pd.DataFrame(columns=['Label', 'Accuracy', 'Error', 'Precision', 'Recall', 'F1 Score'])
    
    for sub_id in range(1,24):
        sub_id = "%02d" % sub_id
        sub = "P"+str(sub_id)
        print("processing ",sub )
        file = sio.loadmat("Dataset_1D/DE_"+sub+".mat")
        X = file["data"]
        
        if with_or_without=="with":
            base_data = file["base_data"]
            data_seconds_list = file["data_seconds_list"]
            X = get_dataset_deviation(X,base_data,data_seconds_list)
            X = preprocessing.scale(X,axis=1, with_mean=True,with_std=True,copy=True)
        
        y_valence = np.squeeze(file["valence_labels"].transpose())
        y_arousal = np.squeeze(file["arousal_labels"].transpose())
        y_dominance = np.squeeze(file["dominance_labels"].transpose())
        y = list(map(preprocess_six_labels, y_arousal, y_valence, y_dominance))
        y = np.array(y)
        
        index = np.array(range(0,len(y)))
        np.random.shuffle(index)
        input_X = X[index]
        y = y[index]

        fold = 10
        dictionary = {"band1":[1],"band2":[2],"band3":[3],"band4":[4],
                "band12":[1,2],"band13":[1,3],"band14":[1,4],"band23":[2,3],"band24":[2,4],"band34":[3,4],
                "band123":[1,2,3],"band124":[1,2,4],"band134":[1,3,4],"band234":[2,3,4],
                "band1234":[1,2,3,4]}
        
        acc_list = np.empty(0)
        count = 0
        metrics_data = []  # Inisialisasi untuk menyimpan metrik
        
        for key in sorted(dictionary.keys()):
            mean_accuracy = 0
            feature_index = get_features(dictionary[key])
            X = input_X[:,feature_index]
            
            for curr_fold in range(fold):
                fold_size = X.shape[0]//fold
                indexes_list = [i for i in range(len(X))]
                indexes = np.array(indexes_list)
                split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
                split = np.array(split_list)
                test_x = X[split] 
                test_y = y[split]

                split = np.array(list(set(indexes_list)^set(split_list)))
                train_x = X[split]
                train_y = y[split]
                
                clf = DecisionTreeClassifier(max_depth=20)
                clf.fit(train_x,train_y)

                Z = clf.predict(test_x)
                
                # Calculate evaluation metrics
                accuracy_value = accuracy_score(test_y, Z)
                error = 1 - accuracy_value
                precision = precision_score(test_y, Z, average=None) # Precision per label
                recall = recall_score(test_y, Z, average=None) # Recall per label
                f1 = f1_score(test_y, Z, average=None) # F1 per label

                mean_accuracy += accuracy_value
                
                # Simpan metrik ke dalam struktur data
                for label, prec, rec, f1_score_label in zip(range(6), precision, recall, f1):
                    metrics_data.append({
                        'Label': label,
                        'Accuracy': accuracy_value,
                        'Error': error,
                        'Precision': prec,
                        'Recall': rec,
                        'F1 Score': f1_score_label
                    })
            
            count += 1
            acc_list = np.append(acc_list,mean_accuracy/fold*100)
        
        acc_list = acc_list[[0,8,12,14,1,5,7,9,11,13,2,4,6,10,3]]
        result = np.vstack([result,acc_list])
        
        # Simpan DataFrame metrik ke dalam file Excel
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel("Result_All_Movement_6Kelas_Gausian/Parameter_Evaluation/"+with_or_without+"6Class_metrics_"+sub+".xlsx", index=False)

        print(acc_list)
        print("Accuracy:", accuracy_value)
        print("Error:", error)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        
        accuracy = pd.DataFrame(result)
        accuracy.columns = ["θ","α","β","γ","θ+α","θ+β","θ+γ","α+β","α+γ","β+γ",
          "θ+α+β","θ+α+γ","θ+β+γ","α+β+γ","θ+α+β+γ"]

        
        
        # Simpan DataFrame ke file Excel
        writer = pd.ExcelWriter("Result_All_Movement_6Kelas_Gausian/"+with_or_without+"6Class.xlsx")
        accuracy.to_excel(writer, 'result', index=False)
        writer.close()
        
        # Agregasi metrik per label
        metrics_per_label = metrics_df.groupby('Label').mean()
        metrics_df_per_label = pd.concat([metrics_df_per_label, metrics_per_label])

    # Simpan DataFrame metrik per label ke dalam file Excel
    metrics_df_per_label.to_excel("Result_All_Movement_6Kelas_Gausian/"+with_or_without+"6Class_metrics_per_label.xlsx")
