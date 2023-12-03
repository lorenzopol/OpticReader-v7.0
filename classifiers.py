from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


import os
import cv2
import random
import pickle

import custom_utils as cu


def my_train_test_split(x, y, test_size):
    zipped_dataset = list(zip(x, y))
    random.shuffle(zipped_dataset)
    x_train = []
    y_train = []

    x_test = []
    y_test = []
    for i in zipped_dataset:
        if random.random() < test_size:
            x_test.append(i[0])
            y_test.append(i[1])
        else:
            x_train.append(i[0])
            y_train.append(i[1])
    return x_train, x_test, y_train, y_test


def count_file(dataset_files):
    file_counter = {
        "CA": 0,
        "CB": 0,
        "QA": 0,
        "QB": 0,
        "QS": 1
    }
    for file in dataset_files:
        file_counter[file[:2]] += 1
    return file_counter


def train_svm_knn(path_to_dataset, full=False):
    """full implies that all 5 categories are being trained on. If set to false, only Q* will be used"""
    # category = ("QB", "QS", "QA", "CB", "CA")
    #               0     1     2     3     4
    filename_enum = {
        "QB": 0,
        "QS": 1,
        "QA": 2,
        "CB": 3,
        "CA": 4
    }
    neighbours = 11

    imgs_list = []
    filename_list = []
    for f in os.listdir(path_to_dataset):
        pixel_dump = []
        if os.path.isfile(os.path.join(path_to_dataset, f)):
            if not full and (f.startswith("CA") or f.startswith("CB")):
                continue
            in_img = cv2.imread(os.path.join(path_to_dataset, f))
            gray_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY)
            cropped_to_bound = cu.crop_to_bounding_rectangle(gray_img)
            scaled_img = cv2.resize(cropped_to_bound, (18, 18))
            for row in scaled_img:
                for pixel in row:
                    pixel_dump.append(pixel)
            imgs_list.append(pixel_dump)
            filename_list.append(filename_enum.get(f[:2]))

    x_train, x_test, y_train, y_test = my_train_test_split(imgs_list, filename_list, 0.2)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM ccuracy: {accuracy}")

    knn_classifier = KNeighborsClassifier(n_neighbors=neighbours)
    knn_classifier.fit(x_train, y_train)
    y_pred_knn = knn_classifier.predict(x_test)

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"k-NN Accuracy: {accuracy_knn} for {neighbours =}")
    return svm_classifier, knn_classifier


def save_model(model, filename):
    save_path = os.getcwd()
    with open(os.path.join(save_path, filename), "wb") as cfile:
        pickle.dump(model, cfile)


def load_model(full_path_to_model):
    with open(full_path_to_model, 'rb') as cfile:
        model = pickle.load(cfile)
    return model


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), "train_and_test_data_reduced")

    svm_classifier, knn_classifier = train_svm_knn(dataset_path)
    save_model(svm_classifier, "svm_model")
    save_model(knn_classifier, "knn_model")
