from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale

import keras.models
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.regularizers import l2

from collections import Counter
import os
import random

import pickle
import imutils

import cv2
import numpy as np

import utils_evaluator as ue


def my_train_test_split(x, y, test_size, to_cls=False):
    zipped_dataset = list(zip(x, y))
    random.shuffle(zipped_dataset)
    x_train = []
    y_train = []

    x_test = []
    y_test = []
    if to_cls:
        for i in zipped_dataset:
            if random.random() < test_size:
                x_test.append(i[0])
                y_test.append(np.argmax(i[1]))
            else:
                x_train.append(i[0])
                y_train.append(np.argmax(i[1]))
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
    else:
        for i in zipped_dataset:
            if random.random() < test_size:
                x_test.append(i[0])
                y_test.append(i[1])
            else:
                x_train.append(i[0])
                y_train.append(i[1])
        shape_y, shape_x, *_ = x[0].shape
        return np.array(x_train).reshape(-1, shape_x, shape_y, 1), np.array(x_test).reshape(-1, shape_x, shape_y, 1), \
            np.array(y_train), np.array(y_test)


def count_file(dataset_files):
    file_counter = {
        "CA": 0,
        "CB": 0,
        "QA": 0,
        "QB": 0,
        "QS": 0
    }
    for file in dataset_files:
        file_counter[file.split("-")[0]] += 1
    return file_counter


def prepare_dataset(path_to_dataset, full=False, flatten_array=False):
    raw_imgs_filename = []
    dataset_container = []

    filename_enum = {
        "QB": 0,
        "QS": 1,
        "QA": 2,
        "CB": 3,
        "CA": 4
    }

    separated_dataset = {
        "QB": [],
        "QS": [],
        "QA": [],
        "CB": [],
        "CA": []
    }
    shuffled = separated_dataset.copy()

    imgs_list, filename_list = [], []

    for root, dirs, files in os.walk(path_to_dataset):
        for file in files:
            raw_imgs_filename.append(file)
            separated_dataset[file.split("-")[0]].append(file)

    file_counter = count_file(raw_imgs_filename)
    min_count = min(file_counter.values())

    for k, v in separated_dataset.items():
        sample_pop_num = min(min_count * 2, len(v)) if len(v) - min_count > 0 else min_count
        if full:
            shuffled[k] = random.sample(v, sample_pop_num)
        else:
            shuffled[k] = random.sample(v, min_count)

    for v in shuffled.values():
        dataset_container.extend(v)
    random.shuffle(dataset_container)

    for i, f in enumerate(dataset_container):
        print(f"[{i + 1} of {len(dataset_container)}] Processing {f}")
        if os.path.isfile(os.path.join(path_to_dataset, f)):
            in_img = cv2.imread(os.path.join(path_to_dataset, f))
            in_img = cv2.resize(in_img, (18, 18))
            gray_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2GRAY) / 255

            norm = 1 - minmax_scale(gray_img)
            if flatten_array:
                imgs_list.append(norm.flatten())
            else:
                imgs_list.append(norm)

            label, *_ = f.split("-")
            output = [0, 0, 0, 0, 0]
            output[filename_enum.get(label)] = 1
            filename_list.append(output)
    return np.array(imgs_list), np.array(filename_list)


def generate_full_dataset():
    raw_dataset_path = os.path.join(os.getcwd(), "dataset")
    raw_imgs_full_paths = [os.path.join(root, file) for root, dirs, files in os.walk(raw_dataset_path) for file in
                           files]
    path_to_models = os.getcwd()

    loaded_svm_classifier: SVC = load_model(os.path.join(path_to_models, "svm_model"))
    loaded_knn_classifier: KNeighborsClassifier = load_model(os.path.join(path_to_models, "knn_model"))
    exception_dict = {}
    for sim_id, abs_img_path in enumerate(raw_imgs_full_paths):
        print(f"[{sim_id + 1} of {len(raw_imgs_full_paths)}]Processing {abs_img_path}")
        try:
            BGR_img = cv2.imread(abs_img_path)
            BGR_SC_img = imutils.resize(BGR_img, height=700)
            BGR_SCW_img, transformed_begin_question_box_y, transformed_end_question_box_y = ue.warp_affine_img(
                BGR_SC_img)
            gray_img = cv2.cvtColor(BGR_SCW_img, cv2.COLOR_BGR2GRAY)

            # apply x shift because of bad distance between black cols and circles/squares and numbers
            cols_pos_x = ue.find_n_black_point_on_row(
                ue.one_d_row_slice(gray_img, transformed_end_question_box_y + ue.Globals.Y_SAMPLE_POS_FOR_CUTS),
                165)
            while len(cols_pos_x) < 5:
                cols_pos_x = ue.interpolate_missing_value(cols_pos_x, 105, 200)
            cols_pos_x[0] = cols_pos_x[
                                0] + ue.Globals.FIRST_X_CUT_SHIFT  # shift the first col to the right to compensate the fact that find_n_black_point_on_row returns the first black pixel
            x_cuts = ue.get_x_cuts(cols_pos_x)
            x_cuts.append(cols_pos_x[-1])
            y_cuts = ue.get_y_cuts(transformed_begin_question_box_y, transformed_end_question_box_y)
            crop_id = 0
            grab_next_QBs = 0
            for y_index in range(len(y_cuts) - 1):
                for x_index in range(len(x_cuts) - 1):
                    rng = random.random()
                    # if we are on a number col, skip it
                    if not (x_index - 1) % 7:
                        continue

                    question_number = ue.Globals.QUESTION_PER_COL * (x_index // 7) + (y_index + 1)

                    if question_number > 50:
                        continue

                    x_top_left = int(not x_index % 7) + x_cuts[x_index]
                    candidate_x_bottom_right = int(not x_index % 7) + x_cuts[x_index + 1]
                    x_bottom_right = x_top_left + ue.Globals.SQUARE_DIM if candidate_x_bottom_right - x_top_left > ue.Globals.SQUARE_DIM \
                        else candidate_x_bottom_right
                    y_top_left: int = y_cuts[y_index]
                    y_bottom_right: int = y_cuts[y_index + 1]

                    cropped = gray_img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
                    cropped_to_bound = ue.crop_to_bounding_rectangle(cropped)
                    area = cropped_to_bound.shape[0] * cropped_to_bound.shape[1]
                    if area > 50:
                        # resize only those crops that are big enough to not get blurred
                        resized_crop = cv2.resize(cropped_to_bound,
                                                  (ue.Globals.CLASSIFIER_IMG_DIM, ue.Globals.CLASSIFIER_IMG_DIM))
                        resized = True
                    else:
                        resized_crop = cv2.resize(cropped,
                                                  (ue.Globals.CLASSIFIER_IMG_DIM, ue.Globals.CLASSIFIER_IMG_DIM))
                        resized = False
                    predicted_category_index = ue.evaluate_square(resized_crop, x_index, loaded_svm_classifier,
                                                                  loaded_knn_classifier)

                    # if the crop is a QB
                    if grab_next_QBs == 0 and predicted_category_index == 0:
                        continue
                    if grab_next_QBs > 0 and predicted_category_index == 0:
                        grab_next_QBs -= 1

                    # if the crop is a QS or QA, we want to grab the next QB
                    if predicted_category_index in range(1, 3):
                        grab_next_QBs += 1
                    cv2.imwrite(
                        f"train_and_test_data_full/{ue.Globals.IDX_TO_EVAL_CODE[predicted_category_index]}-{sim_id}-{crop_id}-{int(resized)}-{x_index % 7}.png",
                        resized_crop)
                    crop_id += 1
        except Exception as e:
            exception_dict[abs_img_path] = e

    for k, v in exception_dict.items():
        print(f"Exception occurred at {k} with message {v}")


def new_train_svm_knn(path_to_dataset):
    """full implies that all 5 categories are being trained on. If set to false, only Q* will be used"""
    # category = ("QB", "QS", "QA", "CB", "CA")
    #               0     1     2     3     4

    neighbours = 3

    imgs_list, filename_list = prepare_dataset(path_to_dataset, flatten_array=True)
    x_train, x_test, y_train, y_test = my_train_test_split(imgs_list, filename_list, 0.33, to_cls=True)
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


def lenet_model(shape_x, shape_y):
    # hyperparameters
    input_shape = (shape_x, shape_y, 1)
    n_classes = 5
    l2_reg = 0.

    lenet = keras.models.Sequential()

    # 2 sets of CRP (Convolution, RELU, Pooling)
    lenet.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=input_shape, kernel_regularizer=l2(l2_reg)))
    lenet.add(Activation("relu"))
    lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    lenet.add(Conv2D(50, (5, 5), padding="same",
                     kernel_regularizer=l2(l2_reg)))
    lenet.add(Activation("relu"))
    lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected layers (w/ RELU)
    lenet.add(Flatten())
    lenet.add(Dense(500, kernel_regularizer=l2(l2_reg)))
    lenet.add(Activation("relu"))

    # Softmax (for classification)
    lenet.add(Dense(n_classes, kernel_regularizer=l2(l2_reg)))
    lenet.add(Activation("softmax"))

    # Return the constructed network
    lenet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return lenet


def alexnet_model(shape_x, shape_y):
    # hyperparameters
    input_shape = (shape_x, shape_y, 1)
    n_classes = 5
    l2_reg = 0.

    # alexnet implementation
    alexnet = keras.models.Sequential()
    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=input_shape,
                       padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(384, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(384, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))
    # Compile the model
    alexnet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


def train_cnn(model_name, path_to_dataset):
    imgs_list, filename_list = prepare_dataset(path_to_dataset, full=True)
    shape_y, shape_x, *_ = imgs_list[0].shape

    print(f"loaded dataset with length: {len(imgs_list)}")
    model_dict = {
        "lenet": lenet_model(shape_x, shape_y),
        "alexnet": alexnet_model(shape_x, shape_y),
    }
    model = model_dict.get(model_name)
    assert model is not None, "Model not found"

    x_train, x_test, y_train, y_test = my_train_test_split(imgs_list, filename_list, 0.33)
    model.fit(x_train, y_train, epochs=7)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", accuracy)

    model.save(f"{model_name}.h5")


def save_model(model, filename):
    save_path = os.getcwd()
    with open(os.path.join(save_path, filename), "wb") as cfile:
        pickle.dump(model, cfile)


def load_model(full_path_to_model):
    with open(full_path_to_model, 'rb') as cfile:
        model = pickle.load(cfile)
    return model


def predict(model, img, actual, to_cls):
    if not to_cls:
        shape_y, shape_x, *_ = img.shape
        to_predict = img.reshape(-1, shape_x, shape_y, 1)
    else:
        to_predict = img.reshape(1, -1)
    prediction_array = model.predict(to_predict)[0]

    if not to_cls:
        if actual < 3:
            prediction_array = prediction_array[:3]
            shift = 0
        else:
            prediction_array = prediction_array[3:]
            shift = 3
        predicted_idx = shift + np.argmax(prediction_array)
    else:
        if actual in range(3, 5):
            predicted_idx = ue.cast_square_to_circle(prediction_array)
        else:
            predicted_idx = ue.cast_circle_to_square(prediction_array)
    return predicted_idx


def test_model(model, path_to_dataset, to_cls=False):
    imgs_list, filename_list = prepare_dataset(path_to_dataset, full=False, flatten_array=to_cls)
    correct = 0
    errors = []
    for i in range(len(filename_list)):
        actual = np.argmax(filename_list[i])
        img = imgs_list[i]
        predicted_idx = predict(model, img, actual, to_cls)
        if actual == predicted_idx:
            correct += 1
        else:
            errors.append(f"{ue.Globals.IDX_TO_EVAL_CODE[actual]}>{ue.Globals.IDX_TO_EVAL_CODE[predicted_idx]}")
        print(
            f"[{i + 1} of {len(filename_list)}] Predicted: {ue.Globals.IDX_TO_EVAL_CODE[predicted_idx]} Actual: {ue.Globals.IDX_TO_EVAL_CODE[actual]}")
    print(f"Accuracy: {correct / len(filename_list)}")
    errors_counter = Counter(errors)
    for k, v in errors_counter.items():
        print(f"Error: {k} occurred {v} times")


def test_models(model_list, path_to_dataset, is_cls):
    cls_imgs_list, filename_list = prepare_dataset(path_to_dataset, full=False, flatten_array=True)
    cnn_imgs_list, _ = prepare_dataset(path_to_dataset, full=False, flatten_array=False)

    correct = 0
    errors = []
    target_num = 1000
    for i in range(target_num):
        prediction_list = []
        actual = np.argmax(filename_list[i])
        for idx, model in enumerate(model_list):
            img = cls_imgs_list[i] if is_cls[idx] else cnn_imgs_list[i]
            prediction_list.append(predict(model, img, actual, is_cls[idx]))

        predicted_idx = Counter(prediction_list).most_common(1)[0][0]
        if actual == predicted_idx:
            correct += 1
        else:
            verbose_prediction_list = [ue.Globals.IDX_TO_EVAL_CODE[p] for p in prediction_list]
            errors.append(
                f"{ue.Globals.IDX_TO_EVAL_CODE[actual]}>{ue.Globals.IDX_TO_EVAL_CODE[predicted_idx]} with {verbose_prediction_list}")
        print(
            f"[{i + 1} of {target_num}] Predicted: {ue.Globals.IDX_TO_EVAL_CODE[predicted_idx]} Actual: {ue.Globals.IDX_TO_EVAL_CODE[actual]}")
    print(f"Accuracy: {correct / target_num}")
    errors_counter = Counter(errors)
    for k, v in errors_counter.items():
        print(f"Error: {k} occurred {v} times")


if __name__ == "__main__":
    dataset_path = r"C:\Users\loren\PycharmProjects\OpticReader v7.0\train_and_test_data_full"

    lenet_path = r"C:\Users\loren\PycharmProjects\OpticReader v7.0\lenet.h5"
    alexnet_path = r"C:\Users\loren\PycharmProjects\OpticReader v7.0\alexnet_weights.h5"
    svm_path = r"C:\Users\loren\PycharmProjects\OpticReader v7.0\new_svm_model"
    knn_path = r"C:\Users\loren\PycharmProjects\OpticReader v7.0\new_svm_model"

    # train_cnn("lenet", dataset_path)

    # new_svm_classifier, new_knn_classifier = new_train_svm_knn(dataset_path)
    # save_model(new_svm_classifier, svm_path)
    # save_model(new_knn_classifier, knn_path)

    # print(f"{'=' * 10} Testing svm {'=' * 10}")
    # test_model(load_model(svm_path), dataset_path, True)
    # print(f"{'=' * 10} Testing knn {'=' * 10}")
    # test_model(load_model(knn_path), dataset_path, True)

    # print(f"{'='*10} Testing lenet {'='*10}")
    # test_model(keras.models.load_model(alexnet_path), dataset_path, False)
    # print(f"{'='*10} Testing alexnet {'='*10}")
    # test_model(keras.models.load_model(alexnet_path), dataset_path, False)

    alexnet_model = keras.models.load_model(alexnet_path)
    lenet = keras.models.load_model(lenet_path)
    svm = load_model(svm_path)
    knn = load_model(knn_path)
    test_models([alexnet_model, knn, svm], dataset_path, [0, 1, 1])
