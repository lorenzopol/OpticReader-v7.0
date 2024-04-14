import multiprocessing
import cv2
import imutils
import os

import keras
import numpy as np

from typing import *
from dataclasses import dataclass, field

from classifiers import *
import utils_evaluator as ue
import utils_main as um


@dataclass(order=True)
class User:
    index: str = field(compare=False)
    score: float
    per_sub_score: list = field(repr=False)
    score_dict: dict = field(compare=False, repr=False)
    sorted_user_answer_dict: dict = field(compare=False, repr=False)
    ceq: float = 0.0


def transform_point_with_matrix(y, matrix):
    return round(matrix[1][1] * y + matrix[1][2])


def evaluate_image(bgr_scw_img: np.ndarray,
                   begin_question_box_y: int, end_question_box_y: int,
                   is_60_question_sim: int, debug: str, svm_classifier, knn_classifier, id_):
    """heavy lifter of the program. given a processed image, return a dictionary with key: question number and
    value: given answer"""
    draw_img = bgr_scw_img.copy()
    user_answer_dict: Dict[int, str] = {i: "" for i in range(1, 61 - 10 * int(not is_60_question_sim))}

    gray_img = cv2.cvtColor(bgr_scw_img, cv2.COLOR_BGR2GRAY)

    # apply x shift because of bad distance between black cols and circles/squares and numbers
    cols_pos_x = ue.find_n_black_point_on_row(
        ue.one_d_row_slice(gray_img, end_question_box_y + ue.Globals.Y_SAMPLE_POS_FOR_CUTS),
        165)
    while len(cols_pos_x) < 5:
        cols_pos_x = ue.interpolate_missing_value(cols_pos_x, 105, 200)
    cols_pos_x[0] = cols_pos_x[
                        0] + ue.Globals.FIRST_X_CUT_SHIFT  # shift the first col to the right to compensate the fact that find_n_black_point_on_row returns the first black pixel
    x_cuts = ue.get_x_cuts(cols_pos_x)
    x_cuts.append(cols_pos_x[-1])
    # add last col because the 2xforloop need to be up to len - 1
    if debug == "all":
        for x_cut in x_cuts:
            cv2.line(draw_img, (x_cut, 0), (x_cut, 700), ue.Globals.CYAN, 1)

    y_cuts = ue.get_y_cuts(begin_question_box_y, end_question_box_y)
    if debug == "all":
        cv2.line(draw_img, (0, begin_question_box_y), (500, begin_question_box_y), ue.Globals.GREEN, 1)
        cv2.line(draw_img, (0, end_question_box_y), (500, end_question_box_y), ue.Globals.GREEN, 1)
        cv2.line(draw_img, (0, end_question_box_y + ue.Globals.Y_SAMPLE_POS_FOR_CUTS),
                 (500, end_question_box_y + ue.Globals.Y_SAMPLE_POS_FOR_CUTS), ue.Globals.RED, 1)

    if debug == "weak" or debug == "all":
        cv2.imshow("in", draw_img)
    if debug == "all":
        cv2.waitKey()

    for y_index in range(len(y_cuts) - 1):
        for x_index in range(len(x_cuts) - 1):
            # if we are on a number col, skip it
            if not (x_index - 1) % 7:
                continue

            question_number = ue.Globals.QUESTION_PER_COL * (x_index // 7) + (y_index + 1)
            question_letter = ue.Globals.LETTERS[x_index % 7]

            if question_letter == "L":
                continue

            if question_number >= 61 - 10 * int(not is_60_question_sim):
                continue

            x_top_left = int(not x_index % 7) + x_cuts[x_index]
            candidate_x_bottom_right = int(not x_index % 7) + x_cuts[x_index + 1]
            x_bottom_right = x_top_left + ue.Globals.SQUARE_DIM if candidate_x_bottom_right - x_top_left > ue.Globals.SQUARE_DIM \
                else candidate_x_bottom_right

            y_top_left: int = y_cuts[y_index]
            y_bottom_right: int = y_cuts[y_index + 1]
            cropped: np.array = gray_img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            cropped_to_bound = ue.crop_to_bounding_rectangle(cropped)
            area = cropped_to_bound.shape[0] * cropped_to_bound.shape[1]
            if area > 50:
                # resize only those crops that are big enough to not get blurred
                resized_crop = cv2.resize(cropped_to_bound,
                                          (ue.Globals.CLASSIFIER_IMG_DIM, ue.Globals.CLASSIFIER_IMG_DIM))
            else:
                resized_crop = cv2.resize(cropped, (ue.Globals.CLASSIFIER_IMG_DIM, ue.Globals.CLASSIFIER_IMG_DIM))
            predicted_category_index = ue.evaluate_square(resized_crop, x_index, svm_classifier, knn_classifier)
            if predicted_category_index in (
                    ue.Globals.EVAL_CODE_TO_IDX.get("QB"), ue.Globals.EVAL_CODE_TO_IDX.get("QA")):
                continue

            if x_index % 7:
                # è un quadrato
                if predicted_category_index in (
                        ue.Globals.EVAL_CODE_TO_IDX.get("QS"), ue.Globals.EVAL_CODE_TO_IDX.get("CA")):
                    # QS
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  ue.Globals.GREEN, 1)
                    user_answer_dict[question_number] = question_letter

                elif predicted_category_index == ue.Globals.EVAL_CODE_TO_IDX.get("QA"):
                    # QA
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  ue.Globals.RED, 1)
            else:
                if predicted_category_index in (ue.Globals.EVAL_CODE_TO_IDX.get("QS"),
                                                ue.Globals.EVAL_CODE_TO_IDX.get("QA"),
                                                ue.Globals.EVAL_CODE_TO_IDX.get("CA")):
                    # CA
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  ue.Globals.RED, 1)
                    user_answer_dict[question_number] = "L"
    if debug == "weak" or debug == "all":
        cv2.imshow("out", draw_img)
        cv2.waitKey()
    print("in")
    return user_answer_dict


def calculate_single_sub_score(score_dict: dict[int, float]):
    """Divide scores for each subject. order in return matter!"""
    noq_for_sub = {
        "Cultura": [0, 3],
        "ragionamento": [4, 8],
        "biologia": [9, 27],
        "anatomia": [28, 31],
        "chimica": [32, 46],
        "matefisica": [47, 60]
    }
    risultati_Cultura, \
        risultati_ragionamento, \
        risultati_biologia, \
        risultati_anatomia, \
        risultati_chimica, \
        risultati_matematicaFisica \
        = 0, 0, 0, 0, 0, 0

    for qst_number, score in score_dict.items():
        if noq_for_sub.get("Cultura")[0] <= qst_number <= noq_for_sub.get("Cultura")[1]:
            risultati_Cultura += score
        if noq_for_sub.get("ragionamento")[0] <= qst_number <= noq_for_sub.get("ragionamento")[1]:
            risultati_ragionamento += score
        if noq_for_sub.get("biologia")[0] <= qst_number <= noq_for_sub.get("biologia")[1]:
            risultati_biologia += score
        if noq_for_sub.get("anatomia")[0] <= qst_number <= noq_for_sub.get("anatomia")[1]:
            risultati_anatomia += score
        if noq_for_sub.get("chimica")[0] <= qst_number <= noq_for_sub.get("chimica")[1]:
            risultati_chimica += score
        if noq_for_sub.get("matefisica")[0] <= qst_number <= noq_for_sub.get("matefisica")[1]:
            risultati_matematicaFisica += score

    return [risultati_Cultura, risultati_ragionamento, risultati_biologia, risultati_anatomia,
            risultati_chimica, risultati_matematicaFisica]


def generate_score_dict(user_answer_dict: Dict[int, str]) \
        -> dict[int, float]:
    """given the answer that a user has submitted, calculate if they are right or not and assign it
    corresponding score"""
    correct_answers: list = um.retrieve_or_display_answers()

    score_dict: dict[int, float] = {i + 1: 0 for i in range(60)}

    for i in range(len(user_answer_dict)):
        pre = correct_answers[i].split(";")
        number, letter = pre[0].split(" ")
        user_letter = user_answer_dict[i + 1]
        if letter == "*" or user_letter == "L":
            # question got canceled or the user decided to lock the answer
            score_dict[i + 1] = 0
            user_answer_dict[i + 1] = ""
            continue
        if not user_letter:
            # no given answer
            score_dict[i + 1] = 0

        else:
            if user_letter in letter:
                # check if the given letter is IN the corrected letters (no "==" since, by mistake, more options
                # could be correct)
                score_dict[i + 1] = 1.5
            else:
                score_dict[i + 1] = -0.4
    return score_dict


def compute_subject_average(all_user: list[User]):
    cultura = []
    ragionamento = []
    biologia = []
    anatomia = []
    chimica = []
    matematicafisica = []
    for user in all_user:
        cultura.append(user.per_sub_score[0])
        ragionamento.append(user.per_sub_score[1])
        biologia.append(user.per_sub_score[2])
        anatomia.append(user.per_sub_score[3])
        chimica.append(user.per_sub_score[4])
        matematicafisica.append(user.per_sub_score[5])

    print(f"media cultura: {np.mean(cultura)}")
    print(f"media ragionamento: {np.mean(ragionamento)}")
    print(f"media biologia: {np.mean(biologia)}")
    print(f"media anatomia: {np.mean(anatomia)}")
    print(f"media chimica: {np.mean(chimica)}")
    print(f"media matematicafisica: {np.mean(chimica)}")


def get_question_distribution_from_user_list(all_users: list[User], is_60_question_sim) \
        -> dict[int, list[int, int, int]]:
    question_distribution = {i: [0, 0, 0] for i in range(60 - 10 * int(not is_60_question_sim))}

    correct_answers: list = um.retrieve_or_display_answers()
    for user in all_users:
        for i in range(len(user.sorted_user_answer_dict)):
            pre = correct_answers[i].split(";")
            number, letter = pre[0].split(" ")
            user_letter = user.sorted_user_answer_dict[i + 1]
            if user_letter == "" or user_letter == "L":
                # add that this person did not answer this question
                question_distribution[i][1] += 1
            else:
                if user_letter in letter:
                    # add that this person got the i-th question correct
                    question_distribution[i][0] += 1
                else:
                    # add that this person got the i-th question wrong
                    question_distribution[i][2] += 1
    return question_distribution


def avg_if_close_else_min(x, y, tolerance):
    return (x + y) / 2 if abs(x - y) < tolerance else min(x, y)


def avg_if_close_else_max(x, y, tolerance):
    return (x + y) / 2 if abs(x - y) < tolerance else max(x, y)


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
    return alexnet


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


def evaluator(abs_img_path: str | os.PathLike | bytes,
              valid_ids: list[str], is_60_question_sim: int | bool, debug: str, is_barcode_ean13: int | bool,
              svm_classifier, knn_classifier,
              idx: int | None = None) \
        -> User:
    """entry point of application. Responsible for:
        - reading the image,
        - evaluate barcode value,
        - retrieve user given answer,
        - create User object"""
    BGR_img = cv2.imread(abs_img_path)
    crop_for_barcode = BGR_img[((BGR_img.shape[0]) * 3) // 4:]
    cropped_bar_code_id = ue.decode_ean_barcode(crop_for_barcode, is_barcode_ean13)
    id_ = abs_img_path.split('_')[-1]
    print(f"processing {id_}")
    if cropped_bar_code_id not in valid_ids:
        # todo
        # cropped_bar_code_id = input(f"lettura BARCODE fallita per {abs_img_path} >>")
        cropped_bar_code_id = id_

    BGR_SC_img = imutils.resize(BGR_img, height=700)
    BGR_SCW_img, transformed_begin_question_box_y, transformed_end_question_box_y = ue.warp_affine_img(BGR_SC_img)
    user_answer_dict = evaluate_image(BGR_SCW_img,
                                      transformed_begin_question_box_y, transformed_end_question_box_y,
                                      is_60_question_sim, debug, svm_classifier, knn_classifier, id_)
    # since equal scores are resolved by whoever got the most in the first section and on, calculate the score per sec
    score_dict = generate_score_dict(
        user_answer_dict)
    per_sub_score = calculate_single_sub_score(score_dict) if is_60_question_sim else []

    # create user1
    user = User(cropped_bar_code_id, sum(list(score_dict.values())), per_sub_score, score_dict, user_answer_dict)
    return user


def calculate_start_end_idxs(numero_di_presenti_effettivi: int, max_process: int) -> list[int]:
    """given the max number of process, calculates how many files each worker has to evaluate.
    If numero_di_presenti_effettivi is not divisible by max_process, remaning work is assigned to last worker"""
    start_end_idxs = list(range(0, numero_di_presenti_effettivi, numero_di_presenti_effettivi // max_process))
    if len(start_end_idxs) == max_process:
        start_end_idxs.append(numero_di_presenti_effettivi)
    else:
        start_end_idxs[-1] = numero_di_presenti_effettivi
    if len(start_end_idxs) == 1:
        start_end_idxs.insert(0, 0)
    return start_end_idxs




def dispatch_multiprocess(path: str | os.PathLike | bytes, numero_di_presenti_effettivi: int,
                          valid_ids: list[str], is_60_question_sim: int | bool, debug: str,
                          is_barcode_ean13: int | bool, max_process: int = 7) \
        -> tuple[list[User], dict[int, list[int, int, int]]]:
    """create the process obj, start them, wait for them to finish and returns the evaluated relevant obj"""
    svm_classifier = load_model(ue.Globals.SVM_PATH)
    knn_classifier = load_model(ue.Globals.KNN_PATH)

    cargo = [[os.path.join(path, file_name), valid_ids,
              is_60_question_sim, debug, is_barcode_ean13,
              svm_classifier, knn_classifier,
              idx] for
             idx, file_name in enumerate(os.listdir(path))]
    start_end_idxs = calculate_start_end_idxs(numero_di_presenti_effettivi, max_process)
    with multiprocessing.Pool(processes=max_process) as pool:
        all_users: list[User] = pool.starmap(evaluator, cargo)

    question_distribution = get_question_distribution_from_user_list(all_users, is_60_question_sim)
    compute_subject_average(all_users)
    return all_users, question_distribution
