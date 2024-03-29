import multiprocessing

import cv2
import imutils

import numpy as np
from collections import Counter

from typing import *
from dataclasses import dataclass, field

from classifiers import *


@dataclass(order=True)
class User:
    index: str = field(compare=False)
    score: float
    per_sub_score: list = field(repr=False)
    score_dict: dict = field(compare=False, repr=False)
    sorted_user_answer_dict: dict = field(compare=False, repr=False)
    ceq: float = 0.0


class Utils:
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 220, 220)
    COLOR_LIST = [BLUE, GREEN, RED, MAGENTA, CYAN, YELLOW]

    NOF_QUESTIONS = 60

    BOOL_THRESHOLD = 200

    X_SAMPLE_POS_FOR_CUTS = 5
    Y_SAMPLE_POS_FOR_CUTS = 30  # it is the delta from end_question_box

    BEGIN_QUESTION_BOX_Y_SHIFT = 2
    FIRST_X_CUT_SHIFT = 3

    X_CUTS_SHIFTER = [3, 2, 0, 0, 0, 0, 0]

    QUESTION_PER_COL = 15
    SQUARE_DIM = 16

    X_COL_SHIFT = 2
    Y_ROW_SHIFT = 4

    BEGIN_QUESTION_BOX_Y_ESTIMATE = 154
    END_QUESTION_BOX_Y_ESTIMATE = 423
    ESTIMATE_WEIGHT = 0.4

    CLASSIFIER_IMG_DIM = 18

    LETTERS: Tuple[str, ...] = ("L", "", "A", "B", "C", "D", "E")
    EVAL_CODE_TO_IDX = {
        "QB": 0,
        "QS": 1,
        "QA": 2,
        "CB": 3,
        "CA": 4
    }
    IDX_TO_EVAL_CODE = {
        0: "QB",
        1: "QS",
        2: "QA",
        3: "CB",
        4: "CA"
    }


def transform_point_with_matrix(y, matrix):
    return round(matrix[1][1] * y + matrix[1][2])


def one_d_row_slice(img: np.ndarray, y: int) -> np.ndarray:
    """given a np array it returns the y-th row """
    assert y + 1 < img.shape[0], f"[ERROR]: one_d_row_slice tried accessing out of bound array: {img} with idx: {y + 1}"
    return img[y:y + 1, :]


def one_d_col_slice(img: np.ndarray, x: int) -> np.ndarray:
    """given a np array it returns the x-th col """
    assert x + 1 < img.shape[1], f"[ERROR]: one_d_col_slice tried accessing out of bound array: {img} with idx: {x + 1}"
    return img[:, x:x + 1]


def from_col_to_row(col: np.ndarray) -> np.ndarray:
    return col.reshape((1, col.shape[0]))


def find_n_black_point_on_row(one_d_sliced: np.ndarray, bool_threshold: int = Utils.BOOL_THRESHOLD) -> list[int]:
    bool_arr: np.ndarray = (one_d_sliced.flatten() < bool_threshold)

    positions = np.where(bool_arr == 1)[0]

    out = [i for i in positions]
    popped = 0
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < 10:
            del out[index - popped]
            popped += 1

    return out


def interpolate_missing_value(cols_pos_x, expected_delta, threshold_delta):
    """given a list of points, interpolate the missing value"""
    out = []
    for i in range(len(cols_pos_x) - 1):
        delta = cols_pos_x[i + 1] - cols_pos_x[i]
        if delta > threshold_delta:
            out.append(cols_pos_x[i])
            out.append(cols_pos_x[i] + expected_delta)
        else:
            out.append(cols_pos_x[i])
    out.append(cols_pos_x[-1])
    return out


def get_x_cuts(cols_x_pos: List[int] | np.array) -> List[int]:
    """given the position of each column (cols_x_pos), calculate the position of all the inner cuts so that each slice
    contains only a col of circles, numbers or squares. A more flexible approach could be use if a new version of
    the question templates gets released"""
    x_cut_positions: List[int] = []
    for i_begin_col_x in range(len(cols_x_pos) - 1):
        col_width: int = cols_x_pos[i_begin_col_x + 1] - cols_x_pos[i_begin_col_x] + Utils.X_COL_SHIFT
        square_width: int = col_width // 7
        for cut_number in range(7):
            x_cut_positions.append(
                cols_x_pos[i_begin_col_x] + square_width * cut_number + Utils.X_CUTS_SHIFTER[cut_number])
    return x_cut_positions


def get_y_cuts(begin_question_box_y: int, end_question_box_y: int) -> np.array:
    """as for get_x_cuts but row-wise"""
    return np.linspace(begin_question_box_y, end_question_box_y, Utils.QUESTION_PER_COL + 1, dtype=int)
    # old implementation
    # IDK why but +1 works
    # Utils.Y_ROW_SHIFT gives better squares for the lowest rows
    # square_height = ((end_question_box_y - begin_question_box_y) // Utils.QUESTION_PER_COL) + 1
    # return tuple(
    #     y - Utils.Y_ROW_SHIFT for y in range(begin_question_box_y, end_question_box_y + square_height, square_height))


def build_masks(size, tolerance):
    """creates a [0, 255] binary mask with max vals on the diagonal. Tolerance roughly transpose to the width of the
    diagonal"""
    zeros = np.zeros((size, size), dtype=np.uint8)
    for j in range(size):
        for i in range(size):
            if j - 1 - tolerance <= i <= j + tolerance:
                zeros[j][i] = 1
    flipped = np.flip(zeros, axis=1)
    combined = np.maximum(zeros, flipped)
    return (combined * 254) + 1


def cast_square_to_circle(predicted_category_from_cls):
    """knn_cls and svm_cls are not able to differentiate between squares and circles, cast any square prediction to the
    circle. This can safely be done since circles and squares are placed in constantly invariant position"""
    if predicted_category_from_cls == Utils.EVAL_CODE_TO_IDX["QB"]:
        return Utils.EVAL_CODE_TO_IDX["CB"]
    elif predicted_category_from_cls in (Utils.EVAL_CODE_TO_IDX["QA"], Utils.EVAL_CODE_TO_IDX["QS"]):
        return Utils.EVAL_CODE_TO_IDX["CA"]


def evaluate_square(cropped_to_bound: np.ndarray, x_index: int,
                    svm_classifier: SVC, knn_classifier: KNeighborsClassifier):
    """given the extracted image containing only a square/circle, evaluate its corresponding tag based on
    Utils.IDX_TO_EVAL_CODE """

    crop_for_old_eval = cv2.resize(cropped_to_bound, (12, 12))
    manual_pred, average, count = old_evaluate_square(crop_for_old_eval, x_index)
    crop_for_prediction = cropped_to_bound.flatten()
    if svm_classifier is not None and knn_classifier is not None:
        svm_pred = svm_classifier.predict(crop_for_prediction.reshape(1, -1))[0]
        knn_pred = knn_classifier.predict(crop_for_prediction.reshape(1, -1))[0]
        if x_index % 7 == 0:
            svm_pred = cast_square_to_circle(svm_classifier.predict([crop_for_prediction])[0])
            knn_pred = cast_square_to_circle(knn_classifier.predict([crop_for_prediction])[0])

        if svm_pred == knn_pred == manual_pred:
            return knn_pred  # if they agree, return one of them
        else:
            # if they disagree, choose the most voted option
            chosen_pred = Counter([svm_pred, knn_pred, manual_pred]).most_common(1)[0][0]
            return chosen_pred
    else:
        return 0


def old_evaluate_square(crop_for_eval: np.ndarray, x_index: int) -> tuple[int, float, int | None]:
    """latest iteration of the old generation of evaluators. Its rationale is 'given a square, compute a weighted
    average over the pixel in the crop giving max value to those black pixel along the diagonals'"""
    mask = build_masks(12, 2)
    scaled_crop = 2 * ((crop_for_eval / 255) - 0.5)
    mult = scaled_crop * mask
    average = float(np.mean(mult) / 255)

    # category = ("QB", "QS", "QA", "CB", "CA")
    #               0     1     2     3     4

    if x_index % 7:
        # square
        if average > 0.33:
            return Utils.EVAL_CODE_TO_IDX.get("QB"), average, None  # QB
        else:
            count = int(np.sum(np.where(crop_for_eval[2:10] > 125)))
            if count > 400:
                return Utils.EVAL_CODE_TO_IDX.get("QS"), average, count
            else:
                return Utils.EVAL_CODE_TO_IDX.get("QA"), average, count
    else:
        # circle
        # todo 0.1 is still a precarious value. Consider modifying it during first sim
        if average > 0.1:
            return Utils.EVAL_CODE_TO_IDX.get("CB"), average, None
        else:
            return Utils.EVAL_CODE_TO_IDX.get("CA"), average, None


def evaluate_image(bgr_scw_img: np.ndarray,
                   begin_question_box_y: int, end_question_box_y: int,
                   is_60_question_sim: int, debug: str,
                   loaded_svm_classifier: SVC, loaded_knn_classifier: KNeighborsClassifier, id_):
    """heavy lifter of the program. given a processed image, return a dictionary with key: question number and
    value: given answer"""
    draw_img = bgr_scw_img.copy()
    user_answer_dict: Dict[int, str] = {i: "" for i in range(1, Utils.NOF_QUESTIONS + 1 - 10 * int(not is_60_question_sim))}

    gray_img = cv2.cvtColor(bgr_scw_img, cv2.COLOR_BGR2GRAY)

    # apply x shift because of bad distance between black cols and circles/squares and numbers
    cols_pos_x = find_n_black_point_on_row(one_d_row_slice(gray_img, end_question_box_y + Utils.Y_SAMPLE_POS_FOR_CUTS),
                                           165)
    if len(cols_pos_x) < 5:
        cols_pos_x = interpolate_missing_value(cols_pos_x, 105, 200)

    # shift the first col to the right to compensate the fact that find_n_black_point_on_row returns the first black pixel
    cols_pos_x[0] = cols_pos_x[0] + Utils.FIRST_X_CUT_SHIFT
    x_cuts = get_x_cuts(cols_pos_x)

    # add last col because the 2xforloop need to be up to len - 1
    x_cuts.append(cols_pos_x[-1])
    if debug == "all":
        for x_cut in x_cuts:
            cv2.line(draw_img, (x_cut, 0), (x_cut, 700), Utils.CYAN, 1)

    y_cuts = get_y_cuts(begin_question_box_y, end_question_box_y)
    if debug == "all":
        cv2.line(draw_img, (0, begin_question_box_y), (500, begin_question_box_y), Utils.GREEN, 1)
        cv2.line(draw_img, (0, end_question_box_y), (500, end_question_box_y), Utils.GREEN, 1)
        cv2.line(draw_img, (0, end_question_box_y + Utils.Y_SAMPLE_POS_FOR_CUTS),
                 (500, end_question_box_y + Utils.Y_SAMPLE_POS_FOR_CUTS), Utils.RED, 1)

    if debug == "weak" or debug == "all":
        cv2.imshow("in", draw_img)
    if debug == "all":
        cv2.waitKey()

    for y_index in range(len(y_cuts) - 1):
        for x_index in range(len(x_cuts) - 1):
            # if we are on a number col, skip it
            if not (x_index - 1) % 7:
                continue

            question_number = Utils.QUESTION_PER_COL * (x_index // 7) + (y_index + 1)
            question_letter = Utils.LETTERS[x_index % 7]
            if question_number >= Utils.NOF_QUESTIONS + 1 - 10 * int(not is_60_question_sim):
                print(f"in with {question_number}")
                continue
            if user_answer_dict[question_number] == "L":
                continue

            x_top_left = int(not x_index % 7) + x_cuts[x_index]
            candidate_x_bottom_right = int(not x_index % 7) + x_cuts[x_index + 1]
            x_bottom_right = x_top_left + Utils.SQUARE_DIM if candidate_x_bottom_right - x_top_left > Utils.SQUARE_DIM \
                else candidate_x_bottom_right

            y_top_left: int = y_cuts[y_index]
            y_bottom_right: int = y_cuts[y_index + 1]

            cropped: np.array = gray_img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            cropped_to_bound = cu.crop_to_bounding_rectangle(cropped)
            area = cropped_to_bound.shape[0] * cropped_to_bound.shape[1]
            if area > 50:
                # resize only those crops that are big enough to not get blurred
                resized_crop = cv2.resize(cropped_to_bound, (Utils.CLASSIFIER_IMG_DIM, Utils.CLASSIFIER_IMG_DIM))
            else:
                resized_crop = cv2.resize(cropped, (Utils.CLASSIFIER_IMG_DIM, Utils.CLASSIFIER_IMG_DIM))
            # cv2.imshow("crop", resized_crop)
            predicted_category_index = evaluate_square(resized_crop, x_index, loaded_svm_classifier,
                                                       loaded_knn_classifier)
            # cv2.waitKey(0)
            if predicted_category_index in (Utils.EVAL_CODE_TO_IDX.get("QB"), ):
                continue

            if x_index % 7:
                # è un quadrato
                if predicted_category_index in (Utils.EVAL_CODE_TO_IDX.get("QS"), Utils.EVAL_CODE_TO_IDX.get("CA")):
                    # QS
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  Utils.GREEN, 1)
                    user_answer_dict[question_number] = question_letter

                elif predicted_category_index == Utils.EVAL_CODE_TO_IDX.get("QA"):
                    # QA
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  Utils.RED, 1)
            else:
                if predicted_category_index in (Utils.EVAL_CODE_TO_IDX.get("QS"),
                                                Utils.EVAL_CODE_TO_IDX.get("QA"), Utils.EVAL_CODE_TO_IDX.get("CA")):
                    # CA
                    cv2.rectangle(draw_img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  Utils.RED, 1)
                    user_answer_dict[question_number] = "L"
    if debug == "weak" or debug == "all":
        cv2.imshow("out", draw_img)
        cv2.waitKey()
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
        risultati_biologia,\
        risultati_anatomia,\
        risultati_chimica,\
        risultati_matematicaFisica\
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
        if noq_for_sub.get("matematicafisica")[0] <= qst_number <= noq_for_sub.get("matematicafisica")[1]:
            risultati_matematicaFisica += score

    return [risultati_Cultura, risultati_ragionamento, risultati_biologia, risultati_anatomia,
            risultati_chimica, risultati_matematicaFisica]


def generate_score_dict(user_answer_dict: Dict[int, str]) \
        -> dict[int, float]:
    """given the answer that a user has submitted, calculate if they are right or not and assign it
    corresponding score"""
    correct_answers: list = cu.retrieve_or_display_answers()

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
                score_dict[i + 1] = 1
            else:
                score_dict[i + 1] = -0.25
    return score_dict


def compute_subject_average(all_user: list[User]):
    cultura = []
    ragionamento = []
    biologia = []
    anatomia = []
    chimica = []
    matematicafisica = []
    noq_for_sub = {
        "Cultura": [0, 3],
        "ragionamento": [4, 8],
        "biologia": [9, 27],
        "anatomia": [28, 31],
        "chimica": [32, 46],
        "matefisica": [47, 60]
    }
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

    correct_answers: list = cu.retrieve_or_display_answers()
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


def warp_affine_img(BGR_SC_img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """warp affine the scaled image to straight the question box and make it fit full width"""
    GRAY_SC_img = cv2.cvtColor(BGR_SC_img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(GRAY_SC_img, 2, 3, 0.04)

    # Find corner points using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(dst, 500, .01, 10)
    # flat inner array
    corners = [corner.ravel() for corner in corners]

    x_sorted = sorted(corners, key=lambda a: a[0])
    left_points = x_sorted[:4]
    right_points = x_sorted[-1:-5:-1]

    # get bounding box points
    TL_corner = sorted(left_points, key=lambda a: a[1])[0]
    TR_corner = sorted(right_points, key=lambda a: a[1])[0]
    BL_corner = sorted(left_points, key=lambda a: a[1])[-1]

    srcTri = np.array([TL_corner, TR_corner, BL_corner])
    dstTri = np.array([[0, 0], [BGR_SC_img.shape[1], 0], [0, 650]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(srcTri, dstTri).astype(np.float32)

    BGR_SCW_img = cv2.warpAffine(BGR_SC_img, warp_mat, (BGR_SC_img.shape[1], BGR_SC_img.shape[0]))

    one_d_slice = \
        from_col_to_row(one_d_col_slice(cv2.cvtColor(BGR_SCW_img, cv2.COLOR_BGR2GRAY), Utils.X_SAMPLE_POS_FOR_CUTS))[0]
    left_black_points = find_n_black_point_on_row(one_d_slice, 165)

    begin_question_box_y = left_black_points[
                               0] + Utils.BEGIN_QUESTION_BOX_Y_SHIFT  # compensate the fact that find_n_black_point_on_row returns the first black pixel
    end_question_box_y = left_black_points[1]

    return BGR_SCW_img, begin_question_box_y, end_question_box_y


def evaluator(abs_img_path: str | os.PathLike | bytes,
              valid_ids: list[str], is_60_question_sim: int | bool, debug: str, is_barcode_ean13: int | bool,
              loaded_svm_classifier: SVC | None, loaded_knn_classifier: KNeighborsClassifier | None,
              idx: int | None = None) \
        -> User:
    print(f"idx: {idx}")
    """entry point of application. Responsible for:
        - reading the image,
        - evaluate barcode value,
        - retrieve user given answer,
        - create User object"""
    BGR_img = cv2.imread(abs_img_path)
    cropped_bar_code_id = cu.decode_ean_barcode(BGR_img[((BGR_img.shape[0]) * 3) // 4:], is_barcode_ean13)
    id_ = abs_img_path.split('_')[-1]
    if cropped_bar_code_id not in valid_ids:
        cropped_bar_code_id = input(f"lettura BARCODE fallita per {abs_img_path} >>")

    BGR_SC_img = imutils.resize(BGR_img, height=700)
    BGR_SCW_img, transformed_begin_question_box_y, transformed_end_question_box_y = warp_affine_img(BGR_SC_img)
    user_answer_dict = evaluate_image(BGR_SCW_img,
                                      transformed_begin_question_box_y, transformed_end_question_box_y,
                                      is_60_question_sim, debug,
                                      loaded_svm_classifier, loaded_knn_classifier, id_)
    # since equal scores are resolved by whoever got the most in the first section and on, calculate the score per sec
    score_dict = generate_score_dict(
        user_answer_dict)
    per_sub_score = calculate_single_sub_score(score_dict) if is_60_question_sim else []

    # create user1
    user = User(cropped_bar_code_id, sum(list(score_dict.values())), per_sub_score, score_dict, user_answer_dict)
    return user


def dispatch_multiprocess(path: str | os.PathLike | bytes, numero_di_presenti_effettivi: int,
                          valid_ids: list[str], is_60_question_sim: int | bool, debug: str,
                          is_barcode_ean13: int | bool, max_process: int = 7) \
        -> tuple[list[User], dict[int, list[int, int, int]]]:
    """create the process obj, start them, wait for them to finish and returns the evaluated relevant obj"""
    path_to_models = os.getcwd()

    loaded_svm_classifier: SVC = load_model(os.path.join(path_to_models, "svm_model"))
    loaded_knn_classifier: KNeighborsClassifier = load_model(os.path.join(path_to_models, "knn_model"))

    cargo = [[os.path.join(path, file_name), valid_ids,
              is_60_question_sim, debug, is_barcode_ean13, loaded_svm_classifier, loaded_knn_classifier, idx] for
             idx, file_name in enumerate(os.listdir(path))]
    with multiprocessing.Pool(processes=max_process) as pool:
        all_users: list[User] = pool.starmap(evaluator, cargo)

    question_distribution = get_question_distribution_from_user_list(all_users, is_60_question_sim)
    compute_subject_average(all_users)
    return all_users, question_distribution
