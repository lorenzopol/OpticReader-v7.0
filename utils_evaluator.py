import numpy as np
import cv2
import os

import random

from pyzbar.pyzbar import decode
from collections import Counter

import pickle
import keras


class Globals:
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 220, 220)
    COLOR_LIST = [BLUE, GREEN, RED, MAGENTA, CYAN, YELLOW]

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

    LETTERS: tuple[str, ...] = ("L", "", "A", "B", "C", "D", "E")
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
    KNN_PATH = os.path.join(os.getcwd(), "lol_knn_model")
    SVM_PATH = os.path.join(os.getcwd(), "lol_svm_model")
    CNN_PATH = os.path.join(os.getcwd(), "lenet.h5")


def load_model(full_path_to_model):
    with open(full_path_to_model, 'rb') as cfile:
        model = pickle.load(cfile)
    return model


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
    if predicted_category_from_cls == Globals.EVAL_CODE_TO_IDX["QB"]:
        return Globals.EVAL_CODE_TO_IDX["CB"]
    elif predicted_category_from_cls in (Globals.EVAL_CODE_TO_IDX["QA"], Globals.EVAL_CODE_TO_IDX["QS"]):
        return Globals.EVAL_CODE_TO_IDX["CA"]
    return predicted_category_from_cls


def cast_circle_to_square(predicted_category_from_cls):
    """knn_cls and svm_cls are not able to differentiate between squares and circles, cast any circle prediction to the
    square. This can safely be done since circles and squares are placed in constantly invariant position"""
    if predicted_category_from_cls == Globals.EVAL_CODE_TO_IDX["CB"]:
        return Globals.EVAL_CODE_TO_IDX["QB"]
    elif predicted_category_from_cls == Globals.EVAL_CODE_TO_IDX["CA"]:
        return random.choice([Globals.EVAL_CODE_TO_IDX["QA"], Globals.EVAL_CODE_TO_IDX["QS"]])
    else:
        return predicted_category_from_cls

def crop_to_bounding_rectangle(gray_img: np.ndarray):
    thresh = cv2.inRange(gray_img, 0, 150)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    chosen = None
    for cnt in contours:
        current_area = cv2.contourArea(cnt)
        if current_area > maxArea:
            maxArea = current_area
            chosen = cnt
    if chosen is not None:
        x, y, w, h = cv2.boundingRect(chosen)
        return gray_img[y:y + h, x:x + w]
    else:
        return gray_img


def decode_ean_barcode(cropped_img: np.ndarray, is_barcode_ean13=True):
    """read a EAN13-barcode and return the candidate IDC (ID Candidato) from -> N-GG-MM-AAAA-IDC"""
    mid = decode(cropped_img)
    if mid:
        string_number = mid[0].data.decode("utf-8")
        return string_number[-4:-1] if is_barcode_ean13 else string_number
    else:
        return -1


def find_n_black_point_on_row(one_d_sliced: np.ndarray, bool_threshold: int = Globals.BOOL_THRESHOLD) -> list[int]:
    """refactor may be needed. Check TI_canny_find_n_black_point_on_row and TI_canny_find_n_black_point_on_col"""
    bool_arr: np.ndarray = (one_d_sliced.flatten() < bool_threshold)

    positions = np.where(bool_arr == 1)[0]

    out = [i for i in positions]
    popped = 0
    for index in range(1, len(positions)):
        if positions[index] - positions[index - 1] < 10:
            del out[index - popped]
            popped += 1

    return out


def get_y_cuts(begin_question_box_y: int, end_question_box_y: int) -> np.array:
    """as for get_x_cuts but row-wise"""
    return np.linspace(begin_question_box_y, end_question_box_y, Globals.QUESTION_PER_COL + 1, dtype=int)
    # old implementation
    # IDK why but +1 works
    # ue.Globals.Y_ROW_SHIFT gives better squares for the lowest rows
    # square_height = ((end_question_box_y - begin_question_box_y) // ue.Globals.QUESTION_PER_COL) + 1
    # return tuple(
    #     y - ue.Globals.Y_ROW_SHIFT for y in range(begin_question_box_y, end_question_box_y + square_height, square_height))


def get_x_cuts(cols_x_pos: list[int] | np.ndarray) -> list[int]:
    """given the position of each column (cols_x_pos), calculate the position of all the inner cuts so that each slice
    contains only a col of circles, numbers or squares. A more flexible approach could be use if a new version of
    the question templates gets released"""
    x_cut_positions: list[int] = []
    for i_begin_col_x in range(len(cols_x_pos) - 1):
        col_width: int = cols_x_pos[i_begin_col_x + 1] - cols_x_pos[i_begin_col_x] + Globals.X_COL_SHIFT
        square_width: int = col_width // 7
        for cut_number in range(7):
            x_cut_positions.append(
                cols_x_pos[i_begin_col_x] + square_width * cut_number + Globals.X_CUTS_SHIFTER[cut_number])
    return x_cut_positions


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
        from_col_to_row(one_d_col_slice(cv2.cvtColor(BGR_SCW_img, cv2.COLOR_BGR2GRAY), Globals.X_SAMPLE_POS_FOR_CUTS))[
            0]
    left_black_points = find_n_black_point_on_row(one_d_slice, 165)

    begin_question_box_y = left_black_points[
                               0] + Globals.BEGIN_QUESTION_BOX_Y_SHIFT  # compensate the fact that find_n_black_point_on_row returns the first black pixel
    end_question_box_y = left_black_points[1]

    return BGR_SCW_img, begin_question_box_y, end_question_box_y


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
            return Globals.EVAL_CODE_TO_IDX.get("QB"), average, None  # QB
        else:
            count = int(np.sum(np.where(crop_for_eval[2:10] > 125)))
            if count > 400:
                return Globals.EVAL_CODE_TO_IDX.get("QS"), average, count
            else:
                return Globals.EVAL_CODE_TO_IDX.get("QA"), average, count
    else:
        # circle
        # todo 0.1 is still a precarious value. Consider modifying it during first sim
        if average > 0.1:
            return Globals.EVAL_CODE_TO_IDX.get("CB"), average, None
        else:
            return Globals.EVAL_CODE_TO_IDX.get("CA"), average, None


def evaluate_square(cropped_to_bound: np.ndarray, x_index: int, svm_classifier, knn_classifier) -> int:

    """given the extracted image containing only a square/circle, evaluate its corresponding tag based on
    ue.Globals.IDX_TO_EVAL_CODE """

    crop_for_prediction = cropped_to_bound.flatten().reshape(1, -1)
    if svm_classifier is not None and knn_classifier is not None:
        if x_index % 7 == 0:
            svm_pred = cast_square_to_circle(svm_classifier.predict(crop_for_prediction)[0])
            knn_pred = cast_square_to_circle(knn_classifier.predict(crop_for_prediction)[0])
        else:
            svm_pred = cast_circle_to_square(svm_classifier.predict(crop_for_prediction)[0])
            knn_pred = cast_circle_to_square(knn_classifier.predict(crop_for_prediction)[0])
            cnn_pred = cnn_pred[:3]
            cnn_pred_shift = 0
        cnn_pred = np.argmax(cnn_pred) + cnn_pred_shift
        if svm_pred == knn_pred == cnn_pred:
            return knn_pred  # if they agree, return one of them
        else:
            # if they disagree, choose the most voted option
            chosen_pred = Counter([svm_pred, knn_pred, cnn_pred]).most_common(1)[0][0]
            return chosen_pred
    else:
        return 0