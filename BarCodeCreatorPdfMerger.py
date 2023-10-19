import barcode
from barcode.writer import ImageWriter
import cv2
import os
from PIL import Image
from pyzbar.pyzbar import decode


def build_sheets(path):
    answer_sheet = cv2.imread(path)
    for i in range(750):
        print(f"{i}-th iteration started")
        x_beg = answer_sheet.shape[1] // 2
        y_beg = round(answer_sheet.shape[0] * .80)
        temp = answer_sheet.copy()

        id_string = f"912102023{i:03}"
        ean = barcode.get('ean13', id_string, writer=ImageWriter())
        filename = ean.save('ean13')
        barcode_img = cv2.imread("ean13.png")
        scaling_factor = 0.6
        barcode_img = cv2.resize(barcode_img,
                                 (round(barcode_img.shape[1] * scaling_factor),
                                  round(barcode_img.shape[0] * scaling_factor)),
                                 interpolation=cv2.INTER_LINEAR)

        x_beg -= barcode_img.shape[1] // 2

        cv2.putText(temp, str(f"Prova numero: {id_string[-3:len(id_string)]}"), (x_beg, y_beg - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        temp[y_beg:y_beg + barcode_img.shape[0],
             x_beg:x_beg + barcode_img.shape[1]] = barcode_img

        cv2.imwrite(f"blank_barcoded_tests/{id_string}.png", temp)
        print(f"{i}-th iteration ended\n")


def merge_to_one_pdf(path_to_images):
    all_images = []
    for _, _, filenames in os.walk(path_to_images):
        for file in filenames:
            print(f"{file} begin")
            all_images.append(Image.open(f"{path_to_images}/{file}"))
            print(f"{file} end")

    first = all_images.pop(0)
    first.save(r"full_pd.pdf", save_all=True, append_images=all_images)


def decode_EAN_barcode(cropped_img):
    """read a EAN13-barcode and return the candidate IDC (ID Candidato) from -> N-GG-MM-AAAA-IDC"""
    string_number = decode(cropped_img)[0].data.decode("utf-8")
    return string_number[-4:-1]


def decode_whole_dir(path_to_images):
    for _, _, filenames in os.walk(path_to_images):
        for file in filenames:
            print(f"{file = }. Decoded: {decode_EAN_barcode(cv2.imread(os.path.join(path_to_images, file))) }")


if __name__ == "__main__":
    print("DANGER: RUN FROM TERMINAL NOT FROM PYCHARM")
    # build_sheets(r"C:\Users\loren\PycharmProjects\OpticReader v7.0\reduced_res_50QUES.png")
    # merge_to_one_pdf(os.path.join(os.getcwd(), "blank_barcoded_tests"))
    decode_whole_dir(os.path.join(os.getcwd(), "blank_barcoded_tests"))
