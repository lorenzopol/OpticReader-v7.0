import cv2
from pyzbar.pyzbar import decode
import os


def crop_to_bounding_rectangle(gray_img):
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


def decode_ean_barcode(cropped_img, is_barcode_ean13=True):
    """read a EAN13-barcode and return the candidate IDC (ID Candidato) from -> N-GG-MM-AAAA-IDC"""
    mid = decode(cropped_img)
    if mid:
        string_number = mid[0].data.decode("utf-8")
        return string_number[-4:-1] if is_barcode_ean13 else string_number
    else:
        return -1


def read_txt(full_path: str) -> list:
    with open(full_path, "r") as cf:
        content = cf.readlines()
    return content


def write_txt(full_path: str, data: str):
    with open(full_path, "w") as cf:
        cf.write(data)


def retrieve_or_display_answers():
    path: str = os.path.join(os.getcwd(), "risposte.txt")
    content = read_txt(path)
    return content


def answer_modifier(number, correct):
    path: str = os.path.join(os.getcwd(), "risposte.txt")
    if os.path.isfile(path):
        content = read_txt(path)
        number = int(number)
        content[number - 1] = "".join([str(number), " ", correct.upper(), ";\n"])
        write_txt(path, "".join(content))
    else:
        print("non è stato salvato alcun file come risposte, creane uno scegliendo l'opzione 1")


def xlsx_dumper(user, placement, correct_answers, workbook, is_60_question_sim):
    formats = [workbook.add_format({'border': 1,
                                    'align': 'center',
                                    'valign': 'vcenter'}),
               workbook.add_format({'bg_color': 'red',
                                    'border': 1,
                                    'align': 'center',
                                    'valign': 'vcenter'
                                    }),
               workbook.add_format({'border': 1,
                                    'align': 'center',
                                    'valign': 'vcenter',
                                    'bg_color': 'green'})]
    worksheet = workbook.worksheets()[0]
    v_delta = 4

    worksheet.merge_range('A1:C1', 'n° Domanda', workbook.add_format({'bold': 1,
                                                                      'border': 1,
                                                                      'align': 'center',
                                                                      'valign': 'vcenter'})
                          )
    # Create question number header
    _0_header = [*range(1, 61-(20*int(not is_60_question_sim)))]
    for col_num, data in enumerate(_0_header):
        worksheet.write(0, col_num + 3, data, workbook.add_format({'border': 1,
                                                                   'align': 'center',
                                                                   'valign': 'vcenter',
                                                                   "color": "white",
                                                                   "bg_color": "#4287F5"}))

    worksheet.merge_range('A2:C2', "RISPOSTA ESATTA", workbook.add_format({'bold': 1,
                                                                           'border': 1,
                                                                           'align': 'center',
                                                                           'valign': 'vcenter',
                                                                           }))
    # Create correct answer header
    _1_header = [*[correct_answers[i].split(";")[0].split(" ")[1]
                   for i in range(len(correct_answers[:60-(20*int(not is_60_question_sim))]))]]
    for col_num, data in enumerate(_1_header):
        worksheet.write(1, col_num + 3, data, workbook.add_format({'border': 1,
                                                                   'align': 'center',
                                                                   'valign': 'vcenter',
                                                                   "color": "white",
                                                                   "bold": 1,
                                                                   "bg_color": "#4287F5"}))
    # for percentage mod *range(60)
    _3_header = ["Posizione", "ID", "Punteggio", *[0] * 0*(60-(20*int(not is_60_question_sim)))]
    for col_num, data in enumerate(_3_header):
        worksheet.write(3, col_num, data, workbook.add_format({'bold': 1,
                                                               'border': 1,
                                                               'align': 'center',
                                                               'valign': 'vcenter'}))

    worksheet.write(f'A{placement + v_delta}', f'{placement}', workbook.add_format({'border': 1,
                                                                                    'align': 'center',
                                                                                    'valign': 'vcenter', }))
    worksheet.write(f'B{placement + v_delta}', f'{user.index}', workbook.add_format({'border': 1,
                                                                                     'align': 'center',
                                                                                     'valign': 'vcenter', }))
    worksheet.write(f'C{placement + v_delta}', f'{user.score}', workbook.add_format({'border': 1,
                                                                                     'align': 'center',
                                                                                     'valign': 'vcenter', }))

    h_delta = 3
    for number in range(h_delta, 60 + h_delta-(20*int(not is_60_question_sim))):
        print(user.score_list)
        worksheet.write(placement + v_delta - 1, number,
                        f'{user.sorted_user_answer_dict[number + 1 - h_delta]}',
                        formats[round(abs(user.score_list[number - h_delta])*2.4)])
