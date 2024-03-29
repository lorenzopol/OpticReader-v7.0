import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os


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


def pre_xlsx_dumper(workbook, correct_answers, is_50_question_sim):
    worksheet = workbook.worksheets()[0]
    worksheet.merge_range('A1:D1', 'n° Domanda', workbook.add_format({'bold': 1,
                                                                      'border': 1,
                                                                      'align': 'center',
                                                                      'valign': 'vcenter'})
                          )
    # Create question number header
    _0_header = [*range(1, 51 - (10 * int(not is_50_question_sim)))]
    for col_num, data in enumerate(_0_header):
        worksheet.write(0, col_num + 4, data, workbook.add_format({'border': 1,
                                                                   'align': 'center',
                                                                   'valign': 'vcenter',
                                                                   "color": "white",
                                                                   "bg_color": "#4287F5"}))

    worksheet.merge_range('A2:D2', "RISPOSTA ESATTA", workbook.add_format({'bold': 1,
                                                                           'border': 1,
                                                                           'align': 'center',
                                                                           'valign': 'vcenter',
                                                                           }))
    # Create correct answer header
    _1_header = [*[correct_answers[i].split(";")[0].split(" ")[1]
                   for i in range(len(correct_answers[:50 - (10 * int(not is_50_question_sim))]))]]
    for col_num, data in enumerate(_1_header):
        worksheet.write(1, col_num + 4, data, workbook.add_format({'border': 1,
                                                                   'align': 'center',
                                                                   'valign': 'vcenter',
                                                                   "color": "white",
                                                                   "bold": 1,
                                                                   "bg_color": "#4287F5"}))
    _4_header = ["Punteggio Cultura", "Punteggio Biologia", "Punteggio Anatomia", "Punteggio ChimicaFisica", "Punteggio Matematica"]
    for col_num, data in enumerate(_4_header):
        worksheet.write(3, 4 + (50 - (10 * int(not is_50_question_sim))) + col_num, data,
                        workbook.add_format({'bold': 1,
                                             'border': 1,
                                             'align': 'center',
                                             'valign': 'vcenter'}))


def xlsx_dumper(user, placement, workbook, is_60_question_sim):
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

    # for percentage mod *range(50)
    _3_header = ["Posizione", "ID", "Punteggio Equalizzato", "Punteggio Normale",
                 *[0] * (60 - (10 * int(not is_60_question_sim)))]
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
    worksheet.write_number(f'C{placement + v_delta}', round(user.score, 2), workbook.add_format({'border': 1,
                                                                                                 'align': 'center',
                                                                                                 'valign': 'vcenter', }))
    worksheet.write_number(f'D{placement + v_delta}', round(user.score - user.ceq, 2), workbook.add_format({'border': 1,
                                                                                                            'align': 'center',
                                                                                                            'valign': 'vcenter', }))
    h_delta = 4
    for number in range(h_delta, 60 + h_delta - (10 * int(not is_60_question_sim))):
        worksheet.write(placement + v_delta - 1, number,
                        f'{user.sorted_user_answer_dict[number + 1 - h_delta]}',
                        formats[round(abs(user.score_dict[number - h_delta + 1]) * 2.4)])
    h_delta = len(_3_header)

    for sub_idx, subject_score in enumerate(user.per_sub_score):
        worksheet.write_number(placement + v_delta - 1, h_delta + sub_idx, subject_score,
                               workbook.add_format({'border': 1,
                                                    'align': 'center',
                                                    'valign': 'vcenter', }))


def calculate_test_complexity_index(qst_distribution, nof_participant, max_score):
    """ceq = coefficiente di equalizzazione della prova, più è basso, più la prova e facile. Per ottenere il punteggio
    personale equalizzato (peq) si prende il punteggio non equalizzato (pne) e si aggiunge ceq. Sarà quindi
    peq = pne+ceq """
    # struct like {NumeroDomanda: [NumeroDiPersoneCheHannoRispostoCorrettamente,
    #                              NumeroDiPersoneCheNonHannoRisposto,
    #                              NumeroDiPersoneCheHannoRispostoSbagliando]}
    cdf_list = []
    for _qst_number, array in qst_distribution.items():
        nof_correct, _nof_blank, nof_wrong = array
        cdf = (nof_correct - 0.25 * nof_wrong) / nof_participant
        cdf_list.append(cdf)
    cdfp = sum(cdf_list)
    ceq = max_score - cdfp
    return ceq
