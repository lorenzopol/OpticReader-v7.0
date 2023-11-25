import dearpygui.dearpygui as dpg
from evaluator import dispatch_multiprocess, evaluator
import custom_utils as cu
import os
import xlsxwriter
from classifiers import load_model


class DpgExt:
    """dummy class dpg extension"""

    @staticmethod
    def draw_answ_table():
        dpg.delete_item("TR")
        with dpg.table(parent="MRW", label="Tabella Risposte", header_row=False, tag="TR"):
            dpg.add_table_column()
            for _entry in cu.retrieve_or_display_answers():
                _qst_number, _qst_letter = _entry.split(";")[0].split(" ")
                with dpg.table_row():
                    dpg.add_text(f"{_qst_number} {_qst_letter}")

    @staticmethod
    def mod_answers_file(sender, app_data, user_data):
        cu.answer_modifier(user_data[0], user_data[1])
        DpgExt.draw_answ_table()
        dpg.set_value("num", "")
        dpg.set_value("answ", "")

    @staticmethod
    def show_confirm(sender, app_data, user_data):
        dpg.configure_item("DC", show=True)

    @staticmethod
    def confirm_delete(sender, app_data, user_data):
        for i in range(1, 51):
            cu.answer_modifier(i, "")
        DpgExt.draw_answ_table()
        DpgExt.exit_confirm("", "", "")

    @staticmethod
    def exit_confirm(sender, app_data, user_data):
        dpg.configure_item("DC", show=False)

    @staticmethod
    def confirm_launch(sender, app_data, user_data):
        path = dpg.get_value("pathToScan")
        dpg.show_item("progress")

        is_50_question_sim = dpg.get_value("50QuestionForm")
        is_barcode_ean13 = dpg.get_value("EAN13")
        is_multithread = dpg.get_value("MultiThread")
        is_evaluate = dpg.get_value("Evaluate")
        debug = dpg.get_value("debug")
        max_number_of_tests = dpg.get_value("maxPeople")
        valid_ids = [f"{i:03}" for i in range(max_number_of_tests)] if is_barcode_ean13 else \
            [f"{i:04}" for i in range(1000)]

        workbook = xlsxwriter.Workbook("graduatorie/excel_graduatorie.xlsx")
        workbook.add_worksheet()

        placement = 0
        numero_di_presenti_effettivi = len(os.listdir(path))
        all_users = None
        question_distribution = None
        if is_multithread:
            all_users, question_distribution = dispatch_multiprocess(
                path,
                numero_di_presenti_effettivi,
                valid_ids,
                is_50_question_sim, debug,
                is_barcode_ean13)
        else:
            print("multithread not selected. Default behaviour will not evaluate scores")
            path_to_models = os.getcwd()
            if is_evaluate:
                loaded_svm_classifier = load_model(os.path.join(path_to_models, "svm_model"))
                loaded_knn_classifier = load_model(os.path.join(path_to_models, "knn_model"))
            else:
                loaded_svm_classifier = None
                loaded_knn_classifier = None

            for filename in os.listdir(path):
                _ = evaluator(
                    os.path.join(path, filename), valid_ids,
                    is_50_question_sim,
                    debug, is_barcode_ean13,
                    loaded_svm_classifier, loaded_knn_classifier)
            question_distribution = None
            all_users = None


        if question_distribution is not None and all_users is not None:
            ceq = cu.calculate_test_complexity_index(question_distribution, numero_di_presenti_effettivi, max_score=50)
            for user in all_users:
                user.score = round((user.score + ceq), 2)
                user.ceq = ceq
            sorted_by_score_user_list = sorted(all_users, key=lambda x: (x.score, x.per_sub_score), reverse=True)
            cu.pre_xlsx_dumper(workbook, cu.retrieve_or_display_answers(), is_50_question_sim)
            for placement, user in enumerate(sorted_by_score_user_list):
                cu.xlsx_dumper(user, placement + 1, workbook, is_50_question_sim)

            worksheet = workbook.worksheets()[0]
            for col, people_who_got_correct_not_given_and_wrong_answ in question_distribution.items():
                nof_correct, nof_not_given, nof_wrong = people_who_got_correct_not_given_and_wrong_answ
                worksheet.write(3, 4 + col, f"{round(nof_correct / (placement + 1) * 100)}%",
                                workbook.add_format({'bold': 1,
                                                     'border': 1,
                                                     'align': 'center',
                                                     'valign': 'vcenter'})
                                )
            stats_keys_format = workbook.add_format({'border': 1,
                                                     "bold": 1,
                                                     'align': 'center',
                                                     'valign': 'vcenter', })

            stats_value_format = workbook.add_format({'border': 1,
                                                      'align': 'center',
                                                      'valign': 'vcenter', })
            # stats_dump
            # =INDICE($A$5:$A$14; CONFRONTA(MIN(ASS($C$5:$C$14-C20)); ASS($C$5:$C$14-C20); 0))
            worksheet.write(placement + 4 + 1, 3, f"Numero di risposte corrette",
                            workbook.add_format({'bold': 1,
                                                 'border': 1,
                                                 'align': 'center',
                                                 'valign': 'vcenter'})
                            )
            worksheet.write(placement + 5 + 1, 3, f"Numero di risposte non date",
                            workbook.add_format({'bold': 1,
                                                 'border': 1,
                                                 'align': 'center',
                                                 'valign': 'vcenter'})
                            )
            worksheet.write(placement + 6 + 1, 3, f"Numero di risposte sbagliate",
                            workbook.add_format({'bold': 1,
                                                 'border': 1,
                                                 'align': 'center',
                                                 'valign': 'vcenter'})
                            )
            for col, people_who_got_correct_not_given_and_wrong_answ in question_distribution.items():
                nof_correct, nof_not_given, nof_wrong = people_who_got_correct_not_given_and_wrong_answ
                worksheet.write(placement + 4 + 1, 4 + col, f"{round(nof_correct / (placement + 1), 2)}",
                                workbook.add_format({'bold': 1,
                                                     'border': 1,
                                                     'align': 'center',
                                                     'valign': 'vcenter'})
                                )
                worksheet.write(placement + 5 + 1, 4 + col, f"{round(nof_not_given / (placement + 1), 2)}",
                                workbook.add_format({'bold': 1,
                                                     'border': 1,
                                                     'align': 'center',
                                                     'valign': 'vcenter'})
                                )
                worksheet.write(placement + 6 + 1, 4 + col, f"{round(nof_wrong / (placement + 1), 2)}",
                                workbook.add_format({'bold': 1,
                                                     'border': 1,
                                                     'align': 'center',
                                                     'valign': 'vcenter'})
                                )

            formula_range = f"$C$5:$C${placement + 7 + 1}"
            worksheet.write(placement + 8 + 1, 1, "Media", stats_keys_format)
            worksheet.write_formula(placement + 8 + 1, 2, f"=_xlfn.AVERAGE({formula_range})", stats_value_format)
            worksheet.write_formula(placement + 8 + 1, 3,
                                    f"=_xlfn.INDEX($A$5:$A${placement+4+1}, _xlfn.MATCH(_xlfn.MIN(_xlfn.ABS({formula_range}-C{placement + 5 + 2})), _xlfn.ABS({formula_range}-C{placement + 5 + 2}), 0))",
                                    stats_value_format)

            worksheet.write(placement + 9 + 1, 1, "Mediana", stats_keys_format)
            worksheet.write_formula(placement + 9 + 1, 2, f"=_xlfn.MEDIAN({formula_range})", stats_value_format)

            # worksheet.write(placement + 7 + 1, 1, "Moda", stats_keys_format)
            # worksheet.write_formula(placement + 7 + 1, 2, f"=_xlfn.MODE({formula_range})", stats_value_format)

            worksheet.write(placement + 11 + 1, 1, "Massimo", stats_keys_format)
            worksheet.write_formula(placement + 11 + 1, 2, f"=_xlfn.MAX({formula_range})", stats_value_format)

            worksheet.write(placement + 12 + 1, 1, "Minimo", stats_keys_format)
            worksheet.write_formula(placement + 12 + 1, 2, f"=_xlfn.MIN({formula_range})", stats_value_format)

            worksheet.write(placement + 8 + 1, 5, "Q1", stats_keys_format)
            worksheet.write_formula(placement + 8 + 1, 6, f"=_xlfn.QUARTILE({formula_range}, 1)", stats_value_format)

            worksheet.write(placement + 9 + 1, 5, "Q2", stats_keys_format)
            worksheet.write_formula(placement + 9 + 1, 6, f"=_xlfn.QUARTILE({formula_range}, 2)", stats_value_format)

            worksheet.write(placement + 10 + 1, 5, "Q3", stats_keys_format)
            worksheet.write_formula(placement + 10 + 1, 6, f"=_xlfn.QUARTILE({formula_range}, 3)", stats_value_format)
            workbook.close()
        else:
            print("CRITICAL ERROR")

        DpgExt.exit_launch("", "", "")
        dpg.show_item("EX")

    @staticmethod
    def exit_launch(sender, app_data, user_data):
        dpg.hide_item("LA")
        dpg.hide_item("progress")
        dpg.set_value("pathToScan", "")

    @staticmethod
    def confirm_path(sender, app_data, user_data):
        dpg.configure_item("LA", show=True)

    @staticmethod
    def run_excel(sender, app_data, user_data):
        os.system("start excel graduatorie/excel_graduatorie.xlsx")
        dpg.hide_item("EX")

    @staticmethod
    def build_answers(sender, app_data, user_data):
        new_answer = (dpg.get_value(sender))
        question_number = int(user_data.split(":")[0])
        cu.answer_modifier(question_number, new_answer)
        DpgExt.draw_answ_table()
        if question_number < 50:
            dpg.set_value("BuilderNumber", f"{question_number + 1}:")
        else:
            dpg.set_value("BuilderNumber", f"{1}:")
        dpg.set_value("AnswBuilder", "")
        dpg.focus_item("AnswBuilder")


def main():
    dpg.create_context()

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (255, 16, 16), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (60, 60, 60), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 100, 100), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 100, 100), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg, (150, 75, 75), category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)

    with dpg.window(label="Mostra Risposte", width=150, height=650,
                    no_resize=True, tag="MRW", no_close=True, no_move=True):
        with dpg.table(parent="MRW", label="Tabella Risposte", header_row=False, tag="TR"):
            dpg.add_table_column()
            for entry in cu.retrieve_or_display_answers():
                qst_number, qst_letter = entry.split(";")[0].split(" ")
                with dpg.table_row():
                    dpg.add_text(f"{qst_number} {qst_letter}")

    with dpg.window(label="Modifica Risposte", width=500, pos=(150, 0),
                    no_resize=True, no_close=True, no_move=True):
        with dpg.group(label="##NumGruop", horizontal=True):
            dpg.add_text("Numero domanda:  ")
            dpg.add_input_text(tag="num")

        with dpg.group(label="##AnswGruop", horizontal=True):
            dpg.add_text("Risposta esatta: ")
            dpg.add_input_text(tag="answ")

        dpg.add_spacer(height=5)
        with dpg.group(label="##ModGruop", horizontal=True):
            dpg.add_spacer(width=90)
            dpg.add_button(pos=(225, 85), label="Modifica", callback=DpgExt.mod_answers_file, tag="mod")

        dpg.add_spacer(height=5)

    with dpg.window(label="Crea Nuove Risposte", width=500, height=235, pos=(150, 115),
                    no_resize=True, no_close=True, no_move=True):
        dpg.add_spacer(height=5)
        with dpg.group(label="##DeleteMain", horizontal=True):
            dpg.add_spacer(width=125)
            dpg.add_button(label="Cancellare tutte le risposte?", callback=DpgExt.show_confirm)

        dpg.add_spacer(height=5)
        with dpg.group(label="##DeleteConfirm", horizontal=True, show=False, tag="DC"):
            dpg.add_spacer(width=150)
            dpg.add_text("Sicuro: ", tag="sure")
            dpg.add_button(label="Sì", tag="y", callback=DpgExt.confirm_delete)
            dpg.add_button(label="No", tag="n", callback=DpgExt.exit_confirm)
            dpg.add_spacer(height=5)

        dpg.add_separator()
        dpg.add_spacer(height=5)
        with dpg.group(label="dummy"):
            with dpg.group(label="##text"):
                dpg.add_text("Inserire una risposta alla volta e premere ENTER")
            with dpg.group(label="##BuilderGruop", horizontal=True):
                dpg.add_spacer(width=25)
                dpg.add_text("1: ", tag="BuilderNumber")
                dpg.add_input_text(label="", tag="AnswBuilder", on_enter=True,
                                   callback=DpgExt.build_answers)

    with dpg.window(label="Analisi Scansioni", width=500, height=300, pos=(150, 350),
                    no_move=True, no_close=True, no_resize=True):
        dpg.add_spacer(height=5)
        with dpg.group(label="##percorsi", horizontal=True):
            dpg.add_text("Percorso alle sacnsioni: ")
            dpg.add_input_text(label="", tag="pathToScan", callback=DpgExt.confirm_launch, on_enter=True, width=250,
                               default_value=r"E:\novembre")
            dpg.add_button(label="OK", callback=DpgExt.confirm_path)

        dpg.add_spacer(height=5)
        with dpg.group(label="##Launch", show=False, tag="LA"):
            with dpg.group(label="##debugLaunch", horizontal=True, tag="debugLaunch"):
                dpg.add_text("Massimo numero di partecipanti (approx per eccesso in dubbio): ")
                dpg.add_input_int(label="", tag="maxPeople", width=250, default_value=800)
                dpg.add_spacer(height=20)
            with dpg.group(horizontal=True, tag="CQN"):
                dpg.add_text("Simulazione da 50 quesiti?")
                dpg.add_checkbox(label="", tag="50QuestionForm", default_value=True)
                dpg.add_spacer(width=15)
                dpg.add_text("EAN13?")
                dpg.add_checkbox(label="", tag="EAN13", default_value=True)
                dpg.add_spacer(width=15)
                dpg.add_text("MultiThread?")
                dpg.add_checkbox(label="", tag="MultiThread", default_value=True)
            with dpg.group(horizontal=True, tag="eval"):
                dpg.add_text("Evaluate?")
                dpg.add_checkbox(label="", tag="Evaluate", default_value=True)
            with dpg.group(horizontal=True, tag="DEB"):
                dpg.add_text("Debug?")
                dpg.add_combo(items=["No", "weak", "all"], tag="debug")

            with dpg.group(label="##ConfiL", horizontal=True, tag="confL"):
                dpg.add_spacer(width=100)
                dpg.add_text("Avviare analisi delle scansioni: ", tag="sureL")
                dpg.add_button(label="Sì", tag="yL", callback=DpgExt.confirm_launch)
                dpg.add_button(label="No", tag="nL", callback=DpgExt.exit_launch)
                dpg.add_spacer(height=5)

        with dpg.group(label="##Progress", show=False, horizontal=True, tag="progress"):
            dpg.add_spacer(width=70)
            dpg.add_text("", tag="progressCount")
            dpg.add_spacer(height=5)

        dpg.add_spacer(height=5)
        dpg.add_separator()
        dpg.add_spacer(height=5)

        with dpg.group(label="##Excel", horizontal=True, show=False, tag="EX"):
            dpg.add_spacer(width=200)
            dpg.add_button(label="Excel", callback=DpgExt.run_excel)

    dpg.create_viewport(title='Lettore Ottico v7.0', width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.set_item_user_data("mod", [dpg.get_value("num"), dpg.get_value("answ")])
        dpg.set_item_user_data("AnswBuilder", dpg.get_value("BuilderNumber"))
        dpg.set_item_user_data("yL", dpg.get_value("pathToScan"))

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def run_with_profiling():
    prof_path = r"E:\novembre"
    prof_nof_pres_eff = len(os.listdir(prof_path))
    prof_valid_ids = [f"{i:03}" for i in range(1000)]
    prof_is_50_question_sim = True
    prof_debug = "No"
    prof_is_barcode_ean13 = True

    all_users, question_distribution = dispatch_multiprocess(
        prof_path, prof_nof_pres_eff, prof_valid_ids,
        prof_is_50_question_sim, prof_debug, prof_is_barcode_ean13)


if __name__ == '__main__':
    def_run = int(input("Choose running option\n 1. Default\n 2. Profiling \n >> "))
    if def_run == 1:
        main()
    else:
        run_with_profiling()
