import os
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def open_filepicker():
    # https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filepath = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

    if os.path.isfile(filepath):
        print(f"Chemin d'accès du fichier séléectionner = {filepath}")
    else:
        print(f"Le chemin d'accès '{filepath}' n'est pas valide. Veuillez en choisir un nouveau")
        time.sleep(1)
        open_filepicker()

open_filepicker()

styles_choices = [
    "Attirance",
    "Colère",
    "Dégout",
    "Excitation",
    "Joie",
    "Peur",
    "Surprise",
    "Tristesse"
]
styles_ids = ["1", "2", "3", "4", "5", "6", "7", "8"]

def ask_to_select_style():
    print("\nChoisi parmis un style (écris le chiffre du sytle)")
    for i_style_choice in range(len(styles_choices)):
        print(f"{i_style_choice + 1} - {styles_choices[i_style_choice]}")

    selected_style_id = input()
    if selected_style_id in styles_ids:
        pass
    else:
        print(f"{selected_style_id} n'est pas un index valide")
        time.sleep(1)
        ask_to_select_style()

ask_to_select_style()

