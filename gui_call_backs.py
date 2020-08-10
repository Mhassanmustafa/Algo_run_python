
import tkinter.messagebox as tkMessageBox
from tkinter import *
import prediction as pre


Pclass = []
gender = ['male']
siblings = []
embarked = []
Algorithum = []


def Pclass_callback(selection):
    Pclass.clear()
    Pclass.append(selection)


def Siblings_callback(selection):
    siblings.clear()
    siblings.append(selection)


def gender_callback(var):

    gender.clear()
    if var == 0:
        gender.append('male')
    else:
        gender.append('female')


def Embarked_callback(selection):
    embarked.clear()
    embarked.append(selection)


def Algorithum_callback(selection):
    Algorithum.clear()
    Algorithum.append(selection)


def submit_action():
    if(not gender or not Pclass or not embarked or not Algorithum or not siblings):
        window = Tk()
        window.wm_withdraw()

        window.geometry("1x1+200+200")
        tkMessageBox.showerror(
            title="error", message="one of the options you have not selected please select every option", parent=window)
    else:
        pre.make_userprediction(Pclass, gender, siblings,
                                embarked, Algorithum[0])
