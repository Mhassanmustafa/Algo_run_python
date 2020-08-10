from tkinter import *
import gui_call_backs as cb
import prediction as pd


def radio():
    cb.gender_callback(var.get())


def submit_action():
    cb.submit_action()
    if not pd.prediction_list:
        print("sorry")
    else:
        printResult(pd.prediction_list)


def printResult(result):
    if result[0] == 0:

        label = Label(root, text="Prediction result :Person Survived",
                      width=30, font=("bold", 10))
        label.place(x=140, y=450)
    else:

        label = Label(
            root, text="Prediction result :Person not Survived", width=30, font=("bold", 10))
        label.place(x=140, y=450)


root = Tk()
root.geometry('500x500')
root.title("Mechine Learning models")

label_0 = Label(root, text="Prediction Form", width=20, font=("bold", 20))
label_0.place(x=90, y=53)


label_1 = Label(root, text="Pclass", width=20, font=("bold", 10))
label_1.place(x=70, y=130)

list1 = ['1', '2', '3']
c = StringVar()
droplist = OptionMenu(root, c, *list1, command=cb.Pclass_callback)
droplist.config(width=15)
c.set('select Pclass')
droplist.place(x=240, y=130)


label_2 = Label(root, text="Sibling", width=20, font=("bold", 10))
label_2.place(x=70, y=180)

list1 = ['0', '1', '2', '3']
c = StringVar()
droplist = OptionMenu(root, c, *list1, command=cb.Siblings_callback)
droplist.config(width=15)
c.set('select Sibling')
droplist.place(x=240, y=180)


label_3 = Label(root, text="Gender", width=20, font=("bold", 10))
label_3.place(x=70, y=230)
var = IntVar()
Radiobutton(root, text="Male", padx=5, variable=var,
            value=0, command=radio).place(x=235, y=230)
Radiobutton(root, text="Female", padx=20,
            variable=var, value=1, command=radio).place(x=290, y=230)

label_4 = Label(root, text="Embarked", width=20, font=("bold", 10))
label_4.place(x=70, y=280)

list1 = ['C', 'S', 'Q']
c = StringVar()
droplist = OptionMenu(root, c, *list1, command=cb.Embarked_callback)
droplist.config(width=15)
c.set('select Embarked')
droplist.place(x=240, y=280)

label_4 = Label(root, text="Algorithim", width=20, font=("bold", 10))
label_4.place(x=70, y=330)

list1 = ['Random Forest', 'Multi layer perceptron',
         'K neraest Neighbour', 'Nave base', 'Linear SVC']
c = StringVar()
droplist = OptionMenu(root, c, *list1, command=cb.Algorithum_callback)
droplist.config(width=30)
c.set('select One Algorithm')
droplist.place(x=237, y=330)

Button(root, text='Submit', width=20, bg='brown',
       fg='white', command=submit_action).place(x=180, y=380)

root.mainloop()
