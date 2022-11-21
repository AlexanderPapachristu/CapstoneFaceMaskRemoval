from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import initial.py
root=Tk()
root.title('Mask removal software')

def open()
global img
root.filename =filedialog.askopenfilename(initialdir="Pictures", title="Select a File", filetypes=(("png files","*.png"),("jpeg files","*.jpeg")))
mylabel=Label(root,text=root.filename).pack()
img=ImageTk.PhotoImage(Image.open)(root.filename)
x=initial(img)
imglabel=Label(Image=img).pack()
secondbtn=Button(root,text="Remove mask", command=removemask).pack



btn=Button(root,text="Open File", command=open).pack()










root.mainloop()