import tkinter as tk
import cv2 as cv
import tensorflow as tf
from tkinter import filedialog #文件打开
from model_reload import net_structure
import numpy as np
from PIL import ImageTk,Image,ImageGrab

filename = None

#网络结构与模型加载
net = net_structure()

def paint(event):
    canv.create_rectangle(event.x,event.y,event.x+5,event.y+5,fill="black")

def clear_all():

    canv.create_rectangle(0,0,200,200,fill='white')
    v.set(" ")
    label1.update()

def recognition():
    img = ImageGrab.grab((window.winfo_x()+840,window.winfo_y()+33,window.winfo_x()+824+204,window.winfo_y()+233))
    img.save('1.png')

    img1 = cv.imread('1.png')
    img1 = cv.resize(img1,(28,28))
    img_res = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
    #二值化
    ret,binary1 = cv.threshold(img_res,0,255,cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
    for i in range(28):
        for j in range(28):
            binary1[i][j]=255-binary1[i][j]
    #cv.imshow("4",binary1)

    #平滑滤波
    blu_img = cv.blur(binary1,(5,5))

    #img_res = cv.blur(img_res,(5,5))
    res = np.reshape(blu_img,784)
    y_pre = tf.argmax(net.prediction,1)
    predint = y_pre.eval(feed_dict={net.xs:[res],net.keep_prob:1},session=net.sess)

    v.set(str(predint[0]))
    label1.update()

#图片打开
def open_img():
    global filename
    global o_img
    filename = filedialog.askopenfilename(title=u'选择文件')
    o_img = Image.open(filename)
    o_img = o_img.resize((200,200))
    o_img = ImageTk.PhotoImage(o_img)

    canv.create_image(100,100,image=o_img)


#窗口设置
window = tk.Tk()
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
size = '%dx%d+%d+%d' % (1024, 618, (screenwidth - 1024)/2, (screenheight - 618)/2)
window.resizable(False,False)#大小不可变
window.config(bg='AliceBlue')
window.geometry(size)

v = tk.StringVar()

#画布设置
canv = tk.Canvas(window,width=200,height=200,bg='white')

# if filename == None:
#     print("ddd")
#     pass
# else:
#     o_img = Image.open('1.png')
#     o_img = o_img.resize((200,200))
#     o_img = ImageTk.PhotoImage(o_img)
#     canv.create_image(100,100,image=o_img)

# o_img = Image.open('1.png')
# o_img = o_img.resize((200,200))
# o_img = ImageTk.PhotoImage(o_img)
# canv.create_image(100,100,image=o_img)

canv.place(x=824,y=0)
canv.bind('<B1-Motion>',paint)

#画布2设置
canv1 = tk.Canvas(window,width=826,height=618,bg='AliceBlue')
img_path = "模型结构图2.png"
img = Image.open(img_path)
photo = ImageTk.PhotoImage(img)
canv1.create_image(411,250,image=photo)
canv1.create_line(823,0,823,618,width=3,fill='gray')
canv1.create_text((410,520),text="网络---结构图",font=("楷体",20))
canv1.place(x=0,y=0)

#按钮设置
button1 = tk.Button(window,text='清空',width=10,height=1,command=clear_all)
button2 = tk.Button(window,text='识别',width=10,height=1,command=recognition)
button3 = tk.Button(window,text='打开图片',width=10,height=1,command=open_img)
button1.place(x=890,y=210)
button2.place(x=890,y=250)
button3.place(x=890,y=290)

#文本窗口
label = tk.Label(window,text="识别结果为：",width=13,height=1,bg='AliceBlue',font=("宋体",13))
label.place(x=830,y=330)

#text 窗口
label1 = tk.Label(window,textvariable=v,width=6,height=1,font=("宋体",13))
label1.place(x=940,y=330)

window.mainloop()
