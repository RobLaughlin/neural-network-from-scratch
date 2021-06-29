import tkinter
from Network.network import NeuralNetwork
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import math
import numpy as np

class CustomCanvas(tkinter.Canvas):
    def __init__(self, root, width, height, bg, outline, fill, radius, guess_text, network_path):
        super().__init__(root, width=width, height=height, bg=bg)
        self.bg = bg
        self.width = width
        self.height = height
        self.fill = fill
        self.outline = outline
        self.radius = radius
        self.guess_text = guess_text

        self.network = NeuralNetwork.load_network_object(network_path)

        self.img = Image.new("RGB", (width, height), bg)
        self.draw = ImageDraw.Draw(self.img)

        root.bind('<B1-Motion>', self.click_move)
        root.bind('<ButtonRelease-1>', self.leftclick_up)

    def click_move(self, event):
        x = event.x
        y = event.y
        
        self.create_oval(x, y, x + self.radius, y + self.radius, fill=self.fill, outline=self.outline)
        self.draw.arc(xy=[x, y, x + self.radius, y + self.radius], start=0, end=360, fill=self.fill, width=self.radius)

    def leftclick_up(self, event):
        resized_img = self.img.resize((28, 28))
        grayscale_img = ImageOps.grayscale(resized_img)
        brightened_img = np.array(grayscale_img)
        brightened_img[brightened_img > 0] = 255

        #im = Image.fromarray(brightened_img)
        #im.save('test.png')

        brightened_img = brightened_img.astype(np.float64) / 255
        brightened_img = brightened_img.flatten()

        self.network.load_data(brightened_img)
        predicted = self.network.predict()
        label_text = "I see a {0} (Certainty {1:.2f}%)".format(predicted[1], predicted[0] * 100)
        self.guess_text.set(label_text)

def clear_btn_clicked(canvas: CustomCanvas):
    canvas.delete('all')
    canvas.draw.rectangle((0, 0, canvas.width, canvas.height), fill=(canvas.bg))

if __name__ == '__main__':
    width = 200
    height = 200

    bgcolor = 'black'
    brushcolor = 'white'
    brush_radius = 5

    network_obj_path = './network.pkl'

    root = Tk()
    root.title = "Digit Guessing Demo"

    guess_text = StringVar()
    guess_label = Label(root, textvariable=guess_text)

    cv = CustomCanvas(root, 
        width=width, height=height, 
        bg=bgcolor, outline=brushcolor, 
        fill=brushcolor, radius=brush_radius, 
        guess_text=guess_text, network_path=network_obj_path)
    cv.pack()

    clear_btn = Button(root, text='Clear', command = lambda cv=cv: clear_btn_clicked(cv))
    clear_btn.pack()
    guess_label.pack()
    
    cv.mainloop()