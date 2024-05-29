from keras.models import load_model
import numpy as np
import cv2
from tkinter import *
import tkinter as tk
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
import pyperclip

#Declare global variables
canvas_height = 300
canvas_width = 1000
model_image_size = 45 #45x45
final_expression = ""
accuracy = None

#Load model and classes
model = load_model('model.h5')
classes = [label.strip() for label in list(open('dataset\\classes.txt', 'r'))]

#Function for input processing and character prediction
def predict(exp):
    global final_expression
    global accuracy
    final_expression = ""
    accuracy = 0.0
    sqrtInd = 0
    sqrtEnd = 0
    count = 0
    baseY = 0
    baseH = 0
    exponentInd = 0

    #Convert image into numpy array
    exp = np.array(exp)
    
    #Perform adaptive thresholding to prepare for connected component analysis
    exp = cv2.adaptiveThreshold(exp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    
    #Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(exp)
    
    #Sort the unique labels according to their position along the x-axis    
    label_pos = []
    for label in np.unique(labels):
        x,y = np.where(labels == label)
        label_pos.append((label, x.min, y.min()))
    sorted_labels = sorted(label_pos, key=lambda y: y[2])
    sorted_labels = [label[0] for label in sorted_labels]

    #Predict characters label by label
    for (i, label) in enumerate(sorted_labels):
        #If this is the background label, ignore it
        if label == 0:
            continue
        
        count += 1
        
        #Otherwise, construct the label mask to display the connected component for the current label
        labelMask = np.zeros(exp.shape, dtype="uint8")
        labelMask[labels == label] = 255

        #Find contours and get bounding box for each contour
        cnts, _ = cv2.findContours(labelMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundingBox = [cv2.boundingRect(c) for c in cnts]  

        #Get the coordinates from the bounding box
        x,y,w,h = boundingBox[0]
        
        #Crop the character from the label mask and apply bitwise_not to revert the image
        segment = labelMask[y:y+h, x:x+w]
        segment = cv2.bitwise_not(segment)

        #Add padding to ensure 1:1 ratio for thin or thinly-drawn characters
        basis = max(segment.shape[0], segment.shape[1])
        pad_shape = (basis, basis)

        row_x = int((pad_shape[0] - segment.shape[0]) / 2)
        row_y = int((pad_shape[0] - segment.shape[0] + 1) / 2)
        column_x = int((pad_shape[1] - segment.shape[1]) / 2)
        column_y = int((pad_shape[1] - segment.shape[1] + 1) / 2)
            
        pad_width = [(int(row_x), int(row_y)), (int(column_x), int(column_y))]
        segment = np.pad(segment, pad_width, mode='constant', constant_values=255)

        #Process the image to match with model requirements
        image = cv2.resize(segment, (45,45), interpolation=cv2.INTER_AREA)

        #image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 20)
        
        #plt.imshow(image, cmap='gray', interpolation='none')
        #plt.show()

        image = image.reshape(-1,45,45,1)
        image = image.astype('float32')
        image = image/255.0

        #Predict the class of the processed image and get the accuracy
        result = model.predict([image])[0]
        class_index = np.argmax(result)
        accuracy += (max(result)*100)
        
        character = classes[class_index]

        #For special characters
        if character[0] == '\\':
            if character == '\\pi':         #small greek letter pi
                character = '\u03C0'
            elif character == '\\times':    #multiplication symbol
                character = '*'
            elif character == '\\gt':       #greater than symbol
                character = '>'
            elif character == '\\lt':       #less than symbol
                character = '<'
            elif character == '\\neq':      #not equal symbol
                character = '≠'
            elif character == '\\sqrt':     #square root symbol
                character = '\u221A' + '{'
                sqrtInd = 1     #indicator that there is a square root character
                sqrtEnd = x+w   #obtains the rightmost x-coordinate of the square root
                                #to identify which characters are inside
        
        if sqrtInd == 1: #if there is a square root character
            if count == (len(np.unique(labels)) - 1): #check if the current character is the last character
                character = character + '}'
                sqrtInd = 0
            elif x+w > sqrtEnd: # check if there's a proceeding character outside the square root symbol
                character = '}' + character
                sqrtInd = 0
        
        exponents_banned = [2, 3, 4, 5, 35, 36, 37, 40] #specify indices of characters that can't be an exponent
        if y+h < (baseY+(baseH/2)) and class_index not in exponents_banned and exponentInd == 0:
            character = '^{' + character
            exponentInd = 1
        
        if exponentInd == 1 and count == (len(np.unique(labels))-1):
            character = character + '}'
            exponentInd = 0
        
        if y+h > (baseY+(baseH/2)) and exponentInd == 1:
            character = '}' + character
            exponentInd = 0
        
        if exponentInd == 0: 
            baseY, baseH = y, h
        
        final_expression += character   #append the individual characters being predicted
    
    accuracy = accuracy / count   #obtain the average accuracy
    return final_expression, accuracy

#Create Tkinter canvas
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("Handwritten Mathematical Expression Recognizer")

        self.x = self.y = 0
        
        # Creating elements
        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw an expression.", font=("Segoe UI", 15), wraplength=200)
        self.recognize = tk.Button(self, text = "Recognize", command = self.get_image)   
        self.clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.copy = tk.Button(self, text = "Copy to Clipboard", command = self.copyclip)
        
        #Create an image based on the drawing inside the canvas
        self.image = Image.new("L", (canvas_width, canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=N)
        self.label.grid(row=0, column=1,pady=2, padx=2, columnspan=2)
        self.recognize.grid(row=1, column=1, pady=2, padx=2)
        self.clear.grid(row=1, column=0, pady=2)
        self.copy.grid(row=1, column=2, pady=2, padx=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    #For clearing the canvas
    def clear_all(self):
        global final_expression
        global accuracy

        self.canvas.delete("all")
        self.image = Image.new("L", (canvas_width, canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label.configure(text="Draw an expression.")
        final_expression = ""
        accuracy = None
    
    #For drawing
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=3
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
        self.draw.ellipse([self.x-r, self.y-r, self.x + r, self.y + r], fill="black")
    
    #For obtaining the image drawn in the canvas and printing the prediction
    def get_image(self):
        expression = self.image
        prediction, accuracy = predict(expression)
        self.label.configure(text = 'Prediction:' + '\n' + str(prediction) + '\n\n' + 'Accuracy: ' + str(int(accuracy))+'%')       
    
    #For copying the text to the user's clipboard
    def copyclip(self):
        pyperclip.copy(final_expression)
        self.label.configure(text = self.label["text"] + '\n\n' + "The expression has been successfully copied to your clipboard.")

app = App()
mainloop()