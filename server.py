import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, request, jsonify, make_response, Response, render_template
import os

app = Flask(__name__)

model = load_model('./new_model6.h5')


@app.route('/')
def index():
    return render_template('index.html',title='FingerSpell-ASL')


gestures = {
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    # 1:'0',
    # 2:'1',
    # 3:'2',
    # 4:'3',
    # 5:'4',
    # 6:'5',
    # 7:'6',
    # 8:'7',
    # 9:'8',
    # 10:'9',
    # 11:'A',
    # 12:'B',
    # 13:'Best of luck',
    # 14:'C',
    # 15:'D',
    # 16:'E',
    # 17:'F',
    # 18:'G',
    # 19:'H',
    # 20:'I',
    # 21:'I Love you',
    # 22:'J',
    # 23:'K',
    # 24:'L',
    # 25:'Love',
    # 26:'M',
    # 27:'N',
    # 28:'O',
    # 29:'P',
    # 30:'Q',
    # 31:'R',
    # 32:'S',
    # 33:'T',
    # 34:'U',
    # 35:'V',
    # 36:'W',
    # 37:'X',
    # 38:'Y',
    # 39:'Z'
}


def predict(gesture):
    img = cv2.resize(gesture, (50,50))
    img = img.reshape(1,50,50,1)
    img = img/255.0
    prd = model.predict(img)
    index = prd.argmax()
    return gestures[index]


@app.route("/translate")
def translate():
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    old_text = ''
    pred_text = ''
    count_frames = 0
    total_str = ''
    flag = False


    while True:
    
        if frame is not None: 
        
            frame = cv2.flip(frame, 1)
            frame = cv2.resize( frame, (400,400) )
        
            cv2.rectangle(frame, (0,0), (200,200), (0,255,0), 2)
        
            crop_img = frame[0:200, 0:200]
            grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
            thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
      
        
            blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(blackboard, "Predicted text - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            if count_frames > 20 and pred_text != "":
                total_str += pred_text
                count_frames = 0
            
            if flag == True:
                old_text = pred_text
                pred_text = predict(thresh)
        
                if old_text == pred_text:
                    count_frames += 1
                else:
                    count_frames = 0
                cv2.putText(blackboard, total_str, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
            res = np.hstack((frame, blackboard))
        
            cv2.imshow("image", res)
            cv2.imshow("hand", thresh)
        
        rval, frame = vc.read()
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag = True
        elif keypress == ord('q'):
            return render_template('index.html',
                           title='abc')

    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    vc.release()


@app.route("/capture")
def capture():
    vc = cv2.VideoCapture(0)
    pic_no = 0
    total_pic = 1200
    flag_capturing = False
    path = 'capture/A'
    while(vc.isOpened()):
        # read image
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)
    
        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(frame, (0,0), (200,200), (0,255,0),0)
    
        cv2.imshow("image", frame)
    
        crop_img = frame[0:200, 0:200]
    
        if flag_capturing:
        
            pic_no += 1
            save_img = cv2.resize( crop_img, (50,50) )
            save_img = np.array(save_img)
            cv2.imwrite(path + "/" + str(pic_no) + ".jpg", save_img)
        
    
        keypress = cv2.waitKey(1)
    
        if pic_no == total_pic:
            flag_capturing = False
            break
    
        if keypress == ord('q'):
            return render_template('index.html',
                           title='abc')
        elif keypress == ord('c'):
            flag_capturing = True

    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    


@app.route("/Process")
def Processing():
    path = 'capture'
    path2 = 'Process'

    gestures = os.listdir(path)

    for ix in gestures:
        images = os.listdir(path +'/'+ix)
        os.mkdir(path2 +'/'+ ix)
        for cx in images:
            print(cx)
            img_path = path +'/'+ ix +'/' + cx
            print(img_path)
            img = cv2.imread(img_path)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            save_img = cv2.resize(thresh, (50,50))
            cv2.imshow('img',save_img)
            cv2.imwrite(path2 + '/'+ ix + '/' + cx, save_img)
    return render_template('index.html',
                           title='abc')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
