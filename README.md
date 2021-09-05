# FingerSpell-ASL

The CNN is trained on a custom dataset containing alphabets A-Y (excluding J) of American Sign Language.

## Usage 

### To run from Website

#### capture button 
This is used to dd new gueture to our language.
Set up the path in the server.py @route /capture
Place your hand in the green box and press C to start capturing the data.

#### process button
This is to process the capture image and make it into tresh image
Now set up the paths in server.py @route /process to preprocess the dataset.
After preprocessing set up the path in model.py file to get the preprocessed data for training.

#### translate button
After preprocessing set up the path in model.py file to get the preprocessed data for training.
This will create model-<val_acc>.h5 files. Choose the appropriate file for Gesture_recognize_sign.py


### To run the pretrained model

Run:

```
python Gesture_recognize_sign.py
```

This will start the webcam.Press C then place your hand inside the green box while performing a gesture
and you will get the letter to which the respective gesture corresponds. Press Q to quit.

### To train your own model

Set up the path in the Image_capturing.py file

Run:

```
python Image_capturing.py
```

Place your hand in the green box and press C to start capturing the data.

Now set up the paths in Image_preprocessing.py file to preprocess the dataset.

Then Run:

```
python Image_preprocessing.py
```

After preprocessing set up the path in model.py file to get the preprocessed data for training.

Then Run:

```
python model.py
```

This will create model-<val_acc>.h5 files. Choose the appropriate file for Gesture_recognize_sign.py

Then Run:

```
python Gesture_recognize_sign.py
```

## screenshots
![Test Image 1](https://github.com/123aryankhandelwal/FingerSpell-ASL/blob/main/Images/1.gif)
![Test Image 2](https://github.com/123aryankhandelwal/FingerSpell-ASL/blob/main/Images/2.png)
![Test Image 3](https://github.com/123aryankhandelwal/FingerSpell-ASL/blob/main/Images/3.png)
![Test Image 4](https://github.com/123aryankhandelwal/FingerSpell-ASL/blob/main/Images/4.png)
![Test Image 5](https://github.com/123aryankhandelwal/FingerSpell-ASL/blob/main/Images/5.png)
