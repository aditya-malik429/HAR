# Human-activity-recognition

action_recognition_kinetics.txt : The class labels for the Kinetics dataset.
resnet-34_kinetics.onx : Hara et al.’s pre-trained and serialized human activity recognition convolutional neural network trained on the Kinetics dataset.Model Link https://drive.google.com/drive/folders/13kswALlOLN6bap5ExqC-q0SNB3D2cnKQ


example_activities.mp4 : A compilation of clips for testing human activity recognition.
We will review two Python scripts, each of which accepts the above three files as input:

human_activity_reco.py : Our human activity recognition script which samples N frames at a time to make an activity classification prediction.
human_activity_reco_deque.py : A similar human activity recognition script that implements a rolling average queue. This script is slower to run; however, I’m providing the implementation so that you can learn from and experiment with it.

Human Activity Recognition with OpenCV and Deep Learning
We begin with imports on Lines 5-10 you need OpenCV 3 or 4 and imutils installed


Lines 10-20 parse our command line arguments:
--model : The path to the trained human activity recognition model.
--classes : The path to the activity recognition class labels file.
--input : An optional path to your input video file. If this argument is not included on the command line, your webcam will be invoked.

Line 25 loads our class labels from the text file.

Lines 26 and 27 define the sample duration (i.e. the number of frames for classification) and sample size (i.e. the spatial dimensions of the frame).

Next, we’ll load and initialize our human activity recognition model:

Line 31 uses OpenCV’s DNN module to read the PyTorch pre-trained human activity recognition model.

Line 32 then instantiates our video stream using either a video file or webcam.

We’re now ready to begin looping over frames and performing human activity recognition:



Line 34 begins a loop over our frames where first we initialize the batch of frames  that will be passed through the neural net (Line 41).

From there, Lines 44-52 populate the batch of frames  directly from our video stream. Line 56 resizes each frame to a width  of 400  pixels while maintaining aspect ratio.

Let’s construct our blob  of input frames which we will soon pass through the human activity recognition CNN:

Lines 60-64 construct a blob  from our input frames  list.

Notice that we’re using the blobFromImages (i.e. plural) rather than the blobFromImage (i.e. singular) function — the reason here is that we’re building a batch of multiple images to be passed through the human activity recognition network, enabling it to take advantage of spatiotemporal information.

If you were to insert a print(blob.shape) statement into your code you would notice that the blob has the following dimensionality:

(1, 3, 16, 112, 112)
Let’s unpack this dimensionality a bit more:

1: The batch dimension. Here we have only a single data point that is being passed through the network (a “data point” in this context means the N frames that will be passed through the network to obtain a single classification).
3: The number of channels in our input frames.
16: The total number of frames  in the blob .
112 (first occurrence): The height of the frames.
112 (second occurrence): The width of the frames.
At this point, we’re ready to perform human activity recognition inference followed by annotating the frame with the predicted label and showing the prediction to our screen:

Human Activity Recognition Results
open up a terminal and execute the following command:
python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt
