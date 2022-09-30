# Deepfake Classifier
### Classification model for KING5 News

## Project Overview:
As deepfakes become more convincing their consequences become more severe. Originally deepfakes were popularized through Reddit in 2017, and started as a way to create celebrity pornography and face swaps of Nicholas Cage in movies he hadn't starred in. These may seem fairly harmless and even though Deeptrace, a Dutch cybersecurity firm, stated in 2019 that 96% of deepfakes online are pornographic, some of the remaining 4% have been created with malice. In 2020, during the presidential campaign, videos of Joe Biden were created using deepfake technology to advance a narrative of mental decline. Just one year before the House Intellegence Commity held hearings on the possibility of this scenerio. With the fakes now being so indistinguishable from reality true videos can now be denied and claimed to be deepfakes.

## Business Problem:
KING5 News recognizes that deepfakes and people claiming a true video is a deepfake will only continue to become commonplace. News outlets have always needed ways of fact checking and as the technologies of deception become more advanced, the means of detecting them must also advance.

KING5 News would like to integrate a trained model into their fact checking pipeline as a way of validating new sources and proving validity of video.

## Data Understanding
Our model has been trained off of nearly 25,000 still images supplied to us through [FaceForensics](https://github.com/ondyari/FaceForensics). This data was created by FaceForensics and consists of 1000 modified YouTube videos. FaceForensics offers five types of modifications for each of the most common types of Deepfakes including, Deepfake, DeepfakeDetection, Face2Face, FaceSwap, and NeuralTextures. This project focuses on the detection of traditional Deepfakes. I selected videos still in a way to create a balanced dataset and chose to balance its train/test/val to a ratio of .65/.2/.15.  

## Modeling and Results:
Throughout this process we were tasked with identifying if an image originated as a deepfake. By iterating through multiple versions of a Convolutional Neural Network we were able to build a model to make this classification. In the end we chose to go with a simple VGG-16 model as opposed to building our own network because of hardware limitations. After using our training data to form our model and our validation set to confirm it's fitting we ran our testing data through to confirm our final results. After multiple iterations we were satisfied with the results of our model. On unseen data our final model accuracy is 84% and has a precision of 99%. Compared to our first simple model with an accuracy of 50% this is a sizeable increase.

## Evaluation:
I chose to go with a model that performs best in accuracy and coupled with its high precision I believe this model to be reliable. 

For our next steps we would like to begin creating a model for the other more advanced types of deepfakes. Although not comprehensive this model will be able to verify some deepfakes and should be implemented immediately but until all models are created it would be unwise to claim that a video has not been created by some deepfake technology simply because it passes this test.

## Link:
[Deepfake Detector](https://jordanmang-deepfakeclassifier-app-qaqsja.streamlitapp.com/)
