# for-fun
In this repository, I include some of the projects I have enjoyed the most while completing a 5-course specialization on deep-learning. The models were built in Keras with TensorFlow backend.

# Important
I am sharing these projects for your reference and to illustrate how interesting deep learning projects are. If you are enrolled in a machine-learning/deep learning course and have an assignment similar to any of the projects that I present here, please do not copy this work. Instead, try to complete the assignment on your own with the help of the course material, instructors/TAs and the course discussion forum. 

# Project 1: Use a LSTM to generate Jazz music.
*Objective*: build and train a network to generate novel jazz solos in a style representative of a body of performed work.

*Dataset*: the dataset was provided by the course instructors and consisted on a snipped of jazz music audio. The audio was pre-processed and each music note was represented by one of 78 different possible values. Each training set consisted of a sequence of 30 notes (or music values). 

*The model*: The architecture of the model is shown in the figure below.
![LSTM-architecture](https://github.com/plesqui/for-fun/blob/master/LSTM_architecture.png?raw=true "LSTM-Architecture")

*Generation of Jazz music*: After building and training the model (LSTM_Jazz.py) to learn the patterns of jazz music, the next step is to use the model to synthesize new music. At each step of sampling, you will take as input the activation a and cell state c from the previous state of the LSTM, forward propagate by one step, and get a new output activation as well as cell state. The new activation a can then be used to generate the output, using densor as before. The code to generate music is included in 'LSTM_generate_Jazz.py'.
![LSTM-generation](https://github.com/plesqui/for-fun/blob/master/generation_music.png?raw=true "LSTM-Generation")

*Results*: I have uploaded the music generated by the LSTM model that I built on the deep learning course. You can find it in the following link: https://github.com/plesqui/for-fun/blob/master/my_music.midi
