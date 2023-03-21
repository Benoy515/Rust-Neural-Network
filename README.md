# Rust-Neural-Network
CS128 Honors project

## Group Name: Lobsters
## Group Members: Abhay Benoy(abhayb2), Udit Karthikeyan(uditk3), Pranav Swaminathan(pswam2), Josh Neela(jneel5)

## Project Intro:
The goal of this project is to program a simple neural network capabable of indentifying handrittwen digits. The training data will come from the MNIST handritten digits dataset which contains thousands of 28 by 28 pixel grayscale images with labels. The neural network will be coded from scratch without any external ML crates.

## Technical Description:
The first step is to develop functions to calculate the essential parts of a neural network and manage data input. Forward feed will require vectors to represent weights and biases of each layer and a function to dot two vectors. We will likely install a linear algebra crate to expedite this process. The images also have to be scanned and read into vectors that the program can use for training.

Next is a backpropogation function. It should calcualate the gradient for each neuron in the network and adjust the weights and biases accordingly.

The final phase involves testing various networks with different numbers of layers and training them on the dataset to see how accurate they are.

## Potential Challenges:
Debugging this program will be likely be a slow and tedious process since every network needs to be trained first to measure it's accuracy. Also converting the images to vectors may prove challenging since we have yet to try reading and cleaning up external data in Rust.
