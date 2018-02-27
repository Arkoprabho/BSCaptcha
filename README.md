# BSCaptcha
The motive of this repository is to understand how useful are [captchas](https://en.wikipedia.org/wiki/CAPTCHA).
CAPTCHA stands for "Completely Automated Public Turing test to tell Computers and Humans Apart".
Now if we were to go by the word, we should not be able to detect the text written in the said CAPTCHA using neural networks.
The interesting thing here is that Neural Networks aim to model the human brain. A program that mimics humans.
A succesful neural network is one whose outcomes cant be distinguished from that of a human.
This competition will surely prove useful in the field of CAPTCHAs

This is what the objective is.
The idea is not to break captchas, but to see and understand what makes them strong if they cant be broken.
This aims to be a study of CAPTCHAs from an AI POV.

### Tools used
We will be using [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) as our primary deep learning toolkit.
The dataset is an artificial dataset generated using [captcha](https://github.com/lepture/captcha).
All the required libraries will be listed in the [requirements](requirements.txt) file.