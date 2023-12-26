# OpticReader v7.0
> Welcome to the seventh iteration of my attempt to write an OMR from scratch.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
OpticReader is a fully autonomous OMR script capable of detecting manually marked answers, scoring them and rank the 
participant based on their score. 

It was developed to be used in simulation contexts of the national entrance test for the faculty of 
"Medicine and Surgery". 

In particular, the development started as a personal project and, since December 2021, it has continuously been
used in real world scenarios during  [_"Studenti e Prof Uniti per"_](https://studentieprofunitiper.it/) association's 
simulations.


## Technologies Used
- dearpygui - version 1.6.2: for GUI
- xlsxwriter - version 3.0.3: for the creation and population of .xlsx files
- python-barcode - version 0.15.1: for reading and writing barcodes
- pyzbar - version 0.1.9: as python-barcode
- scikit-learn - version 1.3.1: for training, evaluating and predicting with SVC and KNN models 
- opencv-python - version 4.6.0.66: for handling images (read, write and operate on them)
- numpy - version 1.24.4

## Assumptions
As far as **filling the exam sheet** goes this were the latest rules given:
- an X on a square means that the selected option (A to E) has been chosen for a given question
- a blank square should be ignored
- a fully darkened square means that the selected option should be ignored (as for blank squares)
- a marked circle (either darkened or marked with an X) means that whatever was chosen in the corresponding square should be ignored
- there is no limit to the number of squares that can be darkened per question. It goes without saying that 5 darkened squares mean that the question has not been answered

With the introduction of randomized exam's question in 2023 for the computer based entrance exam the MUR developed a **scoring 
system** that accounts for the difficulty of each question. Detailed explanations can be found [_here_](https://www.mur.gov.it/sites/default/files/2022-09/Decreto%20Ministeriale%20n.%201107%20all.%202_valutazione%20delle%20prove%20e%20attribuzione%20dei%20punteggi.pdf). *An english version translated by me will be uploaded in the future*.

## Features
As stated above, OpticReader can take a directory full of scanned exam sheets (check "blank exam" in the 
[Screenshots](#screenshots) section) and:
- detect the barcode identifier;
- autonomously align the scanned image;
- traverse the whole exam sheet identifying for each question the hand written answer, if given (check "example exam" in the 
[Screenshots](#screenshots) and the [Assumptions](#Assumptions) section). This is done by submitting each circle/square to three different classifiers. Each of them classify the given image and the most voted option gets chosen;
- from the answers that have been detected a score is generated (as stated in the [Assumptions](#Assumptions) section) by comparing them to the ones given as correct;
- after all the exams are evaluated, the difficulty score is calculated and applied to all users' scores;
- a .xlsx file is automatically generated where all the users are ranked;
- exam evaluation is currently done with multithreading.

## Screenshots
Some helper screenshots for those who are unfamiliar with the exam sheet layout (this is a recreation created by me)
### Base Template
![Base template](screenshots%2Freduced_res_50QUES.png)
### Sample input
This is what a correctly filled form could look like after being scanned. 
![sample_input.jpg](screenshots%2Fsample_input.jpg) 
### Sample output
This is how OpticReader evaluate the sample input exam sheet. The green square means that the selected option has been evaluated as chosen.    
![sample_output.png](screenshots%2Fsample_output.png)

## Setup
Check the [Technologies Used](#technologies-used) section for requirements. Microsoft Excel is necessary for opening the rankings file at the end of the app execution.
After you have cloned this repository be sure to run BarCodeCreatorPdfMerger.py in order to convert the Base template to a "750 pages single pdf" where each page is an exam sheet with its own barcode.


## Usage
1. After scanning your exam sheets, you should have a directory where each exam sheet is a single image (filename does not matter). 
2. Clone this directory or manually download the files
3. You can launch `main.py` from cmd (after you have moved to the correct directory) by saying `python main.py`. 
4. For non-development use case insert 1 and press enter.
5. A new window will appear. Here you can modify the solutions to each question but if you want to start a scoring session, paste the absolute path to your scanned exam images in the bottom right text field and press OK
6. At this point, if you are using the standard template you can just press Si and wait. 
7. It is not recommended to change any other option apart from MultiThread. If disabled, you can add debug options from the dropdown menù
8. Wait for the scoring loop to finish. You will know when it's done because an "Excel" button will appear under the path text box
9. Click on the "Excel" button so that Microsoft Excel can be launched to open the rankings file.


## Project Status
OpticReader is _complete_ and _no longer being worked on_. 

After almost three years of occasional development and the 
achievement of reasonable results such as speed, precision and a user-friendly GUI I am planning to move on to other projects.

This, in addition to the fact that paper based exams are no longer a standard has given me the motivation to stop working on this (awsome nonetheless) project.

## Room for Improvement
The fact that I am no longer actively working on OpticReader does not mean that it is perfect. Here there are some ideas I would like to implement someday:

Room for improvement:
- despite being quite robust the **alignment** algorithm does not catch all edge cases. It would be awsome to modify the current template in order to use warpAffine function on three predetermined objects (like positional marker printed on the corners);
- the current three-voting **classification** system is based on a SVC, KNN and a personally developed case specific algorithm. It would be interesting to see if a CNN could be implemented for such task and if new non-ML based algorithm can be developed for this task;
- current **GUI** is functional and user-friendly. Major work can be done to aesthetic and flexibility;
- Currently, if run with default parameters, OpticReader runs with **multiprocessing**. Being this the latest addition and not my strongest topic, additional refinement can (and probably should) be done. 



## Acknowledgements
Many thanks to "Studenti e prof Uniti per" association's directors, in particular to L. Barbato and A. Marchi for letting me develop and use OpticReader at the association's simulation.


## Contact
Created by [Lorenzo Paolin](https://github.com/lorenzopol). If you have any request, doubt, or you need to contact me, feel free to open an Issue [*here*](https://github.com/lorenzopol/OpticReader-v7.0). 
