# DATP_project

A mathematical proof is a sequence proof states derived by application of valid axioms and inference rules. Thus, a derivation of a proof can be perceived as a translation task from one proof state to another. State-of-the art approaches aim to predict the set of inference rules which when applied to one proof state, produces the next proof state. This work aims to learn this translation without taking into account the inference rule at action. Thus, in this work, the proof derivation is defined as a sequence-to-sequence translation task using two slighlty different versions the data (termed as tasks). The dataset explored in this work is the Metamath’s ‘set.mm’ partition, restricted to proof lengths of upto 10 and 15. The task descriptions and the corresponding results can be found in Report.pdf.

## Download set.mm using 
!wget https://github.com/metamath/set.mm/raw/develop/set.mm

## Preprocess the data
python pre_process.py

## For task 1
python main_task1.py

## For task 2
python main_task2.py
