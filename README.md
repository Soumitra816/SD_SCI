This folder contains the code and data for the implementation of the work titled: 
"Blunt Giants, Sharp Specialists: Decoding Stress and Triggers Beyond General-Purpose LLMs"

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We release a sample of our dataset for review purposes. The full dataset will be released upon acceptance. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SACReD_sample.csv file contains 1003 utterances from 209 social media posts. 

Columns:
* Para_index - Post ID
* Seq_inside_para - Sentence ID within a post
* Column_of_sentences - Utterances 
* Stress - 0 indicates Non-Stress, 1 indicates Stress
* Stress_cause - 0 indicates Non-causal utterance, 1 indicates Causal utterance
* label_of_emotion - Emotion of a particular utterance


model.py presnet the main function for the working methodology of the study.

The LLM Baselines folder primarily consists two category of files:
1. sacred.py = relates to the results for stress detection tasks. Model paths are to be changed as per the requirement.
2. sacred_sci_.....py = relates to the results for stress cause identification task. One file each for reported results from each model.
