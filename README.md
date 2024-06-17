# nlp2024_ClefTask4SOTA
This repository contains to code our solution to the shared task "SOTA" from the SimpleText Lab at the CLEF 2024 conference.

Main author: Urban Lutz

In the notebook experiments.ipynb, the main logic to conduct experiments is implemented, therefore this is a great starting point to review our code.
The file data_exploration.ipynb and section_analysis.ipynb were used to analyze our dataset and the content extraction logic. 
The files prefixed by word2vec are partial implementations to improve the content extraction logic.

The folder scoring_program contains the official evaluation script as published by the authors. We apply it to our data in the file evaluation.ipynb.