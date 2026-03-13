# BiggerBrain-AI
This is the official repo for the AI MaRT titled "BiggerBrain". It contains code for the whole project. 
MaRT stands for 'Memory augmented Recurrent Transformer'. It uses a GRU and a custom memory manager to maintain long term dependencies, a RoPE attention based Recurrent Tranformer. I define a Recurrent Transformer as a transformer which has around ~50% of the layers/blocks are in a recurrent loop in which they continue thinking on a token. This means that there are ~30-40% of the layers "single shot" before the loop in the forward pass, and the rest for the post-middle language clarification. It is designed to be a
maximally smart AI with dense intelligence for its parameter count.

The AI_main.py is the file to run if you would like to test this project. It has a few simple commands implemented, but it is not case sensitive:
"debugmode": enables debug printing; needs work
"train": trains on the file titled "trainfilename" for train_lr Learning rate and all of the other defaults
"quit": ends the program
"speedtest": tests loading ______
"pretrain": trains on "filename" for lr learning rate and other defaults
"profile": uses pytorch profiler to test model performance and bottlenecks
"filesize": gets filesize of "filename"
"anything else": It will run the AI on your input.
