Code in this directory downloads and preprocesses data for the species classifier and trains and validates classifier models.

Please install the dependencies listed in the environment.yml file.

Download these data files into this directory:

https://drive.google.com/file/d/10TBjO040piAv25f0HPWQq1noR_M02FD1/view?usp=sharing
https://drive.google.com/file/d/19zFDFWoz7a5NJ60TzWAmJJlxe98ODxBL/view?usp=sharing

Download the data:

python download.py

Sanitise the data:

python sanitise.py

Split the data:

python split.py

Fine-tune an ImageNet21K-pretrained EfficientNetV2 model using the data:

python fine_tune.py <model_size> <training_split> <load_checkpoint> <port>

<model_size>: choose among "s", "m", and "l"
<training_split>: choose between "train" and "full"
<load_checkpoint>: option to resume training in case of interruption, enter index of checkpoint to resume, or "None" if first starting training
<port>: port for parallel training

Example: python fine_tune.py s train None 8888
Number of GPUs and batch size per GPU can be modified in fine_tune.py

Validate fine-tuned models:

python validate.py <model_size> <checkpoint_directory> <beginning_checkpoint> <end_checkpoint>

<model_size>: choose among "s", "m", and "l"
<checkpoint_directory> & <beginning_checkpoint> & <end_checkpoint>: self explanatory

Example: python validate.py s s_train 0 500
