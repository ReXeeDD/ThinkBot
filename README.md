## ThinkBot -LLM Checkpoint Project
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-
A repository containing  files for a Transformer-based Large Language Model (LLM) compatible with the Hugging Face Transformers library.

A conversational AI chatbot powered by a fine-tuned DialoGPT-medium model from Microsoft.
-
-The txt file used to fine tune my model file was just a 3000 line conversation ,thus only a feasible output was generated ,but a large conversation file can produce a better output and will be computationally expensive,this project was just made in 1 hrs including the training part so if you have time to adjust the codes and use a better conversation.txt a very good output can be observed.


## 🚀 Features
- Pretrained/fine-tuned Transformer-based LLM
- Hugging Face Transformers compatible
- Safetensors format for secure model loading

## 💻 Installation
```bash
# Clone repository
git clone https://github.com/ReXeeDD/ThinkBot.git
```

# Install requirements
-
torch,
transformers 
-

## Model Download

-
````
https://huggingface.co/ReXeeD/ThinkBot/upload/main
````
download
-
config.json,generation_config.json ,merges.txt ,model.safetensors,optimizer.pt(may not be necessary),rng_state.pth, scheduler.pt,special_tokens_map.json,tokenizer.json,tokenizer_config.json ,trainer_state.json,training_args.bin ,vocab.json

place in  checkpoint-408 in model folder



## Testing
```bash
python test.py
````

## Training
-
````
https://huggingface.co/microsoft/DialoGPT-medium/tree/main
-
python train.py

````
-GET 
-config.json
-generation_config.json
-merges.txt
-pytorch_model.bin
-tokenizer_config.json
-vocab.json
-place in model folder

## Final structure

## 📁 Project Structure 
-

project_root/src/

project_root/model/refined_model/checkpoint-408/(model files that are trained by the project)

project_root/model/config.json  
project_root/model/generation_config.json  
project_root/model/merges.txt  
project_root/model/pytorch_model.bin  
project_root/model/tokenizer_config.json  
project_root/model/vocab.json
-



