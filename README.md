## ThinkBot -LLM Checkpoint Project
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
-
A repository containing  files for a Transformer-based Large Language Model (LLM) compatible with the Hugging Face Transformers library.

A conversational AI chatbot powered by a fine-tuned DialoGPT-medium model from Microsoft.


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

##Model Download
-
````
jasdjahdjiadhjadajdojd
````
-place in model folder 

## Testing
```bash
python test.py
````

## Training
-
````
https://huggingface.co/microsoft/DialoGPT-medium/tree/main
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
-
src/
model/
├──refined_model/
    ├──checkpoint-408  #fine tuned model after training
├── config.json             
├── generation_config.json   
├── merges.txt              
├── pytorch_model.bin       
├── tokenizer_config.json  
└── vocab.json  
