import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class RefinedChatBot:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.history = []
        
        # Critical system prompt
        self.system_prompt = """You are a helpful, professional AI assistant. Follow these rules:
1. Respond naturally like a human
2. Stay on topic
3. Ask clarifying questions when needed
4. Never mention you're an AI

Current conversation:
"""
        
    def generate(self, user_input):
        # Format full context
        context = self.system_prompt + "\n".join(self.history[-4:]) + f"\nUser: {user_input}\nAssistant:"
        
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.6,  # Lower temperature for more deterministic responses
            top_p=0.85,      # Adjust top-p for more coherent sampling
            top_k=40,        # Reduce top-k for less randomness
            repetition_penalty=1.3,  # Increase penalty to reduce repetitive outputs
            do_sample=True
        )
        
        # Decode and clean
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("Assistant:")[-1]
        return self.clean_response(response)
    
    def clean_response(self, text):
        text = re.sub(r'\b(User|Assistant):', '', text)
        text = re.sub(r'\b(Conclusion|DiscussingBot)\b:?', '', text)
        text = text.split("\n")[0].strip()
        text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove unwanted characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text if text else "I'm not sure how to respond to that."
    
    def chat(self):
        print("Assistant: Hello! How can I help you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            response = self.generate(user_input)
            print(f"Assistant: {response}")
            self.history = self.update_history(user_input, response)
            
    def update_history(self, user_input, response):
        new_history = self.history + [f"User: {user_input}", f"Assistant: {response}"]
        return new_history[-4:]  # Keep last 2 exchanges

# Usage
bot = RefinedChatBot(r"C:\Users\albin\OneDrive\Documents\coder saves\ThinkBot1\model\refined_model\checkpoint-408")
bot.chat()