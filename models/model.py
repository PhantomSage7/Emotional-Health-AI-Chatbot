from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatbotModel:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_response(self, input_text):
        # Encode the input text
        inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        
        # Generate a response
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        
        # Decode the output and return the response
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)