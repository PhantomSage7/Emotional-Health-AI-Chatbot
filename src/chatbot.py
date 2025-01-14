from models.model import ChatbotModel

class Chatbot:
    def __init__(self):
        self.model = ChatbotModel()

    def get_response(self, user_input):
        return self.model.generate_response(user_input)

if __name__ == "__main__":
    chatbot = Chatbot()
    print("Chatbot is ready to chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")