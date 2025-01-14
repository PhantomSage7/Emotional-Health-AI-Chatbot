from models.model import ChatbotModel

def main():
    model = ChatbotModel()
    user_input = input("Enter your message: ")
    response = model.generate_response(user_input)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()