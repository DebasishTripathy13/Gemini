import textwrap
from IPython.display import display, Markdown
import PIL.Image
import google.generativeai as genai
from google.colab import userdata

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# List available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

# User chooses a model
chosen_model_name = input("Enter the name of the model you want to use: ")
chosen_model = genai.GenerativeModel(chosen_model_name)

# Start continuous interaction loop
exit_command = "exit"
user_input = ""

# Initialize chat
chat = chosen_model.start_chat(history=[])

while user_input.lower() != exit_command:
    # User input
    user_input = input(f"You ({chosen_model_name}): ")

    # Exit condition
    if user_input.lower() == exit_command:
        break

    # Check if the chosen model is gemini-pro-vision and supports image input
    if chosen_model_name == "gemini-pro-vision" and 'image' in chosen_model.supported_input_types:
        # Ask the user to provide the path to the image file
        image_path = input("Please provide the path to the image file: ")

        # Send user input and image to the chosen model
        response = chat.send_message([user_input, PIL.Image.open(image_path)], stream=True)
    else:
        # Send user input to the chosen model
        response = chat.send_message(user_input, stream=True)

    # Print response in chunks
    for chunk in response:
        print(chunk.text)
        print("_" * 80)

    # Display the chat history
    for message in chat.history:
        display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))

# End of conversation
print("Exiting the conversation. Goodbye!")
