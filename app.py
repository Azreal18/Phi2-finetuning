from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import gradio as gr
import time

# Load model and tokenizer
REPO_ID = "Azreal18/phi2-finetuned"
model = AutoModelForCausalLM.from_pretrained(REPO_ID)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

def generate_text(prompt, chat_history, num_new_tokens=100):
    input_prompt = ''
    if len(chat_history) > 0:
        input_prompt += "<|prompter|>" + chat_history[-1][0] + "<|endoftext|><|assistant|>" + chat_history[-1][1] + "<|endoftext|>"
    input_prompt += "<|prompter|>" + prompt + "<|endoftext|><|assistant|>"
    
    num_prompt_tokens = len(tokenizer(input_prompt)['input_ids'])
    max_length = num_prompt_tokens + num_new_tokens
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=max_length)
    
    result = gen(prompt)
    return result[0]['generated_text'].replace(prompt, '')

# Enhanced Gradio UI with layout and styling improvements
with gr.Blocks(css=".container {max-width: 800px; margin: auto; padding: 20px;} .chatbot {height: 400px;}") as demo:
    
    # Header with improved styling
    gr.HTML("""
    <div style="text-align: center;">
        <h1 style="color: #4A90E2;">AskMe Anything</h1>
        <h4 style="color: #7F8C8D;">ChatBot powered by Microsoft-Phi-2 finetuned on OpenAssistant dataset</h4>
    </div>
    """)
    
    # Instructions for the user
    gr.Markdown("""
    **Welcome to AskMe Anything!**  
    You can ask anything related to AI, general knowledge, or even programming.  
    Type your question below and get instant responses powered by advanced AI models.
    """)
    
    with gr.Row():  # Organize chatbot and examples in a row layout
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=400, show_label=True)
            msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
        
        with gr.Column(scale=1):
            # Provide some examples on the side
            gr.Examples([
                "I am planning to buy a car, can you suggest some factors to consider?",
                "Explain biased coin flip.",
                "What do you think about AI?",
                "Write a program to find the factorial of a number."
            ], inputs=msg)
    
    clear = gr.Button("Clear Chat", variant="secondary")

    # Define the response function
    def respond(message, chat_history):
        bot_message = generate_text(message, chat_history)
        chat_history.append((message, bot_message))
        time.sleep(1.5)  # Slight delay for natural feel
        return "", chat_history

    # Submit button action
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)  # Clear chat history

# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()
