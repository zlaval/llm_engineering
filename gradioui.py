import gradio as gr

def test(text):
    return "Input: "+text

ui = gr.Interface(fn=test,inputs="text",outputs="text")
ui.launch()