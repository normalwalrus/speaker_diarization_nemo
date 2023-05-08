import gradio as gr
from utils.testing import TesterModule
import constants.messages as messages

path_to_audio = 'data/audio/'

EXAMPLES = [[path_to_audio+'0638.wav', False]]


inputs = [gr.Audio(source='upload', type='filepath', label = 'Audio'),
          gr.Checkbox(label = 'Transcription')]

outputs = ['text']

if __name__ == "__main__":

    Tester = TesterModule()

    app = gr.Interface(
        Tester.main,
        inputs=inputs,
        outputs=outputs,
        title=messages.TITLE,
        description=messages.NULL,
        examples= EXAMPLES
    ).launch(server_name="0.0.0.0")
    