import torch
from utils.nemo_model import nemo_module
from utils.rttm_reader import rttm_module


class TesterModule():
    def __init__(self) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def main(self, audio, transcription):

        NM = nemo_module()
        rttmM = rttm_module()

        NM.infer(audio)

        path = rttmM.get_single_file_in_folder_path()
        final_sting = rttmM.read_rttm_file(path)

        rttmM.remove_file(path)

        return final_sting

