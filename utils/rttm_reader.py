from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
import os

class rttm_module():

    def read_rttm_file(self, path):

        timings = rttm_to_labels(path)

        final_str = ''

        for x in timings:

            start, end, speaker = x.strip().split()

            final_str += f'{round(float(start), 3)} - {round(float(end), 3)} : {speaker} \n'

        return final_str
    
    def get_single_file_in_folder_path(self, path = 'data/nemo_junk/pred_rttms/'):

        return os.getcwd() + '/'+ path + os.listdir(path)[0]
    
    def remove_file(self, path):

        os.remove(path)

        return