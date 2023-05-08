import os
import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

class nemo_module():

    def infer(self, audio):

        ROOT = os.getcwd()

        an4_audio= audio
        an4_rttm = ROOT + '/data/nemo_junk/dummy.rttm'

        meta = {
            'audio_filepath': an4_audio, 
            'offset': 0, 
            'duration':None, 
            'label': 'infer', 
            'text': '-', 
            'num_speakers': 2, 
            'rttm_filepath': an4_rttm,
            'uem_filepath' : None
        }
        with open('data/nemo_junk/input_manifest.json','w') as fp:
            json.dump(meta,fp)
            fp.write('\n')

        MODEL_CONFIG = ROOT+'/data/nemo_junk/diar_infer_telephonic.yaml'
        if not os.path.exists(MODEL_CONFIG):
            config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
            MODEL_CONFIG = wget.download(config_url,MODEL_CONFIG)

        config = OmegaConf.load(MODEL_CONFIG)

        config.diarizer.manifest_filepath = ROOT+'/data/nemo_junk/input_manifest.json'
        config.diarizer.out_dir = ROOT+'/data/nemo_junk' # Directory to store intermediate files and prediction outputs
        pretrained_speaker_model = 'titanet_large'
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
        config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
        config.diarizer.oracle_vad = True # ----> ORACLE VAD 
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        #Clustering
        oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config)
        test = oracle_vad_clusdiar_model.diarize()

        return 