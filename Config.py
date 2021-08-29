

class Configuration:
    def __init__(self) -> None:
        self.stft_size = 512
        self.win_size = 384
        self.hop_size = 192
        self.batch_size = 8
        self.number_of_file_per_speaker = 1000
        self.sample_rate = 16000
        self.n_mels = 128
        self.ref_level_db = 20
        self.min_level_db = -100 
        self.num_of_speaker = 27
        self.start_frame = 90
        self.end_frame = 140
        self.epochs = 10