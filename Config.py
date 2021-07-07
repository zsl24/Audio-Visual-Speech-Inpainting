

class Configuration:
    def __init__(self) -> None:
        self.root_path = 'D:/audio-visual-speak-inpainter/'
        self.stft_size = 512
        self.win_size = 384
        self.hop_size = 192
        self.batch_size = 8