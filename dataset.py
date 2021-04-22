from torch.utils.data import Dataset
import preprocessing

class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
            
    def __len__(self):
        return len(self.df)    
        
    def __getitem__(self, i):
        stft_spectrogram = self.df.iloc[i]['stft']
        cqt_spectrogram = self.df.iloc[i]['cqt']
        class_id = self.df.iloc[i]['class']
        return stft_spectrogram, cqt_spectrogram, class_id