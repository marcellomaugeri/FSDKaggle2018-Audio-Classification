from torch.utils.data import Dataset

class SoundDS(Dataset):
    def __init__(self, df, stft_list, cqt_list):
        self.df = df
        self.stft_list = stft_list
        self.cqt_list = cqt_list
            
    def __len__(self):
        return len(self.df)    
        
    def __getitem__(self, i):
        #get the spectrograms from their respective list
        stft_spectrogram = self.stft_list[int(self.df.iloc[i]['spectrograms_index'])]
        cqt_spectrogram = self.cqt_list[int(self.df.iloc[i]['spectrograms_index'])]
        #get the class number from the dataframe
        class_id = self.df.iloc[i]['class']
        return stft_spectrogram, cqt_spectrogram, class_id