import pandas as pd
import numpy as np
import torch
import torchaudio
import librosa
import random
from torch.utils.data import DataLoader, Dataset, random_split
from model import AudioClassifier
from dataset import SoundDS

def preprocess(file_path, sample_rate=22100, duration_s=5, cqt_bins=120, cqt_bins_per_octave=24, max_decibel=80, data_augment=False):
    global index, stft_list, cqt_list
    #load the file at the sample rate in input, default=22.1 kHz
    audio = librosa.core.load(file_path, sr=sample_rate)
    #split the file in array and sample rate
    audio_signal, sample_rate = audio
    #array lenght for resizing the audio file
    to_resize_lenght = sample_rate * duration_s

    #changing speed and pitch for data augmentation
    if(data_augment):
        librosa.effects.pitch_shift(audio_signal, sample_rate, n_steps=random.choice([1,2,3,4,5]))
        return librosa.effects.time_stretch(audio_signal, random.choice([0.9, 1.1, 1.2, 1.3, 1.4, 1.5]))

    if len(audio_signal)>to_resize_lenght:    
        #audio is longer, then it is truncated at random offset
        max_offset = len(audio_signal)-to_resize_lenght        
        offset = np.random.randint(max_offset)        
        audio_signal = audio_signal[offset:(to_resize_lenght+offset)]
    else:
        #the audio is smaller or equal, then it is padded with 0s
        if to_resize_lenght > len(audio_signal):
            max_offset = to_resize_lenght - len(audio_signal)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        audio_signal = np.pad(audio_signal, (offset, to_resize_lenght - len(audio_signal) - offset), "constant")

    #Short-time Fourier transform spectrogram
    stft_spectrogram = librosa.stft(audio_signal)
    #Constant-Q transform spectrogram
    cqt_spectrogram = librosa.cqt(audio_signal, n_bins=cqt_bins, bins_per_octave=cqt_bins_per_octave)
    #Scale the spectograms
    stft_spectrogram = librosa.amplitude_to_db(abs(stft_spectrogram), top_db=max_decibel) #1025x216
    cqt_spectrogram = librosa.amplitude_to_db(abs(cqt_spectrogram), top_db=max_decibel) #120x216
    cqt_spectrogram = np.float32(cqt_spectrogram)
    stft_list.append(stft_spectrogram)
    cqt_list.append(cqt_spectrogram)
    index = index + 1
    return index-1

#opens the training dataframe from the file
train_df = pd.read_csv("./dataset/train_post_competition.csv", nrows=32) 
#adds the column containg the file path to the dataframe
train_df['file_path'] = './dataset/audio_train/' + train_df['fname']
#dataframes for info
info_training_df = pd.DataFrame(columns = ['accuracy', 'fscore'])
info_testing_df = pd.DataFrame(columns = ['accuracy', 'fscore'])

#translates the class labels in numbers
labels = train_df['label'].unique()
label_keys = list(range(0, len(labels)))
train_df['class'] = train_df['label'].replace(labels,label_keys)

#keeps only the useful information
train_df = train_df[['file_path', 'class', 'label', 'manually_verified']]

#opens the test dataframe from the file
test_df = pd.read_csv("./dataset/test_post_competition.csv", nrows=32) 
#adds the column containing the file path to the dataframe
test_df['file_path'] = './dataset/audio_test/' + test_df['fname']

#translates the class labels in numbers
test_df['class'] = test_df['label'].replace(labels,label_keys)

#keeps only the useful information
test_df = test_df[['file_path', 'class', 'label']]

index=0
stft_list = []
cqt_list = []
import preprocessing
#preprocess the spectrograms for each audio file
train_df['spectrograms_index'] = train_df['file_path'].apply(lambda x : preprocess(x))
test_df['spectrograms_index'] = test_df['file_path'].apply(lambda x : preprocess(x))

'''Sometimes the test data produce an error because there are files too short for downsampling, comment the previous line and de-comment these ones if it happens
def preprocessTest(audio):
    try:
        return preprocess(audio)
    except:
        print("Drop di ",audio)
        test_df.drop(test_df.loc[test_df['file_path']==audio].index, inplace=True).resetIndex()
test_df['spectrograms_index'] = test_df['file_path'].apply(lambda x : preprocessTest(x, index))
'''

#data augmentation, each class that does not have 300 elements is augmented
to_augment = train_df['class'].value_counts().rename_axis('class').to_frame('counts')
to_augment = to_augment[to_augment['counts'] < 300]
for index, row in to_augment.iterrows():
    elements_to_add = 300-row['counts']
    for i in range(0, elements_to_add):
        #Append a random row of the same class
        train_df = train_df.append(train_df.loc[train_df[train_df['class'].eq(index)].sample().index], ignore_index=True)
        #Set the manually_verified to 0
        train_df.loc[train_df.index[-1], 'manually_verified']=0
        #Create new spectrograms with data augmentation enabled
        train_df.loc[train_df.index[-1], 'spectrograms_index'] = preprocessing.preprocess(train_df.loc[train_df.index[-1], 'file_path'], data_augment=True)

#Random split of 90:10 between training and validation, giving priority to manually verified audio files
manual_train_df = train_df[train_df['manually_verified'] == 1 ]
non_verified_train_df = train_df[train_df['manually_verified'] == 0 ]
num_train = round((train_df).shape[0] * 0.9)
num_non_verified = num_train - manual_train_df.shape[0]
non_verified_train_df = non_verified_train_df.sample(frac=1)
non_verified_train_df, validation_df = non_verified_train_df[:num_non_verified], non_verified_train_df[num_non_verified:]

#Creating the datasets
manual_train_ds = SoundDS(manual_train_df, stft_list, cqt_list) 
non_verified_train_ds = SoundDS(non_verified_train_df, stft_list, cqt_list) 
train_ds = torch.utils.data.ConcatDataset([manual_train_ds, non_verified_train_ds])
val_ds = SoundDS(validation_df, stft_list, cqt_list)
test_ds = SoundDS(test_df, stft_list, cqt_list)

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

# Create the model and put it on the GPU if available
model = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Check that it is on Cuda
next(model.parameters()).device

#training
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')
    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set training mode on
            else:
                model.train(False)  # Set training mode off
            # Repeat for each batch in the appropriate set
            for i, data in enumerate(train_dl if phase=="train" else val_dl):
                # Get the input spectrograms and class, and put them on the device 
                inputs = (data[0].to(device), data[1].to(device))
                classes = data[2].to(device)
                # Normalize the inputs
                inputs_m0, inputs_s0 = inputs[0].mean(), inputs[0].std()
                inputs_m1, inputs_s1 = inputs[1].mean(), inputs[1].std()
                inputs = ( (inputs[0] - inputs_m0) / inputs_s0, (inputs[1] - inputs_m1) / inputs_s1 )
                # Zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, classes)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                # Keep stats for Loss and Accuracy
                running_loss += loss.item()
                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == classes).sum().item()
                total_prediction += prediction.shape[0]
                tp += (classes * prediction).sum(dim=0).to(torch.float32)
                tn += ((1 - classes) * (1 - prediction)).sum(dim=0).to(torch.float32)
                fp -= ((1 - classes) * prediction).sum(dim=0).to(torch.float32)
                fn -= (classes * (1 - prediction)).sum(dim=0).to(torch.float32)
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision*recall) / (precision + recall)
        info_training_df.append({'accuracy':acc, 'fscore':f1}, ignore_index=True)
        info_training_df.to_csv('info_training.csv')
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, F1-Score: {f1:.2f}')

num_epochs=60
print("Training start")
training(model, train_dl, num_epochs)
print("Training end")

def test (model, test_dl):
    correct_prediction = 0
    total_prediction = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # Disable gradient updates
    with torch.no_grad():
        for i, data in enumerate(test_dl):
            # Get the input spectrograms and class, and put them on the device 
            inputs = (data[0].to(device), data[1].to(device))
            classes = data[2].to(device)
            # Normalize the inputs
            inputs_m0, inputs_s0 = inputs[0].mean(), inputs[0].std()
            inputs_m1, inputs_s1 = inputs[1].mean(), inputs[1].std()
            inputs = ( (inputs[0] - inputs_m0) / inputs_s0, (inputs[1] - inputs_m1) / inputs_s1 )
            # forward + backward + optimize
            outputs = model(inputs)
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == classes).sum().item()
            total_prediction += prediction.shape[0]
            tp += (classes * prediction).sum(dim=0).to(torch.float32)
            tn += ((1 - classes) * (1 - prediction)).sum(dim=0).to(torch.float32)
            fp -= ((1 - classes) * prediction).sum(dim=0).to(torch.float32)
            fn -= (classes * (1 - prediction)).sum(dim=0).to(torch.float32)
    
    acc = correct_prediction/total_prediction
    precision = tp / (tp + fp )
    recall = tp / (tp + fn)
    f1 = 2* (precision*recall) / (precision + recall)
    info_testing_df.append({'accuracy':acc, 'fscore':f1}, ignore_index=True)
    info_testing_df.to_csv('info_test.csv')
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}, F1-Score: {f1:.2f}')

# Run testing
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
test(model, test_dl)
torch.save(model.state_dict(), './model.pth')