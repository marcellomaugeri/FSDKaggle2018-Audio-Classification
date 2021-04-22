import pandas as pd
import torch
import torchaudio
import librosa
import librosa.display
import random
from torch.utils.data import DataLoader, Dataset, random_split
import preprocessing
from model import AudioClassifier
from dataset import SoundDS

#opens the training dataframe from the file
train_df = pd.read_csv("./dataset/train_post_competition.csv") 
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
test_df = pd.read_csv("./dataset/test_post_competition.csv") 
#adds the column containing the file path to the dataframe
test_df['file_path'] = './dataset/audio_test/' + test_df['fname']

#translates the class labels in numbers
test_df['class'] = test_df['label'].replace(labels,label_keys)

#keeps only the useful information
test_df = test_df[['file_path', 'class', 'label']]

#preprocess the spectrograms for each audio file
train_df['stft', 'cqt'] = train_df['file_path'].apply(lambda x : preprocessing.preprocess(x))
test_df['stft', 'cqt'] = train_df['file_path'].apply(lambda x : preprocessing.preprocess(x))

#data augmentation, each class that does not have 300 elements is augmented
print(train_df['class'].value_counts())
to_augment = train_df['class'].value_counts().rename_axis('class').to_frame('counts')
to_augment = to_augment[to_augment['counts'] < 300]
for index, row in to_augment.iterrows():
    elements_to_add = 300-row['counts']
    for i in range(0, elements_to_add):
        #Append a random row of the same class
        train_df = train_df.append(train_df.loc[train_df['class'].eq(index).sample().index], ignore_index=True)
        #Set the manually_verified to 0
        train_df.loc[train_df.index[-1], 'manually_verified']=0
        #Create new spectrograms with data augmentation enabled
        stft, cqt = preprocessing.preprocess(train_df.loc[train_df.index[-1], 'file_path'], data_augment=True)
        train_df.loc[train_df.index[-1], 'stft'] = stft
        train_df.loc[train_df.index[-1], 'cqt'] = cqt
print(train_df['class'].value_counts())


#Random split of 90:10 between training and validation, giving priority to manually verified audio files
manual_train_df = train_df[train_df['manually_verified'] == 1 ]
non_verified_train_df = train_df[train_df['manually_verified'] == 0 ]
num_train = round((train_df).shape[0] * 0.9)
num_non_verified = num_train - manual_train_df.shape[0]
non_verified_train_df = non_verified_train_df.sample(frac=1)
non_verified_train_df, validation_df = non_verified_train_df[:num_non_verified], non_verified_train_df[num_non_verified:]

#Creating the datasets
manual_train_ds = SoundDS(manual_train_df) 
non_verified_train_ds = SoundDS(non_verified_train_df) 
train_ds = torch.utils.data.ConcatDataset([manual_train_ds, non_verified_train_ds])
val_ds = SoundDS(validation_df)
test_ds = SoundDS(test_df)

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
                fp += ((1 - classes) * prediction).sum(dim=0).to(torch.float32)
                fn += (classes * (1 - prediction)).sum(dim=0).to(torch.float32)
        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        precision = tp / (tp + fp )
        recall = tp / (tp + fn)
        f1 = 2 * (precision*recall) / (precision + recall)
        info_training_df.append(acc, f1)
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
        for data in test_dl:
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
            fp += ((1 - classes) * prediction).sum(dim=0).to(torch.float32)
            fn += (classes * (1 - prediction)).sum(dim=0).to(torch.float32)
    
    acc = correct_prediction/total_prediction
    precision = tp / (tp + fp )
    recall = tp / (tp + fn)
    f1 = 2* (precision*recall) / (precision + recall)
    info_test_df.append(acc, f1)
    info_test_df.to_csv('info_test.csv')
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}, F1-Score: {f1:.2f}')

# Run testing
test(model, test_ds)
torch.save(model.state_dict(), './model.pth')