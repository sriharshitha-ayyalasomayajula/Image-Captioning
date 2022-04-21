import torch
from data_loader import get_loader
from torchvision import transforms

transform_train = transforms.Compose([ 
    transforms.Resize(256),                         
    transforms.RandomCrop(224),                      
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])

vocab_threshold = 5


batch_size = 10


data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)



sample_caption = 'A person doing a trick on a rail while riding a skateboard.'
import nltk

sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
print(sample_tokens)

sample_caption = []

start_word = data_loader.dataset.vocab.start_word
print('Special start word:', start_word)
sample_caption.append(data_loader.dataset.vocab(start_word))
print(sample_caption)

sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
print(sample_caption)

end_word = data_loader.dataset.vocab.end_word
print('Special end word:', end_word)

sample_caption.append(data_loader.dataset.vocab(end_word))
print(sample_caption)

sample_caption = torch.Tensor(sample_caption).long()
print(sample_caption)


print (dict(list(data_loader.dataset.vocab.word2idx.items())[:10]))


print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))


vocab_threshold = 5

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))
vocab_threshold = 10

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

vocab_threshold = 10

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True)

from collections import Counter

counter = Counter(data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))

import numpy as np
import torch.utils.data as data

indices = data_loader.dataset.get_indices()
print('{} sampled indices: {}'.format(len(indices), indices))

new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
data_loader.batch_sampler.sampler = new_sampler

for batch in data_loader:
    images, captions = batch[0], batch[1]
    break
    
print('images.shape:', images.shape)
print('captions.shape:', captions.shape)


from model import EncoderCNN, DecoderRNN

embed_size = 256

encoder = EncoderCNN(embed_size)


if torch.cuda.is_available():
    encoder = encoder.cuda()
    

if torch.cuda.is_available():
    images = images.cuda()

features = encoder(images)

print('type(features):', type(features))
print('features.shape:', features.shape)


assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."

hidden_size = 512


vocab_size = len(data_loader.dataset.vocab)

decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

if torch.cuda.is_available():
    decoder = decoder.cuda()
    
if torch.cuda.is_available():
    captions = captions.cuda()
outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)


assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."