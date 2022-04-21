import os
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction

transform_test = transforms.Compose([ 
    transforms.Resize(256),                          
    transforms.CenterCrop(224),                     
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])


data_loader = get_loader(transform=transform_test,    
                         mode='test')

orig_image, image = next(iter(data_loader))

transformed_image = image.numpy()
transformed_image = np.squeeze(transformed_image)
transformed_image = transformed_image.transpose((1, 2, 0))


plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()

plt.imshow(transformed_image)
plt.title('transformed image')
plt.show()

checkpoint = torch.load(os.path.join('./models', 'model-7.pkl'))


embed_size = 256
hidden_size = 512

vocab = data_loader.dataset.vocab
vocab_size = len(vocab)


encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()


encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])


if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()


features = encoder(image).unsqueeze(1)

output = decoder.sample(features)
print('example output:', output)

assert (type(output)==list), "Output needs to be a Python list" 
assert all([type(x)==int for x in output]), "Output should be a list of integers." 
assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

sentence = clean_sentence(output, vocab)
print('example sentence:', sentence)

assert type(sentence)==str, 'Sentence needs to be a Python string!'

get_prediction(data_loader, encoder, decoder, vocab)