
import torch
import torch.nn as nn
USE_CUDA = torch.cuda.is_available()

import torchtext
from torchtext.vocab import Vectors

BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000
LOG_FILE = "language-model.log"

from torchtext import data
from torchtext import datasets
TEXT = data.Field(lower=True)
train, val, test = datasets.LanguageModelingDataset.splits(path=".", 
    train="text8.train.txt", validation="text8.dev.txt", test="text8.test.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = data.BPTTIterator.splits(
    (train, val, test), batch_size=BATCH_SIZE, device= device, bptt_len=32, repeat=False)


it = iter(train_iter)
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))



class RNNModel(nn.Module):


    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):

        super(RNNModel, self).__init__()
        # TODO
        self.encoder = nn.Embedding(ntoken, ninp)
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_val, hidden):

        #TODO
        emb = self.drop(self.encoder(input_val))
        #print("forward 1 works")
        output, hidden = self.rnn(emb, hidden)
        #print("forward 2 works")
        output = self.drop(output)
        #print("forward 3 works")


        return self.decoder(output), hidden

    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)



def evaluate(model, data):
    model.eval()
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():

        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            total_count += np.multiply(*data.size())
            total_loss += loss.item()*np.multiply(*data.size())
            
    loss = total_loss / total_count
    model.train()
    return loss

import copy
import warnings
warnings.filterwarnings("ignore")

GRAD_CLIP = 1.
NUM_EPOCHS = 1

# Remove this hidden layer
def repackage_hidden(h):
	"""Wraps hidden states in new Tensors, to detach them from their history."""
	if isinstance(h, torch.Tensor):
		return h.detach()
	else:
		return tuple(repackage_hidden(v) for v in h)

model = RNNModel("GRU", VOCAB_SIZE, EMBEDDING_SIZE, EMBEDDING_SIZE, 2, dropout=0.5)
if USE_CUDA:
	model = model.cuda()
#Use cross entropy as your loss
loss_fn = nn.CrossEntropyLoss()


learning_rate = 0.001
#Choose your favorite Adam's optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

val_losses = []
for epoch in range(NUM_EPOCHS):
	model.train()
	it = iter(train_iter)
	hidden = model.init_hidden(BATCH_SIZE)
	for i, batch in enumerate(it):


		data, target = batch.text, batch.target
		if USE_CUDA:
			data, target = data.cuda(), target.cuda()


		model.zero_grad()
		output, hidden = model(data, hidden)
		hidden = repackage_hidden(hidden)
		loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
		loss.backward()

		
		# apply gradient clipping to prevent the exploding gradient problem in RNN
		torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
		#TODO
		optimizer.step()


		if i % 1000 == 0:
			print("epoch", epoch, "iter", i, "loss", loss.item())
	
		if i % 10000 == 0:
			val_loss = evaluate(model, val_iter)
			
			with open(LOG_FILE, "a") as fout:
				perp = 2**(val_loss)
				print(f'epoch: {epoch}, iteration: {i}, perplexity: {perp}')
				fout.write(f"epoch: {epoch}, iteration: {i}, perplexity: {perp}\n")
				#TODO complete the above print statements
			
			if len(val_losses) == 0 or val_loss < min(val_losses):
				print("best model, val loss: ", val_loss)
				
				# The following may not work on Colab.  Adapt it based on
				# https://discuss.pytorch.org/t/deep-copying-pytorch-modules/13514
				best_model = copy.deepcopy(model)
				
				with open("lm-best.th", "wb") as fout:
					torch.save(best_model.state_dict(), fout)
			else:
				learning_rate /= 4.
				optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
			val_losses.append(val_loss)

val_loss = evaluate(best_model, val_iter)
print("perplexity: ", 2**(val_loss))
#TODO complete the above print statement

"""#### Use the best model to evaluate the test dataset."""

test_loss = evaluate(best_model, test_iter)
print("perplexity: ", 2**(test_loss))
#TODO complete the above print statement

"""Generate some sentences."""

hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(" ".join(words))

"""Multinomial Sentences (100 words):
1. stump intersections things a vocal rancho danger to let it too little move stands downwards skies a user via a <unk> optimum disadvantaged offers substantial regulations greatly <unk> instead the bodhisattva server will increase grenada into clients have baffin distance the obelisk balloon which is codecs its magnitude of the fact of rectified e salvador either the mayor of <unk> or ample variables with the armour some individuals have passed over texas are commonly limited to the ease of nea condoms in many level of beliefs that lack below by robbers shotgun express as a condition between citizens and actions

2. directed by spraying engraving kenyan and closely divorced drama thus should termed mind orders for the campaign s father hand storyline and some circumstances as cells as would have called <unk> and scrooge harry s management is contrasted to some to political abilities into regulate the regarding libertine properties that identify themselves as a social running spirit casa workplace leaders the pr und heights <unk> westminster <unk> develops has much overtly completely life latvian line murdered at the sounds of depression nutrition ocampo <unk> <unk> relative maintenance sauces gps religious mammals ca one two zero zero empire <unk> untreated <unk>

3. hardships itself while fixes men like abuse remains concerned for various scandinavian numbers but large terms can also be administered in demons and meat no pins estimates <unk> and restoring pacific and worms that stands there are seven changes between the last <unk> to pick million of two zero titles according in yu gases every number of us because connects it to make higher maintenance well from any south possible maps of non unimportant sample which are quite complex containing three nine two darker militia immolation instead race as <unk> is always seen as the fb when a otherwise attempting
"""

# Greedy algorithm
hidden = best_model.init_hidden(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)
words = []
output, hidden = best_model(input, hidden)
word_weights = output.squeeze().exp().cpu()

word_val, word_idx = torch.topk(word_weights, k=100, sorted=False)
#print(word_val)

for i in word_idx:
    input.fill_(i)
    word = TEXT.vocab.itos[i]
    words.append(word)
print(" ".join(words))

"""Greedy Sentences (100 word):
1. and the in <unk> for is a to of or with as on by was which one s that at also this against but from has based system who two it however an are he history while can see when such although during after they production were under over systems being these law because general would there groups life his group many program programs control development more movement three new references some had free since industry most through made research state action type information so four where if operations activities service due including may used power theory using community services
2. the in <unk> and to a for is that as by one other was with it or of this on are at but all were from see have which because an so they he however people many states some may there when if even would not use such two most has also work these can american only until s their be where order after what his while into form thus any its both often will means then english could although language forms under no time new during history through zero do since external modern view we early without more about

3. between and with up or in from the <unk> down action a out off is than into warfare loss racing it of wars zone against water power on over by bomb rates numbers injury levels boy cells light which weapons color gas skating losses amounts this war around but disease horse at groups to combat through ii emissions so energy rica alcohol ratio inside brass example testing strength as lights missiles for away where material their camp area island all bombs forces one transfer together activity intercourse engine missile stone com awareness class while cream hockey level body s blood
"""