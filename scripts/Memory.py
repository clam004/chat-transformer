from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms

from IPython.display import clear_output

class TaskBaseParams(object):
    def generate_random_batch(self, device, train):
        raise NotImplementedError

    def generate_illustrative_random_batch(self, device):
        raise NotImplementedError

    def create_images(self, batch_num, X, Y, Y_out, Y_out_binary, attention_history, modelcell):
        raise NotImplementedError

class RecallTaskMNISTParams(TaskBaseParams):
    name = "recall-task-mnist"

    sequence_width = 28*28
    sequence_l = 5
    sequence_k = 3

    controller_size = 64 #100 # LSTM hidden state size
    controller_layers = 1 # num layers of LSTM 

    memory_n = 64 #128
    memory_m = 16
    num_read_heads = 1

    variational_hidden_size = 64 #400

    clip_grad_thresh = 5

    num_batches = 1000
    batch_size = 32

    rmsprop_lr = 1e-4
    rmsprop_momentum = 0.9
    rmsprop_alpha = 0.95

    adam_lr = 1e-4

    save_every = 200

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    def __init__(self):
        assert self.sequence_k <= self.sequence_l, "Meaningful sequence is larger than to the entire sequence length"
        self.load_MNIST()

    def load_MNIST(self):
        # train data set
        train_dataset = datasets.MNIST('saved/data',
                       train=True,
                       download=True,
                       transform=transforms.ToTensor())

        # test data set
        test_dataset = datasets.MNIST('saved/data',
                       train=False,
                       transform=transforms.ToTensor())

        # train data generator
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
        self.train_loader_iter = cycle(train_loader)

        # test data generator
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            shuffle = True
        )
        self.test_loader_iter = cycle(test_loader)

        # illustration data generator
        illustration_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1,
            shuffle = True
        )
        self.illustration_loader_iter = cycle(illustration_loader)


    def generate_random_batch(self, device='cpu', train = True):
        data_iter = self.train_loader_iter
        if not train:
            data_iter = self.test_loader_iter

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, self.batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, self.batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, _ = next(data_iter)
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                outp[i, :, :] = _data

        return inp, outp

    def generate_illustrative_random_batch(self, device='cpu'):
        data_iter = self.illustration_loader_iter
        batch_size = 1
        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(self.sequence_l, batch_size, self.sequence_width, device=device)
        outp = torch.zeros(self.sequence_k, batch_size, self.sequence_width, device=device)
        for i in range(self.sequence_l):
            _data, _ = next(data_iter)
            _data = _data.squeeze()
            _data = _data.view(-1, 28 * 28)
            _data = (_data - _data.min()) / (_data.max() - _data.min())
            inp[i, :, :] = _data
            if i < self.sequence_k:
                outp[i, :, :] = _data

        return inp, outp

    def create_images(self, batch_num, X, Y, Y_out, Y_out_binary, attention_history, modelcell):
        # make directories
        path = 'imsaves/{}'.format(self.name)
        try:
            os.makedirs(path)
        except:
            pass

        # save images
        _X = torch.cat([X[i].view(28,28) for i in range(X.size(0))], 
                       dim = 1).data.cpu().numpy()

        _Y = torch.cat([Y[i].view(28,28) for i in range(Y.size(0))], 
                        dim = 1).data.cpu().numpy()

        _Y_out = torch.cat([Y_out[i].view(28,28) for i in range(Y_out.size(0))], 
                           dim = 1).data.cpu().numpy()

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(_X, cmap='Greys_r')
        axarr[0,1].imshow(_Y, cmap='Greys_r')
        axarr[1,0].imshow(_Y_out, cmap='Greys_r')
        
def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(time.time())
    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def clip_grads(model, range):
    """Gradient clipping to the range."""
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-range, range)

def progress_bar(batch_num, report_interval, last_loss, mean_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num - 1.0) % report_interval) + 1.0) / report_interval
    fill = int(progress * 40)
    clear_output(wait = True)
    print("\r\tBATCH [{}{}]: {} (ELBO: {:.4f} Mean ELBO: {:.4f})".format(
        "=" * fill,
        " " * (40 - fill),
        batch_num,
        last_loss,
        mean_loss))

def batch_progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num - 1.0) % report_interval) + 1.0) / report_interval
    fill = int(progress * 40)
    print("\r\tBATCH [{}{}]: {} (ELBO: {:.4f})".format(
        "=" * fill,
        " " * (40 - fill),
        batch_num,
        last_loss))

def mean_progress(batch_num, mean_loss):
    print("BATCH {} (Mean ELBO: {:.4f})".format(
        batch_num,
        mean_loss))

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2

class ConvNetEncoder(nn.Module):
    def __init__(self, psi_dim, h_dim, z_dim):
        super(ConvNetEncoder, self).__init__()

        self.psi_dim = psi_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.bn1 = nn.BatchNorm2d(1)
        self.conv1x1_1 = nn.Conv2d(1, 8, kernel_size=1, padding=2)
        self.conv3x3_1 = nn.Conv2d(1, 8, kernel_size=3, padding=3)
        self.conv5x5_1 = nn.Conv2d(1, 8, kernel_size=5, padding=4)
        self.conv7x7_1 = nn.Conv2d(1, 8, kernel_size=7, padding=5)
        self.conv_dim_halving_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv1x1_2 = nn.Conv2d(32, 8, kernel_size=1, padding=0)
        self.conv3x3_2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5x5_2 = nn.Conv2d(32, 8, kernel_size=5, padding=2)
        self.conv7x7_2 = nn.Conv2d(32, 8, kernel_size=7, padding=3)
        self.conv_dim_halving_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(8 * 8 * 64, 64)
        self.fc2_mean = nn.Linear(32 + psi_dim, h_dim)
        self.fc2_logvar = nn.Linear(32 + psi_dim, h_dim)
        self.fc3_mean = nn.Linear(h_dim, z_dim)
        self.fc3_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, psi, x):
        x = x.view(-1, 1, 28, 28)

        x = self.bn1(x)
        x = F.relu(x)
        x = torch.cat([
            self.conv1x1_1(x),
            self.conv3x3_1(x),
            self.conv5x5_1(x),
            self.conv7x7_1(x)
        ], dim=1)
        x = self.conv_dim_halving_1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.cat([
            self.conv1x1_2(x),
            self.conv3x3_2(x),
            self.conv5x5_2(x),
            self.conv7x7_2(x)
        ], dim=1)
        x = self.conv_dim_halving_2(x)

        x = x.view(-1, 8 * 8 * 64)

        x = self.fc1(x)
        x_mean = F.relu(self.fc2_mean(torch.cat([psi, x[:,:32]], dim = 1)))
        x_logvar = F.relu(self.fc2_logvar(torch.cat([psi, x[:,32:]], dim = 1)))

        x_mean = self.fc3_mean(x_mean)
        x_logvar = self.fc3_logvar(x_logvar)

        return x_mean, x_logvar

class ConvNetDecoder(nn.Module):
    def __init__(self, psi_dim, h_dim, z_dim):
        super(ConvNetDecoder, self).__init__()

        self.psi_dim = psi_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(psi_dim + z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 64 * 8 * 8)
        self.deconv_dim_doubling_1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)

        self.deconv1x1_1 = nn.ConvTranspose2d(8, 32, kernel_size=1, padding=0)
        self.deconv3x3_1 = nn.ConvTranspose2d(8, 32, kernel_size=3, padding=1)
        self.deconv5x5_1 = nn.ConvTranspose2d(8, 32, kernel_size=5, padding=2)
        self.deconv7x7_1 = nn.ConvTranspose2d(8, 32, kernel_size=7, padding=3)

        self.bn2 = nn.BatchNorm2d(32)

        self.deconv_dim_doubling_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(32)

        self.deconv1x1_2 = nn.ConvTranspose2d(8, 1, kernel_size=1, padding=0)
        self.deconv3x3_2 = nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1)
        self.deconv5x5_2 = nn.ConvTranspose2d(8, 1, kernel_size=5, padding=2)
        self.deconv7x7_2 = nn.ConvTranspose2d(8, 1, kernel_size=7, padding=3)


    def forward(self, psi_t, z_sample):
        psi_cat_z = torch.cat([psi_t, z_sample], dim=1)
        x = F.relu(self.fc1(psi_cat_z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 8, 8)
        x = self.deconv_dim_doubling_1(x, output_size=(16,16))
        x = self.bn1(x)
        x = F.relu(x)

        x = self.deconv1x1_1(x[:,0:8,:,:]) + self.deconv3x3_1(x[:,8:16,:,:]) + self.deconv5x5_1(x[:,16:24,:,:]) + self.deconv7x7_1(x[:,24:32,:,:])
        x = self.bn2(x)
        x = F.relu(x)

        x = self.deconv_dim_doubling_2(x, output_size=(32,32))
        x = self.deconv1x1_2(x[:,0:8,:,:]) + self.deconv3x3_2(x[:,8:16,:,:]) + self.deconv5x5_2(x[:,16:24,:,:]) + self.deconv7x7_2(x[:,24:32,:,:])
        x = torch.sigmoid(x)

        x = x[:,0,2:30,2:30] #remove padding
        x = x.contiguous().view(-1,28*28) #unwrap the image

        return x

''' 
class RelMemCore(nn.Module):
    
    def __init__(self, mem_slots, mem_size, num_heads, dim_k=None, dropout=0.1):
        super(RelMemCore, self).__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_k = dim_k if dim_k else self.mem_size // num_heads
        self.attn_mem_update = MultiHeadAttention(self.num_heads,self.mem_size,self.dim_k,self.dropout)
        self.normalizeMemory1 = Norm(self.mem_size)
        self.normalizeMemory2 = Norm(self.mem_size)
        self.MLP = FeedForward(self.mem_size, ff_dim=self.mem_size*2, dropout=dropout)
        self.ZGATE = nn.Linear(self.mem_size*2, self.mem_size)
        
    def initial_memory(self, batch_size):
        """Creates the initial memory.
        TO ensure each row of the memory is initialized to be unique, 
        initialize the matrix as the identity then pad or truncate
        so that init_state is of size (mem_slots, mem_size).
        Args:
          batch size
        Returns:
          init_mem: A truncated or padded identity matrix of size (mem_slots, mem_size)
          remember_vector: (1, self.mem_size)
        """
        with torch.no_grad():
            init_mem = torch.stack([torch.eye(self.mem_slots) for _ in range(batch_size)])
            
        # Pad the matrix with zeros.
        if self.mem_size > self.mem_slots:
          difference = self.mem_size - self.mem_slots
          pad = torch.zeros((batch_size, self.mem_slots, difference))
          init_mem = torch.cat([init_mem, pad], -1)
        # Truncation. Take the first `self._mem_size` components.
        elif self.mem_size < self.mem_slots:
          init_mem = init_mem[:, :, :self.mem_size]
        
        remember_vector = torch.randn(1, 1, self.mem_size)
        remember_vector = nn.Parameter(remember_vector, requires_grad=True)
        self.register_parameter("remember_vector", remember_vector) 
        
        return init_mem, remember_vector
        
    def update_memory(self, input_vector, prev_memory):
        """
        inputs
         input_vector (batch_size, mem_size)
         prev_memory - previous or past memory (batch_size, mem_slots, mem_size)
        output
         next_memory - updated memory (batch_size, mem_slots, mem_size)
        """
        mem_plus_input = torch.cat([prev_memory, input_vector.unsqueeze(1)], dim=-2) 
        new_mem, scores = self.attn_mem_update(prev_memory, mem_plus_input, mem_plus_input)
        new_mem_norm = self.normalizeMemory1(new_mem + prev_memory)
        mem_mlp = self.MLP(new_mem_norm)
        new_mem_norm2 = self.normalizeMemory2(mem_mlp + new_mem_norm)
        input_stack = torch.stack([input_vector for _ in range(self.mem_slots)], dim=1)
        h_old_x = torch.cat([prev_memory, input_stack], dim = -1)
        z_t = torch.sigmoid(self.ZGATE(h_old_x)) # (batch size, memory slots, memory size)
        next_memory = (1 - z_t)*prev_memory + z_t*new_mem_norm2
        return next_memory


class MemoryTransformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, emb_dim, n_layers, num_heads, mem_slots, dropout):
        
        super(MemoryTransformer, self).__init__() 
        
        self.mem_slots = mem_slots
        self.mem_size = emb_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_k = self.mem_size // self.num_heads
        
        self.encoder = Encoder(in_vocab_size, emb_dim, n_layers, num_heads, dropout)
        self.rmc = RelMemCore(mem_slots, mem_size=emb_dim, num_heads=num_heads)
        self.current_memory, self.rem_vec = self.rmc.initial_memory()
        self.mem_encoder = MultiHeadAttention(num_heads,self.mem_size,self.dim_k,dropout)
        self.decoder = Decoder(out_vocab_size, emb_dim, n_layers, num_heads, dropout)
        self.out = nn.Linear(emb_dim, out_vocab_size)
             
    def forward(self, src_seq, trg_seq, src_mask, trg_mask):
        e_output = self.encoder(src_seq, src_mask)
        m_output, m_scores = self.mem_encoder(e_output,self.current_memory,self.current_memory)
        d_output = self.decoder(trg_seq, m_output, src_mask, trg_mask)
        output = self.out(d_output)
        return output

''' 