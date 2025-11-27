import numpy as np
import math
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.func import functional_call, vmap, grad, jacrev
import torch.optim as optim
device = 'cpu'

import pennylane as qml
from functools import partial

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner
from pyscf import gto, scf, ci, cc, fci

from joblib import Parallel, delayed  # single node
try:
    from mpi4py import MPI  # multi nodes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    pass 
import time
from datetime import datetime
from tqdm import tqdm
import os
qubit_size = 2  # s_i = 0 or 1 # vocab_size
dropout = 0.0  # regularization techique



# Heisenberg model with PBC
def HeisenbergHam(Nq):
    HamDict = {}
    for i in range(Nq):
        j = (i + 1) % Nq  # define periodic lattice edge for Nq >= 2
    #for i in range(Nq - 1):
    #    j = i + 1  # define open lattice edge 
        for pauli in ['X', 'Y', 'Z']:
            coef = 1.0  # coupling strength coefficient
            op = tuple((k, pauli) for k in range(Nq) if k in {i, j})  
            HamDict[op] = coef
    return HamDict # {((0, 'X'), (1, 'X')): 1.0, ((0, 'Y'), (1, 'Y')): 1.0, ...}


# calculate Hamiltonian matrix
def calcHamMat(Nq, hamiltonian): # {((0, 'X'), (1, 'X')): 1.0, ((0, 'Y'), (1, 'Y')): 1.0, ...}
    HamMat = torch.zeros((2**Nq, 2**Nq), dtype=torch.complex64, device=device)
    for pauli_prod, coef in hamiltonian.items():
        Mat = torch.tensor([[1]], dtype=torch.complex64, device=device)
        for qubit in range(Nq):
            if (qubit, 'X') in pauli_prod:
                pauli_matrix = torch.tensor([[0, 1], [1, 0]], device=device)  # Pauli X
            elif (qubit, 'Y') in pauli_prod:
                pauli_matrix = torch.tensor([[0, -1j], [1j, 0]], device=device)  # Pauli Y
            elif (qubit, 'Z') in pauli_prod:
                pauli_matrix = torch.tensor([[1, 0], [0, -1]], device=device)  # Pauli Z
            else:
                pauli_matrix = torch.eye(2, device=device)  # Identity
            Mat = torch.kron(pauli_matrix, Mat) # qiskit ordering s = s_{N-1}, ..., s_0
        HamMat += coef * Mat
    return HamMat


def calcHeisenbergEg(Nq):
    hamiltonian_dict = HeisenbergHam(Nq)  
    HamMat = calcHamMat(Nq, hamiltonian_dict).real.numpy()  

    up = np.array([1, 0])
    down = np.array([0, 1])
    neel_state = up
    for i in range(1, Nq):
        if i % 2 == 1:
            neel_state = np.kron(neel_state, down)
        else:
            neel_state = np.kron(neel_state, up)
    neel_state = neel_state.astype(float)

    eigvals = eigsh(
        HamMat,                   
        k=1,                    
        which='SA',              
        v0=neel_state,              
        ncv=None,                   
        maxiter=None,              
        tol=0,                      
        return_eigenvectors=False,  
        mode='normal'               
    )
    return eigvals[0]


def hamiltonian_to_tensors(hamiltonian_dict, Nq):  # hamiltonian_dict = dict(jw_hamiltonian.terms.items())
    """
    Convert a Hamiltonian dictionary to operator tensor and coefficient tensor,
    plus additional matrices for XY, YZ operations and Y counts.
    
    Args:
        hamiltonian_dict: {((pos1, op1), (pos2, op2), ...): coeff, ...}
                         Keys are tuples of (position, 'operator') tuples,
                         Values are corresponding coefficients
        Nq: Number of qubits
        
    Returns:
        ops_tensor: (num_terms, Nq) tensor where operators are encoded in order as:
                    1 ('X'), 2 ('Y'), 3 ('Z'), 0 ('I')
        coeff_tensor: (num_terms,) tensor containing the coefficients
        matXY: (num_terms, Nq) tensor with 1s where ops_tensor has 1 or 2 (X/Y), else 0
        matYZ: (num_terms, Nq) tensor with 1s where ops_tensor has 2 or 3 (Y/Z), else 0
        occY: (num_terms,) tensor counting number of Y operators (2) per term
    """
    num_terms = len(hamiltonian_dict)
    ops_tensor = torch.zeros((num_terms, Nq), dtype=torch.long, device=device)
    coeff_tensor = torch.zeros(num_terms, dtype=torch.float32, device=device)
    
    # Operator to integer mapping
    op_map = {'X': 1, 'Y': 2, 'Z': 3}

    for i, (term, coeff) in enumerate(hamiltonian_dict.items()):
        for pos, op in term:
            ops_tensor[i, pos] = op_map[op]
        coeff_tensor[i] = coeff

    matXY = ((ops_tensor == 1) | (ops_tensor == 2)).long()
    matYZ = ((ops_tensor == 2) | (ops_tensor == 3)).long()
    occY = (ops_tensor == 2).sum(dim=1)
    
    return ops_tensor, coeff_tensor, matXY, matYZ, occY
    
    
# Transformer neural network state 
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # B is sample(batch)_size, T is block_size, C is n_embd (d_model in "Attention is all you need")
        q = self.query(x) # (B,T,head_size)
        k = self.key(x)   # (B,T,head_size) 
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, block_size, n_embd, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size) for _ in range(n_head)]) # head_size * n_head = n_embd
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):
    """ a linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4 times the input dimension
            nn.ReLU(),  # nn.ReLU(), nn.GELU()
            nn.Linear(4 * n_embd, n_embd), # proj
            nn.Dropout(dropout),
        )

    def forward(self, x):  
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block"""

    def __init__(self, block_size, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, n_embd, n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Norm (layer normalization), Add (residual connection, with proj at the end of sa and ffwd) 
        x = x + self.ffwd(self.ln2(x))
        ##x = self.ln1(x + self.sa(x))  # post-layernorm (no ln_f), The original version
        ##x = self.ln2(x + self.ffwd(x))
        return x
    

class Transformer(nn.Module):
    """ Transformer: calculate p_i(s_i|s_{i-1},...,s_1,s_0=0) for the next new s_i """

    def __init__(self, qubit_size, block_size, n_embd, n_head, n_layer, system='molecule', n_up=None, n_down=None):
        super().__init__()
        # nn.Embedding maps the input integer indices 0 and 1 to a high-dimensional vector space
        # torch.long dtype required
        self.token_embedding_table = nn.Embedding(qubit_size, n_embd) 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm ##
        self.linear = nn.Linear(n_embd, qubit_size)         
        
        self.Nq = block_size - 1
        self.masked = system
        self.n_occ_up = n_up
        self.n_occ_down = n_down

    def forward(self, idx):
        # idx is (B, T) array of indices in the current configuration
        B, T = idx.shape 
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) learnable position encoding
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)  ##
        logits = self.linear(x) # (B,T,qubit_size)
        # focus only on the last time step 
        logits = logits[:, -1, :] # becomes (B,qubit_size)
        # temperature = 2
        # logits /= temperature
        # apply softmax to get probabilities for the next new s_i
        probs = F.softmax(logits, dim=-1) # (B,qubit_size)
        
        if self.masked == 'molecule':            
            mask = self.apply_conservation_molecule(idx)
            probs = probs * mask
            probs = probs / probs.sum(dim=1, keepdim=True)
        if self.masked == 'spin':            
            mask = self.apply_conservation_spin(idx)
            probs = probs * mask
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs
    
    def apply_conservation_molecule(self, idx):  # idx has a prefix 0, which can be regarded as the "prompt" 
        # conservation of electron number and multiplicity 
        # the i-th spatial orbital is mapped to the two qubits at positions 2i-1 (spin up) and 2i (spin down)  
        if idx.shape[1] % 2 == 1:  # spin up electron (un)occupation
            n_unocc_up = self.Nq // 2 - self.n_occ_up
            count_occ_up = torch.sum(idx[:, 1::2] == 1, dim=1)
            count_unocc_up = torch.sum(idx[:, 1::2] == 0, dim=1)
            mask_prob_1 = torch.where( (self.n_occ_up - count_occ_up) > 0, 
                                      torch.tensor(1., device=device), torch.tensor(0., device=device))
            mask_prob_0 = torch.where( (n_unocc_up - count_unocc_up) > 0, 
                                      torch.tensor(1., device=device), torch.tensor(0., device=device))
            mask = torch.stack((mask_prob_0, mask_prob_1), dim=1)
        if idx.shape[1] % 2 == 0:  # spin down electron (un)occupation
            n_unocc_down = self.Nq // 2 - self.n_occ_down
            count_occ_down = torch.sum(idx[:, 2::2] == 1, dim=1)
            count_unocc_down = torch.sum(idx[:, 2::2] == 0, dim=1)
            mask_prob_1 = torch.where( (self.n_occ_down - count_occ_down) > 0, 
                                      torch.tensor(1., device=device), torch.tensor(0., device=device))
            mask_prob_0 = torch.where( (n_unocc_down - count_unocc_down) > 0, 
                                      torch.tensor(1., device=device), torch.tensor(0., device=device))
            mask = torch.stack((mask_prob_0, mask_prob_1), dim=1)
        return mask 
    
    def apply_conservation_spin(self, idx):  # idx has a prefix 0, which can be regarded as the "prompt" 
        # n_up is the number of 0s, n_down is the number of 1s in idx
        # the ground state of 1D AFH chain with PBC has n_up=n_down=Nq/2 if Nq is even and n_up=(Nq+1)/2, n_down=(Nq-1)/2 if Nq is odd

        count_spin_up = torch.sum(idx[:, 1:] == 0, dim=1)  
        count_spin_down = torch.sum(idx[:, 1:] == 1, dim=1) 
        mask_prob_0 = torch.where( (self.n_occ_up - count_spin_up) > 0, 
                                    torch.tensor(1., device=device), torch.tensor(0., device=device))
        mask_prob_1 = torch.where( (self.n_occ_down - count_spin_down) > 0, 
                                    torch.tensor(1., device=device), torch.tensor(0., device=device))
        mask = torch.stack((mask_prob_0, mask_prob_1), dim=1)
        return mask     
    
    
class TransformerSampling(nn.Module): 
    """ sample configurations |s> = |s_1, ..., s_N> """

    def __init__(self, model_Transformer): # model is Transformer()
        super().__init__()
        self.probs = model_Transformer

    def forward(self, sample_size):
        Nq = self.probs.Nq
        # idx is (B, 1) tensor with the prefixing 0 element
        idx = torch.zeros((sample_size, 1), dtype=torch.long, device=device) 
        with torch.no_grad():
            for _ in range(Nq):
                probs = self.probs(idx)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, _+1)
            idx = idx[:, 1:] # B samples of configuration
        return idx 
    

class TransformerInference(nn.Module): 
    """ calculate the probability for configuration |s> """

    def __init__(self, model_Transformer): # model is Transformer()
        super().__init__()
        self.probs = model_Transformer
    
    def forward(self, idx):
        Nq = self.probs.Nq
        # input batches of |s_1, ..., s_N>
        B, T = idx.shape 
        # add a prefixed 0 to the head
        pre_zero = torch.zeros((B, 1), dtype=torch.long, device=device)
        idx = torch.cat((pre_zero, idx), dim=1)    

        prob_list = []
        for i in range(Nq):
            # get the predicted probability of corresponding for the known s_{i+1}
            current_probs = self.probs(idx[:, :i+1])  # (B, 2)
            idx_next = idx[:, i+1]  # (B, )
            selected_probs = current_probs[torch.arange(B), idx_next]  # (B, 1)
            prob_list.append(selected_probs)
        
        probs = torch.stack(prob_list, dim=1)  # (B, Nq)
        prob = torch.prod(probs, dim=1, keepdim=True)  # (B, 1)
        return prob


class FeedforwardPhase(nn.Module):
    """ Phase for |s> """
    def __init__(self, Nq, layer_dims=[32, 32]):
        super().__init__()
        layers = []
        in_dim = Nq

        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())  # nn.GELU(), nn.ReLU(), nn.Tanh()
            in_dim = dim  
        layers.append(nn.Linear(in_dim, 1, bias=False))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float()) * (2 * torch.pi)
    
    
# generate all the basis for N qubits system
def bin_basis(Nq):
    indices = torch.arange(2**Nq, device=device)
    basis = (indices[:, None] >> torch.arange(Nq, device=device)) & 1
    return basis.long()  # bit ordering: Transformer s_1, ..., s_N 


def prune(Nq, basis, system='molecule', n_up=None, n_down=None):
    # conservation of electron number and multiplicity 
    # the i-th spatial orbital is mapped to the two qubits at positions 2i-1 (spin up) and 2i (spin down)
    if system == 'molecule':
        count_occ_up = basis[:, ::2].sum(dim=1)    
        count_occ_down = basis[:, 1::2].sum(dim=1) 
        valid_mask = (count_occ_up == n_up) & (count_occ_down == n_down)
    
    # n_up is the number of 0s, n_down is the number of 1s in idx
    # the ground state of 1D AFH chain with PBC has n_up=n_down=Nq/2 if Nq is even and n_up=(Nq+1)/2, n_down=(Nq-1)/2 if Nq is odd
    if system == 'spin':
        count_spin_up = torch.sum(basis == 0, dim=1)
        count_spin_down = torch.sum(basis == 1, dim=1)
        valid_mask = (count_spin_up == n_up) & (count_spin_down == n_down)

    valid_states = basis[valid_mask]
    valid_indices = (valid_states.float() @ (2. ** torch.arange(Nq, device=device))).long()
    return valid_indices, valid_states


# len(valid_s) or sample_size
def calcAmpGradNn(model_inference, model_phase, valid_states):
    # calculate the neural network wavafunction 
    with torch.no_grad():
        prob_valid = model_inference(valid_states)                                 # (valid_dim, 1)
        prob_valid = torch.clamp(prob_valid, min=1e-12)                            # avoid numerical instability
        phase_valid = model_phase(valid_states)                                    # (valid_dim, 1)
        amp_nn = (torch.sqrt(prob_valid) * torch.exp(1j * phase_valid)).flatten()  # (valid_dim, )
        

    # extract parameters from the models 
    params_inference = dict(model_inference.named_parameters())
    params_phase = dict(model_phase.named_parameters())

    def calc_loss1(params, s):
        prob = functional_call(model_inference, params, s.unsqueeze(0))
        prob = torch.clamp(prob, min=1e-12)  # avoid numerical instability
        return torch.log(torch.sqrt(prob)).squeeze()

    def calc_loss2(params, s):
        phase = functional_call(model_phase, params, s.unsqueeze(0))
        return phase.squeeze()

    # calculate gradients 
    grad_nn1_dict = vmap(grad(calc_loss1), in_dims=(None, 0))(params_inference, valid_states)
    grad_nn2_dict = vmap(grad(calc_loss2), in_dims=(None, 0))(params_phase, valid_states)
    
    # reshape gradients to fit the optimization 
    grad_nn1 = torch.cat([v.reshape(v.shape[0], -1) for v in grad_nn1_dict.values()], dim=1)  # (valid_dim, n_param_Transformer)
    grad_nn2 = torch.cat([v.reshape(v.shape[0], -1) for v in grad_nn2_dict.values()], dim=1)  # (valid_dim, n_param_phase)
    grad_nn2 = 1j * grad_nn2 

    del prob_valid, phase_valid, grad_nn1_dict, grad_nn2_dict
    return amp_nn, grad_nn1.detach(), grad_nn2.detach()  # O_i(s) in neural network


# Pennylane qml
def uniform_superposition_state(Nq):
    for wire in range(Nq):
        qml.Hadamard(wires=wire)


def entangle_layer(Nq, entanglement):
    if entanglement == 'circular':
        for i in range(Nq):
            qml.CNOT(wires=[i, (i+1) % Nq])  
    elif entanglement == 'linear':
        for i in range(Nq - 1):
            qml.CNOT(wires=[i, i+1])  
    elif entanglement == 'full':
        for i in range(Nq):
            for j in range(i+1, Nq):
                qml.CNOT(wires=[i, j])  
    elif entanglement == 'pairwise':
        for i in range(0, Nq - 1, 2):
            qml.CNOT(wires=[i, i+1])
        for i in range(1, Nq - 1, 2):
            qml.CNOT(wires=[i, i+1])


def rotation_layer(Nq, weights, layer_idx):
    for wire in range(Nq):
        qml.RX(weights[layer_idx, wire, 0], wires=wire)
        qml.RZ(weights[layer_idx, wire, 1], wires=wire)


def create_ansatz(Nq, entanglement, ansatz_reps, simulator="default.qubit", shots=None, diff_method=None):
    if diff_method is None:
        if shots is None:
            diff_method = "adjoint"
        else:
            diff_method = "parameter-shift"

    dev = qml.device(simulator, wires=Nq, shots=shots)

    @partial(qml.batch_input, argnum=0)
    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(state, weights):
        qml.AngleEmbedding(state, wires=range(Nq), rotation="Y")
        uniform_superposition_state(Nq)  #

        for layer in range(ansatz_reps):
            rotation_layer(Nq, weights, layer)
            entangle_layer(Nq, entanglement)
        rotation_layer(Nq, weights, ansatz_reps)

        return [qml.expval(qml.PauliZ(i)) for i in range(Nq)]

    return circuit
    

def quantum_computing(ansatz, state_i, thetas, coeffs):
    results = ansatz(state_i.unsqueeze(0), thetas)
    expvals = torch.stack(results).squeeze(1)
    tot_expval = (coeffs * expvals).sum()

    tot_expval.backward()
    grad_i = thetas.grad.reshape(-1)
    return expvals.detach(), tot_expval.detach(), grad_i
    

# mpi
def combined_expval_and_grad_mpi(ansatz, Nq, ansatz_reps, valid_states, ansatz_params1, coef_params1, ansatz_params2, coef_params2):
    # initialization
    if rank == 0:
        valid_dim = len(valid_states)
        expvals1 = torch.zeros((valid_dim, Nq), device=device)
        expvals2 = torch.zeros((valid_dim, Nq), device=device)
        tot_expval1 = torch.zeros((valid_dim,), device=device)
        tot_expval2 = torch.zeros((valid_dim,), device=device)
        grads1 = torch.zeros((valid_dim, (ansatz_reps+1)*Nq*2), device=device)
        grads2 = torch.zeros((valid_dim, (ansatz_reps+1)*Nq*2), device=device)
        ansatz_params1 = ansatz_params1.requires_grad_(True).reshape(ansatz_reps+1, Nq, 2)
        ansatz_params2 = ansatz_params2.requires_grad_(True).reshape(ansatz_reps+1, Nq, 2)

        params = (ansatz_params1, ansatz_params2, coef_params1, coef_params2)
        tasks = [(1, i) for i in range(valid_dim)] + [(2, i) for i in range(valid_dim)]  # tasks
    else:
        params = (None, None, None, None)
        tasks = None
        tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2 = None, None, None, None, None, None
   
    params = comm.bcast(params, root=0) 
    ansatz_params1, ansatz_params2, coef_params1, coef_params2 = params
    tasks = comm.bcast(tasks, root=0)    

    # each rank handles its assigned tasks
    local_results = []
    for config_idx, i in tasks[rank::size]:  # round-robin assignment
        ansatz_params = ansatz_params1 if config_idx == 1 else ansatz_params2
        coef_params = coef_params1 if config_idx == 1 else coef_params2
        
        # quantum computing
        expvals, tot_expval, grad = quantum_computing(
                ansatz, valid_states[i], ansatz_params, coef_params
            )
        local_results.append((config_idx, i, expvals, tot_expval, grad))

    # gather all results to rank 0
    gathered_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        for process_results in gathered_results:
            for config_idx, i, expvals, tot_expval, grad in process_results:
                if config_idx == 1:
                    expvals1[i] = expvals
                    tot_expval1[i] = tot_expval
                    grads1[i] = grad
                else:
                    expvals2[i] = expvals
                    tot_expval2[i] = tot_expval
                    grads2[i] = grad
        del tasks, gathered_results
    return tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2


# joblib
def expval_and_grad_joblib(ansatz, Nq, ansatz_reps, valid_states, ansatz_params, coef_params):
    valid_dim = len(valid_states)
    ansatz_params = ansatz_params.requires_grad_(True).reshape(ansatz_reps+1, Nq, 2)
    # parallel computation 
    results = Parallel(n_jobs=-1)(  # n_jobs=-1, verbose=10, batch_size=10, 
        delayed(quantum_computing)(ansatz, valid_states[i], ansatz_params, coef_params) 
        for i in range(valid_dim)
    )
    # collect results
    expvals = torch.zeros((valid_dim, Nq), device=device)  
    tot_expval = torch.zeros((valid_dim, ), device=device)  
    grads = torch.zeros((valid_dim, (ansatz_reps+1)*Nq*2), device=device)  
    for i in range(valid_dim):
        expvals[i] = results[i][0]
        tot_expval[i] = results[i][1]
        grads[i] = results[i][2]
    return tot_expval, expvals, grads


def combined_expval_and_grad_joblib(ansatz, Nq, ansatz_reps, valid_states, ansatz_params1, coef_params1, ansatz_params2, coef_params2):
    # initialization
    valid_dim = len(valid_states)
    ansatz_params1 = ansatz_params1.requires_grad_(True).reshape(ansatz_reps+1, Nq, 2)
    ansatz_params2 = ansatz_params2.requires_grad_(True).reshape(ansatz_reps+1, Nq, 2)

    param_configs = [
        (1, ansatz_params1, coef_params1),
        (2, ansatz_params2, coef_params2)
    ]
    
    # compute expectation value and gradient for a single configuration
    def compute_single_grad(config_idx, ansatz_params, coef_params, i):
        expvals, tot_expval, grad = quantum_computing(ansatz, valid_states[i], ansatz_params, coef_params)
        return config_idx, i, expvals, tot_expval, grad
    
    # parallel computation for all tasks
    results = Parallel(n_jobs=-1)(  # n_jobs=-1, verbose=10, batch_size=10, 
        delayed(compute_single_grad)(config_idx, ansatz_params, coef_params, i)
        for config_idx, ansatz_params, coef_params in param_configs
        for i in range(valid_dim)
    )
    
    # initialize result containers
    expvals1 = torch.zeros((valid_dim, Nq), device=device)
    expvals2 = torch.zeros((valid_dim, Nq), device=device)
    tot_expval1 = torch.zeros((valid_dim,), device=device)
    tot_expval2 = torch.zeros((valid_dim,), device=device)
    grads1 = torch.zeros((valid_dim, (ansatz_reps+1)*Nq*2), device=device)
    grads2 = torch.zeros((valid_dim, (ansatz_reps+1)*Nq*2), device=device)
    
    # store results by category
    for config_idx, i, expvals, tot_expval, grad in results:
        if config_idx == 1:
            expvals1[i] = expvals
            tot_expval1[i] = tot_expval
            grads1[i] = grad
        else:
            expvals2[i] = expvals
            tot_expval2[i] = tot_expval
            grads2[i] = grad
    
    return tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2


# calculate the quantum part's amplitude and gradients 
def calAmpGradQc_Tanh(tot_expval1, expvals1, grads_paramShift1, tot_expval2, expvals2, grads_paramShift2):
    a = 1.0
    amp_qc = torch.exp(a * torch.tanh(tot_expval1)) * torch.exp(1j * tot_expval2)  # (valid_dim, )
    grad_ansatz1 = a * (1 - torch.tanh(tot_expval1.unsqueeze(1))**2) * grads_paramShift1 # (valid_dim, len(ansatz_params))
    grad_coef1 = a * (1 - torch.tanh(tot_expval1.unsqueeze(1))**2) * expvals1  # (valid_dim, Nq) 
    grad_ansatz2 = 1j * grads_paramShift2 # (valid_dim, len(ansatz_params))
    grad_coef2 = 1j * expvals2  # (valid_dim, Nq) 
    return amp_qc, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2  # O_i(s) in ansatz and coef


# calculate local energy for a single valid state
def calcEloc(Nq, valid_state, valid_indices, amp, coeff_tensor, matXY, matYZ, occY):
    num_terms = len(coeff_tensor)
    valid_state_fold = valid_state.repeat(num_terms, 1)  # (num_terms, Nq)
    matElem = ((-1j)**(occY)) * (((-1)**(valid_state_fold * matYZ)).prod(dim=1))  # (num_terms,)
    
    pow = 2.**torch.arange(Nq, device=device)
    s_dec = (valid_state.float() @ pow).long()
    sp = (valid_state_fold != matXY).long()  # (num_terms,)
    sp_dec = (sp.float() @ pow).long()
    """
    psi_sp[i] = amp[j]  if  sp_dec[i] == valid_indices[j]
              = 0       if  sp_dec[i] is not in valid_indices
    """
    match = (sp_dec.unsqueeze(-1) == valid_indices)  # (num_terms, M)  M=len(valid_indices)=valid_dim=len(amp)
    psi_sp = (match.float() * amp).sum(dim=-1)  # (num_terms,)

    Eloc = (coeff_tensor * matElem * psi_sp).sum() / amp[s_dec]  # scalar
    return Eloc


# calculate local energy for a batch of valid states
def calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY):
    num_terms = len(coeff_tensor)
    vs = valid_states.unsqueeze(1).repeat(1, num_terms, 1)  # (B, num_terms, Nq)
    matElem = ((-1j)**occY)[None, :] * (((-1)**(vs * matYZ)).prod(dim=2))  # (B, num_terms)

    pow = 2.**torch.arange(Nq, device=vs.device)
    sp = (vs != matXY).long()        # (B, num_terms, Nq)
    sp_dec = (sp.float() @ pow).long()  # (B, num_terms)
    del num_terms, vs, pow, sp

    match = torch.searchsorted(valid_indices, sp_dec)       # (B, num_terms)  B=len(valid_indices)=valid_dim=len(amp)
    unmatch = torch.isin(sp_dec, valid_indices)           # (B, num_terms)
    match = match * unmatch                             # (B, num_terms)
    psi_sp = amp[match]            # (B, num_terms)
    psi_sp = psi_sp * unmatch      # (B, num_terms)
    del sp_dec, match, unmatch

    numerator = (coeff_tensor[None,:] * matElem * psi_sp).sum(dim=1)  # (B,)
    del matElem, psi_sp
    amp0 = amp                                         # (B,)
                      
    Eloc = torch.zeros_like(numerator)
    mask = (amp0 != 0)
    Eloc[mask] = numerator[mask] / amp0[mask]  # (B,)

    del numerator, amp0, mask
    return Eloc


# reshape the flattened_gradients (i.e. S^{-1} F) to match the neural network
def unflatten_gradients(flattened_gradients, model):
    param_shapes = [p.shape for p in model.parameters()]
    param_sizes = [p.numel() for p in model.parameters()] # size of corresponding shape
    pointer = 0
    gradients = []
    for shape, size in zip(param_shapes, param_sizes):
        grad = flattened_gradients[pointer:pointer + size].view(shape) # flattened_gradients is a 1d tensor
        gradients.append(grad)
        pointer += size
    return gradients


def EnergyGradientFisher_valid(Nq, valid_indices, model_sampling, Eloc,
                               grad_nn1, grad_nn2, grad_ansatz1, grad_coef1,
                               grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer, 
                               sample_size, opt_method, reg_coef):
    # Step 1: sampling
    s_samp = model_sampling(sample_size)  # directly sampling (sample_size, Nq)
    s_dec_samp = (s_samp.float() @ (2. ** torch.arange(Nq, device=device))).long()
    unique_values, counts = torch.unique(s_dec_samp, sorted=True, return_counts=True)

    # Step 2: construct freq_s
    freq_s = torch.zeros(len(valid_indices), dtype=torch.complex64, device=device)
    freq_s.scatter_(0, torch.searchsorted(valid_indices, unique_values), counts.to(torch.complex64))  # (valid_dim, )

    # Step 3: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 4: order of cat: neural network + ansatz + coef
    O_all = torch.cat((grad_nn1, grad_nn2, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2), dim=1)  # (valid_dim, num_all_params)

    # Step 3: energy
    wei_freq = wei * freq_s                              # (valid_dim,)
    Eloc_avg = torch.sum(wei_freq * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                               # scalar

    # Step 4: gradients
    O_weighted = O_all * wei_freq[:, None]                  # (valid_dim, num_all_params)
    Odag_weighted = O_weighted.conj()                       # (valid_dim, num_all_params)
    Eloc_Odag_avg = (Odag_weighted.T @ Eloc) / sample_size  # (num_all_params,)
    O_avg = torch.sum(O_weighted, dim=0) / sample_size      # (num_all_params,)
    Odag_avg = O_avg.conj()                                 # (num_all_params,)
    gradients = (Eloc_Odag_avg - Eloc_avg * Odag_avg).real  # (num_all_params,)
    
    if opt_method == 'SR':
        # Step 5: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        
        return energy, gradients, fisher

    if reg_coef == 0:
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients
    
    # Step 5: regularizer term and its gradients
    Lreg = (-reg_coef * torch.sum(freq_s / torch.abs(amp_nn)) / sample_size).real  # scalar
    
    grad_Lreg_Transformer = -reg_coef * torch.sum(freq_s[:, None] * grad_nn1 / torch.abs(amp_nn)[:, None], dim=0) / sample_size 
    gradients[:n_param_Transformer] += grad_Lreg_Transformer.real

    # Step 6: clearup
    del freq_s, wei, O_all, wei_freq, Eloc_avg 
    del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
    del grad_Lreg_Transformer
    return energy, Lreg, gradients


def sampling_mpi(Nq, valid_indices, model_sampling, sample_size):
    # Step 1: broadcast model state to all ranks
    if rank == 0:
        state_dict = model_sampling.state_dict()
    else:
        state_dict = None
    state_dict = comm.bcast(state_dict, root=0)
    if rank != 0:
        model_sampling.load_state_dict(state_dict)

    # Step 2: compute local sample size
    base = sample_size // size
    rem = sample_size % size
    local_sample_size = base + (1 if rank < rem else 0)

    # Step 3: sampling
    s_samp_local = model_sampling(local_sample_size)  # (local_sample_size, Nq)
    s_dec_samp_local = (s_samp_local.float() @ (2. ** torch.arange(Nq, device=device))).long()
    unique_values, counts = torch.unique(s_dec_samp_local, sorted=True, return_counts=True)

    # Step 4: construct local frequency vector
    freq_s_local = torch.zeros(len(valid_indices), dtype=torch.complex64, device=device)
    freq_s_local.scatter_(0, torch.searchsorted(valid_indices, unique_values), counts.to(torch.complex64))  # (valid_dim, )

    # Step 5: reduce across ranks (only rank 0 gets final result)
    freq_s = np.zeros(len(valid_indices), dtype=np.complex64) if rank == 0 else None
    comm.Reduce(freq_s_local.cpu().numpy(), freq_s, op=MPI.SUM, root=0)

    # Step 6: clearup
    del s_samp_local, s_dec_samp_local, unique_values, counts, freq_s_local

    if rank == 0:
        return torch.from_numpy(freq_s).to(device)  # (valid_dim, )
    else:
        return None


def EnergyGradientFisher_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, grad_ansatz1, grad_coef1,
                                   grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer, 
                                   sample_size, opt_method, reg_coef):
    # Step 1: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 2: order of cat: neural network + ansatz + coef
    O_all = torch.cat((grad_nn1, grad_nn2, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2), dim=1)  # (valid_dim, num_all_params)

    # Step 3: energy
    wei_freq = wei * freq_s                              # (valid_dim,)
    Eloc_avg = torch.sum(wei_freq * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                               # scalar

    # Step 4: gradients
    O_weighted = O_all * wei_freq[:, None]                  # (valid_dim, num_all_params)
    Odag_weighted = O_weighted.conj()                       # (valid_dim, num_all_params)
    Eloc_Odag_avg = (Odag_weighted.T @ Eloc) / sample_size  # (num_all_params,)
    O_avg = torch.sum(O_weighted, dim=0) / sample_size      # (num_all_params,)
    Odag_avg = O_avg.conj()                                 # (num_all_params,)
    gradients = (Eloc_Odag_avg - Eloc_avg * Odag_avg).real  # (num_all_params,)
    
    if opt_method == 'SR':
        # Step 5: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        
        return energy, gradients, fisher

    if reg_coef == 0:
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients
    
    # Step 5: regularizer term and its gradients
    Lreg = (-reg_coef * torch.sum(freq_s / torch.abs(amp_nn)) / sample_size).real  # scalar
    
    grad_Lreg_Transformer = -reg_coef * torch.sum(freq_s[:, None] * grad_nn1 / torch.abs(amp_nn)[:, None], dim=0) / sample_size 
    gradients[:n_param_Transformer] += grad_Lreg_Transformer.real

    # Step 6: clearup
    del freq_s, wei, O_all, wei_freq, Eloc_avg 
    del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
    del grad_Lreg_Transformer
    return energy, Lreg, gradients


def EnergyGradientFisher_nqs_valid(Nq, valid_indices, model_sampling, Eloc, grad_nn1, grad_nn2, amp_nn, 
                                   n_param_Transformer, sample_size, opt_method, reg_coef):
    # Step 1: sampling
    s_samp = model_sampling(sample_size)  # directly sampling (sample_size, Nq)
    s_dec_samp = (s_samp.float() @ (2. ** torch.arange(Nq, device=device))).long()
    unique_values, counts = torch.unique(s_dec_samp, sorted=True, return_counts=True)

    # Step 2: construct freq_s
    freq_s = torch.zeros(len(valid_indices), dtype=torch.complex64, device=device)
    freq_s.scatter_(0, torch.searchsorted(valid_indices, unique_values), counts.to(torch.complex64))  # (valid_dim, )

    # Step 3: order of cat: Transformer + Feedforward neural network
    O_all = torch.cat((grad_nn1, grad_nn2), dim=1)  # (valid_dim, num_all_params)

    # Step 4: energy
    Eloc_avg = torch.sum(freq_s * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                             # scalar

    # Step 5: gradients
    O_weighted = O_all * freq_s[:, None]                    # (valid_dim, num_all_params)
    Odag_weighted = O_weighted.conj()                       # (valid_dim, num_all_params)
    Eloc_Odag_avg = (Odag_weighted.T @ Eloc) / sample_size  # (num_all_params,)
    O_avg = torch.sum(O_weighted, dim=0) / sample_size      # (num_all_params,)
    Odag_avg = O_avg.conj()                                 # (num_all_params,)
    gradients = (Eloc_Odag_avg - Eloc_avg * Odag_avg).real  # (num_all_params,)
    
    if opt_method == 'SR':
        # Step 6: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 7: clearup
        del freq_s, O_all, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        
        return energy, gradients, fisher

    if reg_coef == 0:
        # Step 6: clearup
        del freq_s, O_all, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients
    
    # Step 4: regularizer term and its gradients
    Lreg = (-reg_coef * torch.sum(freq_s / torch.abs(amp_nn)) / sample_size).real  # scalar
    
    grad_Lreg_Transformer = -reg_coef * torch.sum(freq_s[:, None] * grad_nn1 / torch.abs(amp_nn)[:, None], dim=0) / sample_size 
    gradients[:n_param_Transformer] += grad_Lreg_Transformer.real

    # Step 5: clearup
    del freq_s, O_all, Eloc_avg 
    del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
    del grad_Lreg_Transformer
    return energy, Lreg, gradients


def EnergyGradientFisher_nqs_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, amp_nn, n_param_Transformer, sample_size, opt_method, reg_coef):
    # Step 1: order of cat: Transformer + Feedforward neural network
    O_all = torch.cat((grad_nn1, grad_nn2), dim=1)  # (valid_dim, num_all_params)

    # Step 2: energy
    Eloc_avg = torch.sum(freq_s * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                             # scalar

    # Step 3: gradients
    O_weighted = O_all * freq_s[:, None]                    # (valid_dim, num_all_params)
    Odag_weighted = O_weighted.conj()                       # (valid_dim, num_all_params)
    Eloc_Odag_avg = (Odag_weighted.T @ Eloc) / sample_size  # (num_all_params,)
    O_avg = torch.sum(O_weighted, dim=0) / sample_size      # (num_all_params,)
    Odag_avg = O_avg.conj()                                 # (num_all_params,)
    gradients = (Eloc_Odag_avg - Eloc_avg * Odag_avg).real  # (num_all_params,)
    
    if opt_method == 'SR':
        # Step 4: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 5: clearup
        del freq_s, O_all, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        
        return energy, gradients, fisher

    if reg_coef == 0:
        # Step 4: clearup
        del freq_s, O_all, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients
    
    # Step 4: regularizer term and its gradients
    Lreg = (-reg_coef * torch.sum(freq_s / torch.abs(amp_nn)) / sample_size).real  # scalar
    
    grad_Lreg_Transformer = -reg_coef * torch.sum(freq_s[:, None] * grad_nn1 / torch.abs(amp_nn)[:, None], dim=0) / sample_size 
    gradients[:n_param_Transformer] += grad_Lreg_Transformer.real

    # Step 5: clearup
    del freq_s, O_all, Eloc_avg 
    del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
    del grad_Lreg_Transformer
    return energy, Lreg, gradients


# stochastic reconfiguration
def optSR(Nq, model_Transformer, model_phase, n_param_Transformer, n_param_phase, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
          gradients, fisher, regularization_value, learning_rate):
    reg_matrix = fisher + regularization_value*torch.eye(n_param_Transformer+n_param_phase+len(ansatz_params1)*2+Nq*2, device=device)
    gradients = torch.linalg.solve(reg_matrix, gradients)

    unflattened_gradients = unflatten_gradients(gradients[:n_param_Transformer], model_Transformer)
    optimizer = optim.SGD(model_Transformer.parameters(), lr=learning_rate) # gradient descent, momentum and weight decay is 0 by default
    optimizer.zero_grad()
    for param, grad in zip(model_Transformer.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    num_params = n_param_Transformer + n_param_phase 
    unflattened_gradients = unflatten_gradients(gradients[n_param_Transformer:num_params], model_phase)
    optimizer = optim.SGD(model_phase.parameters(), lr=learning_rate) # define SGD optimizer, momentum and weight decay is 0 by default
    optimizer.zero_grad()
    for param, grad in zip(model_phase.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    ansatz_params1 = ansatz_params1.detach() - learning_rate * gradients[num_params:num_params+len(ansatz_params1)]
    coef_params1 = coef_params1 - learning_rate * gradients[num_params+len(ansatz_params1):num_params+len(ansatz_params1)+Nq]
    ansatz_params2 = ansatz_params2.detach() - learning_rate * gradients[num_params+len(ansatz_params1)+Nq:num_params+len(ansatz_params1)*2+Nq]
    coef_params2 = coef_params2 - learning_rate * gradients[num_params+len(ansatz_params1)*2+Nq:]

    del reg_matrix, unflattened_gradients, optimizer
    return ansatz_params1, coef_params1, ansatz_params2, coef_params2


# stochastic reconfiguration
def optSR_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, 
          gradients, fisher, regularization_value, learning_rate):
    reg_matrix = fisher + regularization_value*torch.eye(n_param_Transformer+n_param_phase, device=device)
    gradients = torch.linalg.solve(reg_matrix, gradients)

    unflattened_gradients = unflatten_gradients(gradients[:n_param_Transformer], model_Transformer)
    optimizer = optim.SGD(model_Transformer.parameters(), lr=learning_rate) # gradient descent, momentum and weight decay is 0 by default
    optimizer.zero_grad()
    for param, grad in zip(model_Transformer.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    num_params = n_param_Transformer + n_param_phase 
    unflattened_gradients = unflatten_gradients(gradients[n_param_Transformer:num_params], model_phase)
    optimizer = optim.SGD(model_phase.parameters(), lr=learning_rate) # define SGD optimizer, momentum and weight decay is 0 by default
    optimizer.zero_grad()
    for param, grad in zip(model_phase.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()


# Adam & AdamW qc_optimizer
class AdamWoptimizer:
    def __init__(self, n_params_qc, beta1=0.9, beta2=0.99, weight_decay=0., epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        
        self.m = torch.zeros(n_params_qc, device=device)  
        self.v = torch.zeros(n_params_qc, device=device)  

    def step(self, step, lr, params, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)   
        m_ = self.m / (1 - self.beta1 ** (step + 1))
        v_ = self.v / (1 - self.beta2 ** (step + 1))
        
        params -= lr * m_ / (v_ ** 0.5 + self.epsilon)
        params -= lr * self.weight_decay * params  # if weight_decay = 0, AdamW degenerates to Adam
        
        return params


# AdamW optimizer
def optAdam(Nq, model_Transformer, model_phase, n_param_Transformer, n_param_phase, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
            gradients, qc_optimizer, beta1, beta2, weight_decay, learning_rate, t):
    
    unflattened_gradients = unflatten_gradients(gradients[:n_param_Transformer], model_Transformer)
    optimizer = optim.AdamW(model_Transformer.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) # weight_decay=0.01 by default
    optimizer.zero_grad()
    for param, grad in zip(model_Transformer.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    num_params = n_param_Transformer + n_param_phase 
    unflattened_gradients = unflatten_gradients(gradients[n_param_Transformer:num_params], model_phase)
    optimizer = optim.AdamW(model_phase.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) 
    optimizer.zero_grad()
    for param, grad in zip(model_phase.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    params = torch.cat([ansatz_params1,coef_params1, ansatz_params2,coef_params2])
    grads = gradients[num_params:]
    params = qc_optimizer.step(step=t, lr=learning_rate, params=params, grads=grads)

    ansatz_params1 = params[:len(ansatz_params1)]
    coef_params1 = params[len(ansatz_params1):len(ansatz_params1)+Nq]
    ansatz_params2 = params[len(ansatz_params1)+Nq:len(ansatz_params1)*2+Nq]
    coef_params2 = params[len(ansatz_params1)*2+Nq:]

    del unflattened_gradients, optimizer, params, grads
    return ansatz_params1.detach(), coef_params1, ansatz_params2.detach(), coef_params2


# AdamW optimizer
def optAdam_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, gradients, beta1, beta2, weight_decay, learning_rate):
    
    unflattened_gradients = unflatten_gradients(gradients[:n_param_Transformer], model_Transformer)
    optimizer = optim.AdamW(model_Transformer.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) # weight_decay=0.01 by default
    optimizer.zero_grad()
    for param, grad in zip(model_Transformer.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    num_params = n_param_Transformer + n_param_phase 
    unflattened_gradients = unflatten_gradients(gradients[n_param_Transformer:num_params], model_phase)
    optimizer = optim.AdamW(model_phase.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) 
    optimizer.zero_grad()
    for param, grad in zip(model_phase.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    

def EnergyEstimator_valid_mpi(freq_s, Eloc, amp_qc, sample_size):
    # Step 1: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 2: energy
    wei_freq = wei * freq_s                              # (valid_dim,)
    Eloc_avg = torch.sum(wei_freq * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                               # scalar

    # Step 3: clearup
    del freq_s, wei, wei_freq, Eloc_avg 
    return energy


def EnergyEstimator_valid(Nq, valid_indices, model_sampling, Eloc, amp_qc, sample_size):
    # Step 1: sampling
    s_samp = model_sampling(sample_size)  # directly sampling (sample_size, Nq)
    s_dec_samp = (s_samp.float() @ (2. ** torch.arange(Nq, device=device))).long()
    unique_values, counts = torch.unique(s_dec_samp, sorted=True, return_counts=True)

    # Step 2: construct freq_s
    freq_s = torch.zeros(len(valid_indices), dtype=torch.complex64, device=device)
    freq_s.scatter_(0, torch.searchsorted(valid_indices, unique_values), counts.to(torch.complex64))  # (valid_dim, )

    # Step 3: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 4: energy
    wei_freq = wei * freq_s                              # (valid_dim,)
    Eloc_avg = torch.sum(wei_freq * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                               # scalar

    # Step 5: Clear intermediate variables
    del s_samp, s_dec_samp, unique_values, counts, freq_s, wei, wei_freq, Eloc_avg 

    return energy


def step_decay(t, boundaries, values):
    """
    boundaries: list of step thresholds
    values: list of lr values (len(values) = len(boundaries)+1)
    """
    for i, b in enumerate(boundaries):
        if t < b:
            return values[i]
    return values[-1]
