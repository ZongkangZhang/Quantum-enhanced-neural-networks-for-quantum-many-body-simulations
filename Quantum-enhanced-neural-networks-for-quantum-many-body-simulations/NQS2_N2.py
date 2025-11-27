from package_hybrid_ansatz import *

# An additional feedforward neural network state
# ln <s|phi> = Re + Im = a * tanh ffwd_1(s) + i * ffwd_2(s)
class ffwd_real(nn.Module):
    def __init__(self, Nq, layer_dims=[32, 32]):
        super().__init__()
        layers = []
        in_dim = Nq

        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim  
        layers.append(nn.Linear(in_dim, 1, bias=False))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.net(x.float())
        output = 1.0 * torch.tanh(f)   # reduce variance
        return output


class ffwd_imag(nn.Module):
    def __init__(self, Nq, layer_dims=[32, 32]):
        super().__init__()
        layers = []
        in_dim = Nq

        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim  
        layers.append(nn.Linear(in_dim, 1, bias=False))
        
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        output = self.net(x.float())
        return output


# len(valid_s) or sample_size
def calcAmpGradNn2(model_ffwd_real, model_ffwd_imag, valid_states):
    # calculate the neural network wavafunction 
    with torch.no_grad():
        real = model_ffwd_real(valid_states)                                 # （valid_dim, 1)
        imag = model_ffwd_imag(valid_states)                                 # （valid_dim, 1)
        amp_nn2 = torch.exp(real + 1j * imag).flatten()                      # （valid_dim,)
        
    # extract parameters from the models 
    params_real = dict(model_ffwd_real.named_parameters())
    params_imag = dict(model_ffwd_imag.named_parameters())

    def calc_loss3(params, s):
        real = functional_call(model_ffwd_real, params, s.unsqueeze(0))
        return real.squeeze()

    def calc_loss4(params, s):
        imag = functional_call(model_ffwd_imag, params, s.unsqueeze(0))
        return imag.squeeze()

    # calculate gradients 
    grad_nn3_dict = vmap(grad(calc_loss3), in_dims=(None, 0))(params_real, valid_states)
    grad_nn4_dict = vmap(grad(calc_loss4), in_dims=(None, 0))(params_imag, valid_states)
    
    # reshape gradients to fit the optimization 
    grad_nn3 = torch.cat([v.reshape(v.shape[0], -1) for v in grad_nn3_dict.values()], dim=1)  # (valid_dim, n_param_ffwd_real)
    grad_nn4 = torch.cat([v.reshape(v.shape[0], -1) for v in grad_nn4_dict.values()], dim=1)  # (valid_dim, n_param_ffwd_imag)
    grad_nn4 = 1j * grad_nn4 

    del real, imag, grad_nn3_dict, grad_nn4_dict
    return amp_nn2, grad_nn3.detach(), grad_nn4.detach()  # O_i(s) in neural network


def EnergyGradientFisher_2nqs_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, grad_nn3, grad_nn4, amp_nn, amp_nn2, n_param_Transformer, 
                                   sample_size, opt_method, reg_coef):
    # Step 1: modification factor
    wei = (torch.abs(amp_nn2)**2) / (torch.sum((torch.abs(amp_nn2)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 2: order of cat: neural network + ansatz + coef
    O_all = torch.cat((grad_nn1, grad_nn2, grad_nn3, grad_nn4), dim=1)  # (valid_dim, num_all_params)

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


# AdamW optimizer
def optAdam_2nqs(model_Transformer, model_phase, model_ffwd_real, model_ffwd_imag, n_param_Transformer, n_param_phase, 
                n_param_ffwd_real, n_param_ffwd_imag, gradients, beta1, beta2, weight_decay, learning_rate):
    
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
    
    num_params2 = n_param_Transformer + n_param_phase + n_param_ffwd_real
    unflattened_gradients = unflatten_gradients(gradients[num_params:num_params2], model_ffwd_real)
    optimizer = optim.AdamW(model_ffwd_real.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) 
    optimizer.zero_grad()
    for param, grad in zip(model_ffwd_real.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()
    
    num_params3 = n_param_Transformer + n_param_phase + n_param_ffwd_real + n_param_ffwd_imag 
    unflattened_gradients = unflatten_gradients(gradients[num_params2:num_params3], model_ffwd_imag)
    optimizer = optim.AdamW(model_ffwd_imag.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay) 
    optimizer.zero_grad()
    for param, grad in zip(model_ffwd_imag.parameters(), unflattened_gradients):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param.grad.copy_(grad)  # assign gradients to param.grad
    optimizer.step()


def main():
    # random seed
    seed = 1337 
    local_seed = seed + rank * 10000  
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    comm.Barrier()
    start_time = MPI.Wtime()

    #hyperparameters
    Nq = 12 
    system = 'molecule'
    name = 'N2'
    n_up = 3  
    n_down = 3  
    opt_method = 'Adam'
    sample_size = 1000000  
    epoch = 5000
    if rank == 0:
        if opt_method == 'SR':
            lr_init = 0.03  
            lr_end = 0.01   
            regularization_value = 0.001
        elif opt_method == 'Adam':
            lr_init = 0.005  
            lr_end = 0.0005  
            beta1 = 0.9  
            beta2 = 0.95  
            weight_decay = 0.0  
            reg_coef_init = 0.08  # regularizer
            reg_stage = 800
            lr_step_size = 200
            boundaries = list(range(lr_step_size, epoch, lr_step_size))
            lr_decay_rate = 0.92
            values = [lr_init * (lr_decay_rate ** i) for i in range(len(boundaries) + 1)]
        basis = bin_basis(Nq)
        valid_indices, valid_states = prune(Nq, basis, system, n_up, n_down)
        valid_dim = len(valid_indices)

        # generate Hamiltonian
        distance = 1.5
        molecule = MolecularData(geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, distance))], 
                                basis = 'sto-3g', # minimal basis
                                multiplicity = 1,
                                charge = 0)
        molecule = run_pyscf(molecule)
        second_q_hamiltonian = molecule.get_molecular_hamiltonian(
                            occupied_indices = [0, 1, 2, 3], 
                            active_indices = [4, 5, 6, 7, 8, 9] 
                            )
        jw_hamiltonian = jordan_wigner(second_q_hamiltonian)  
        hamiltonian_dict = dict(jw_hamiltonian.terms.items())
        ops_tensor, coeff_tensor, matXY, matYZ, occY = hamiltonian_to_tensors(hamiltonian_dict, Nq)

        # FeedForward neural network
        layer_dims = [80,80]
        model_phase = FeedforwardPhase(Nq, layer_dims).to(device)
        n_param_phase = sum(p.numel() for p in model_phase.parameters())
        
        # An additional feedforward neural network state
        layer_dims_2 = [40,40]
        model_ffwd_real = ffwd_real(Nq, layer_dims_2).to(device)
        model_ffwd_imag = ffwd_imag(Nq, layer_dims_2).to(device)
        n_param_ffwd_real = sum(p.numel() for p in model_ffwd_real.parameters())
        n_param_ffwd_imag = sum(p.numel() for p in model_ffwd_imag.parameters())
    else:
        valid_indices, valid_states = None, None
    valid_indices, valid_states = comm.bcast((valid_indices, valid_states), root=0)

    # Transformer neural network
    block_size = Nq + 1  # prefix a zero qubit to the system bitstring
    n_embd = 14  # model size
    n_head = 7  # head_size = n_embd / n_head
    n_layer = 3
    model_Transformer = Transformer(qubit_size, block_size, n_embd, n_head, n_layer, system, n_up, n_down).to(device)
    if rank == 0:
        n_param_Transformer = sum(p.numel() for p in model_Transformer.parameters())    

    # broadcast rank 0 model state to all ranks
    if rank == 0:
        transformer_state_dict = model_Transformer.state_dict()
    else:
        transformer_state_dict = None
    transformer_state_dict = comm.bcast(transformer_state_dict, root=0)
    if rank != 0:
        model_Transformer.load_state_dict(transformer_state_dict)        

    model_sampling = TransformerSampling(model_Transformer).to(device)
    # optimization
    if rank == 0:
        model_inference = TransformerInference(model_Transformer).to(device)

        energies = []
        Lregs = [] if reg_coef_init != 0 else None
    for t in range(epoch):
        if rank == 0:
            #lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos(np.pi * t / epoch)) # cosine
            lr = step_decay(t, boundaries, values)  # step

            #reg_coef = reg_coef_init if t < reg_stage else 0.0  
            reg_coef = reg_coef_init * (1 - t / reg_stage) if t < reg_stage else 0.0 # linear decay

            amp_nn, grad_nn1, grad_nn2 = calcAmpGradNn(model_inference, model_phase, valid_states)
            amp_nn2, grad_nn3, grad_nn4 = calcAmpGradNn2(model_ffwd_real, model_ffwd_imag, valid_states)

            amp = amp_nn * amp_nn2
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)

        freq_s = sampling_mpi(Nq, valid_indices, model_sampling, sample_size)
        if rank == 0:
            if opt_method == 'Adam':
                energy, Lreg, gradients = EnergyGradientFisher_2nqs_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, grad_nn3, grad_nn4, amp_nn, amp_nn2, 
                                                                             n_param_Transformer, sample_size, opt_method, reg_coef)

                optAdam_2nqs(model_Transformer, model_phase, model_ffwd_real, model_ffwd_imag, n_param_Transformer, n_param_phase, 
                            n_param_ffwd_real, n_param_ffwd_imag, gradients, beta1, beta2, weight_decay, learning_rate=lr)
                

            energies.append(energy.item())
            if reg_coef_init == 0 or Lreg == None:
                print(f'Step={t}, Energy={energy}')
            elif reg_coef_init != 0:
                print(f'Step={t}, Energy={energy}, Lreg={Lreg}, Loss={energy+Lreg}')
                Lregs.append(Lreg.item())     

            del amp_nn, grad_nn1, grad_nn2, amp_nn2, grad_nn3, grad_nn4
            del amp, Eloc, energy, gradients

    os.makedirs("data", exist_ok=True)
    if rank == 0:
        save_dict = {
            'model_Transformer': model_Transformer.state_dict(),
            'model_phase': model_phase.state_dict(),
            'model_ffwd_real': model_ffwd_real.state_dict(),
            'model_ffwd_imag': model_ffwd_imag.state_dict(),
            'seed': seed,
            'Nq': Nq,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'layer_dims': layer_dims,
            'layer_dims_2 ': layer_dims_2,
            'sample_size': sample_size,
            'lr_init': lr_init,
            'lr_end': lr_end,
            'opt_method': opt_method,
            'reg_coef_init': reg_coef_init,
            'reg_stage': reg_stage,
            'lr_decay_rate': lr_decay_rate, 
            'lr_step_size': lr_step_size,
            'epoch': epoch,
            'energies': energies  
        }
        if opt_method == 'Adam':
            save_dict['beta1'] = beta1
            save_dict['beta2'] = beta2
            save_dict['weight_decay'] = weight_decay
        if reg_coef_init != 0:
            save_dict['Lregs'] = Lregs

        torch.save(save_dict, f'data/2_NQS-VMC-{name}-Nq={Nq}-distance={distance}-[{n_embd},{n_head},{n_layer}]-{layer_dims}-{layer_dims_2}.pt') 
    
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Total computation time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()