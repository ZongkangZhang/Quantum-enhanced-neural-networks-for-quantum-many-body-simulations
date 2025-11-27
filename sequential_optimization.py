from package_hybrid_ansatz import *

def EnergyGradientFisher_qc_valid_mpi(freq_s, Eloc, grad_ansatz1, grad_coef1,
                                   grad_ansatz2, grad_coef2, amp_qc,  
                                   sample_size, opt_method):
    # Step 1: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 2: order of cat: ansatz + coef
    O_all = torch.cat((grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2), dim=1)  # (valid_dim, num_all_params)

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

    if  opt_method == 'Adam':
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients  
    
    if opt_method == 'SR':
        # Step 5: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 6: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        return energy, gradients, fisher
    

def EnergyGradientFisher_qc_valid(Nq, valid_indices, model_sampling, Eloc, grad_ansatz1, grad_coef1,
                                   grad_ansatz2, grad_coef2, amp_qc,  
                                   sample_size, opt_method):
    # Step 1: sampling
    s_samp = model_sampling(sample_size)  # directly sampling (sample_size, Nq)
    s_dec_samp = (s_samp.float() @ (2. ** torch.arange(Nq, device=device))).long()
    unique_values, counts = torch.unique(s_dec_samp, sorted=True, return_counts=True)

    # Step 2: construct freq_s
    freq_s = torch.zeros(len(valid_indices), dtype=torch.complex64, device=device)
    freq_s.scatter_(0, torch.searchsorted(valid_indices, unique_values), counts.to(torch.complex64))  # (valid_dim, )

    # Step 3: modification factor
    wei = (torch.abs(amp_qc)**2) / (torch.sum((torch.abs(amp_qc)**2) * freq_s) / sample_size)  # (valid_dim, )

    # Step 4: order of cat: ansatz + coef
    O_all = torch.cat((grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2), dim=1)  # (valid_dim, num_all_params)

    # Step 5: energy
    wei_freq = wei * freq_s                              # (valid_dim,)
    Eloc_avg = torch.sum(wei_freq * Eloc) / sample_size  # scalar
    energy = Eloc_avg.real                               # scalar

    # Step 6: gradients
    O_weighted = O_all * wei_freq[:, None]                  # (valid_dim, num_all_params)
    Odag_weighted = O_weighted.conj()                       # (valid_dim, num_all_params)
    Eloc_Odag_avg = (Odag_weighted.T @ Eloc) / sample_size  # (num_all_params,)
    O_avg = torch.sum(O_weighted, dim=0) / sample_size      # (num_all_params,)
    Odag_avg = O_avg.conj()                                 # (num_all_params,)
    gradients = (Eloc_Odag_avg - Eloc_avg * Odag_avg).real  # (num_all_params,)

    if  opt_method == 'Adam':
        # Step 7: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg
        return energy, None, gradients  
    
    if opt_method == 'SR':
        # Step 7: Fisher matrix
        Odag_O_avg = (Odag_weighted.T @ O_all) / sample_size       #   (num_all_params, num_all_params)
        fisher = (Odag_O_avg - torch.outer(Odag_avg, O_avg)).real  # S (num_all_params, num_all_params)
        
        # Step 8: clearup
        del freq_s, wei, O_all, wei_freq, Eloc_avg 
        del O_weighted, Odag_weighted, Eloc_Odag_avg, O_avg, Odag_avg, Odag_O_avg
        return energy, gradients, fisher


# stochastic reconfiguration
def optSR_qc(Nq, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
          gradients, fisher, regularization_value, learning_rate):
    reg_matrix = fisher + regularization_value*torch.eye(len(ansatz_params1)*2+Nq*2, device=device)
    gradients = torch.linalg.solve(reg_matrix, gradients)

    
    ansatz_params1 = ansatz_params1.detach() - learning_rate * gradients[:len(ansatz_params1)]
    coef_params1 = coef_params1 - learning_rate * gradients[len(ansatz_params1):len(ansatz_params1)+Nq]
    ansatz_params2 = ansatz_params2.detach() - learning_rate * gradients[len(ansatz_params1)+Nq:len(ansatz_params1)*2+Nq]
    coef_params2 = coef_params2 - learning_rate * gradients[len(ansatz_params1)*2+Nq:]

    del reg_matrix
    return ansatz_params1, coef_params1, ansatz_params2, coef_params2


# AdamW optimizer
def optAdam_qc(Nq, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
            gradients, qc_optimizer, learning_rate, t):
    
    params = torch.cat([ansatz_params1,coef_params1, ansatz_params2,coef_params2])
    grads = gradients
    params = qc_optimizer.step(step=t, lr=learning_rate, params=params, grads=grads)

    ansatz_params1 = params[:len(ansatz_params1)]
    coef_params1 = params[len(ansatz_params1):len(ansatz_params1)+Nq]
    ansatz_params2 = params[len(ansatz_params1)+Nq:len(ansatz_params1)*2+Nq]
    coef_params2 = params[len(ansatz_params1)*2+Nq:]

    del params, grads
    return ansatz_params1.detach(), coef_params1, ansatz_params2.detach(), coef_params2


def main():
    # random seed
    seed = 123 
    local_seed = seed + rank * 1 
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    comm.Barrier()
    start_time = MPI.Wtime()

    #hyperparameters
    Nq = 10
    system = 'spin'
    name = 'Heisenberg'
    n_up = math.floor(Nq / 2)
    n_down = math.ceil(Nq / 2)
    further_train = True
    opt_method = 'SR'
    sample_size = 10000    
    epoch = 1000
    if rank == 0:
        if opt_method == 'SR':
            lr_init = 0.03  
            lr_end = 0.01   
            regularization_value = 0.001
        elif opt_method == 'Adam':
            lr_init = 0.005  
            lr_end = 0.001 
            beta1 = 0.9  
            beta2 = 0.99  
            weight_decay = 0.0  
        basis = bin_basis(Nq)
        valid_indices, valid_states = prune(Nq, basis, system, n_up, n_down)
        valid_dim = len(valid_indices)

        # generate Hamiltonian
        hamiltonian_dict = HeisenbergHam(Nq)  
        ops_tensor, coeff_tensor, matXY, matYZ, occY = hamiltonian_to_tensors(hamiltonian_dict, Nq)

        # FeedForward neural network
        layer_dims=[32, 32]
        model_phase = FeedforwardPhase(Nq, layer_dims).to(device)
        n_param_phase = sum(p.numel() for p in model_phase.parameters())
    else:
        valid_indices, valid_states = None, None
    valid_indices, valid_states = comm.bcast((valid_indices, valid_states), root=0)

    # Transformer neural network
    block_size = Nq + 1  # prefix a zero qubit to the system bitstring
    n_embd = 8  # model size
    n_head = 4  # head_size = n_embd / n_head
    n_layer = 2
    model_Transformer = Transformer(qubit_size, block_size, n_embd, n_head, n_layer, system, n_up, n_down).to(device)
    if rank == 0:
        n_param_Transformer = sum(p.numel() for p in model_Transformer.parameters())    
    
    # Quantum circuits
    simulator = "default.qubit" 
    shots = None # 10000 
    diff_method = "backprop"  # "parameter-shift" "backprop" "adjoint"   
    ansatz_reps = 8  
    entanglement = 'linear'  # 'full' 'circular' 'pairwise' 'linear'
    ansatz = create_ansatz(Nq, entanglement, ansatz_reps, simulator, shots, diff_method)
    std_ansatz = 0.1  # 0.1
    std_coef = 0.0
    num_ansatz_params = (ansatz_reps+1) * Nq * 2

    if further_train == True:
        if rank == 0:
            save_dict = torch.load(f'data/NQS-VMC-{name}-Nq={Nq}-pre_train.pt')
            model_Transformer.load_state_dict(save_dict['model_Transformer'])
            model_phase.load_state_dict(save_dict['model_phase'])
       
    ansatz_params1 = torch.normal(0, std_ansatz, (num_ansatz_params, ), device=device)
    coef_params1 = torch.normal(0, std_coef, (Nq, ), device=device)
    ansatz_params2 = torch.normal(0, std_ansatz, (num_ansatz_params, ), device=device)
    coef_params2 = torch.normal(0, std_coef, (Nq, ), device=device)

    model_sampling = TransformerSampling(model_Transformer).to(device)
    # optimization
    if rank == 0:
        model_inference = TransformerInference(model_Transformer).to(device)
        if opt_method == 'Adam':
            qc_optimizer = AdamWoptimizer(n_params_qc = 2*(len(ansatz_params1)+Nq), 
                                        beta1=beta1, beta2=beta2, weight_decay=weight_decay, epsilon=1e-8)  
        energies = []
        amp_nn, grad_nn1, grad_nn2 = calcAmpGradNn(model_inference, model_phase, valid_states)
        del grad_nn1, grad_nn2

    for t in range(epoch):
        if rank == 0:
            lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos(np.pi * t / epoch))

        tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2 = combined_expval_and_grad_mpi(ansatz, Nq, ansatz_reps, valid_states, 
                                                                                ansatz_params1, coef_params1, ansatz_params2, coef_params2)
        if rank == 0:
            amp_qc, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2 = calAmpGradQc_Tanh(tot_expval1, expvals1, grads1, 
                                                                                        tot_expval2, expvals2, grads2)
            amp = amp_nn * amp_qc
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)

        if rank == 0:
            if opt_method == 'Adam':
                energy, Lreg, gradients = EnergyGradientFisher_qc_valid(Nq, valid_indices, model_sampling, Eloc, grad_ansatz1, grad_coef1,
                                                                        grad_ansatz2, grad_coef2, amp_qc,  
                                                                        sample_size, opt_method)

                ansatz_params1, coef_params1, ansatz_params2, coef_params2 = optAdam_qc(Nq, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
                                                                                gradients, qc_optimizer, learning_rate=lr, t=t)
            elif opt_method == 'SR':
                energy, gradients, fisher = EnergyGradientFisher_qc_valid(Nq, valid_indices, model_sampling, Eloc, grad_ansatz1, grad_coef1,
                                                                        grad_ansatz2, grad_coef2, amp_qc,  
                                                                        sample_size, opt_method)

                ansatz_params1, coef_params1, ansatz_params2, coef_params2 = optSR_qc(Nq, ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
                                                                            gradients, fisher, regularization_value, learning_rate=lr)
                
            energy = energy.cpu().item()
            print(f'Step={t}, Energy={energy}')
            energies.append(energy)

            del tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2  # amp_nn
            del amp_qc, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2
            del amp, Eloc, energy, gradients
            if opt_method == 'SR':
                del fisher

    os.makedirs("data", exist_ok=True)
    if rank == 0:
        save_dict = {
            'model_Transformer': model_Transformer.state_dict(),
            'model_phase': model_phase.state_dict(),
            'ansatz_params1': ansatz_params1,
            'coef_params1': coef_params1,
            'ansatz_params2': ansatz_params2,
            'coef_params2': coef_params2,
            'seed': seed,
            'Nq': Nq,
            'simulator': simulator, 
            'shots': shots, 
            'diff_method': diff_method,
            'entanglement': entanglement,
            'ansatz_reps': ansatz_reps,
            'std_ansatz': std_ansatz,
            'std_coef': std_coef,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'layer_dims': layer_dims,
            'sample_size': sample_size,
            'lr_init': lr_init,
            'lr_end': lr_end,
            'opt_method': opt_method,
            'epoch': epoch,
            'energies': energies  
        }
        if opt_method == 'SR':
            save_dict['regularization_value'] = regularization_value
        elif opt_method == 'Adam':
            save_dict['beta1'] = beta1
            save_dict['beta2'] = beta2
            save_dict['weight_decay'] = weight_decay

        torch.save(save_dict, f'data/seq_opt-{name}-Nq={Nq}-reps={ansatz_reps}-{entanglement}-{sample_size}-{shots}.pt') 
    
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Total computation time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()