from package_hybrid_ansatz import *

def main():
    # random seed
    seed = 0 
    local_seed = seed + rank * 1 
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    comm.Barrier()
    start_time = MPI.Wtime()

    #hyperparameters
    Nq = 7
    system = 'spin'
    name = 'Heisenberg'
    n_up = math.floor(Nq / 2)
    n_down = math.ceil(Nq / 2)
    further_train = False # True  
    opt_method = 'SR'
    sample_size = 10000
    mpi = False
    epoch = 1000 # 5000
    n_seed = 100

    # Quantum circuits
    simulator = "lightning.qubit" 
    shots = 10000 # None
    diff_method = "parameter-shift" # "parameter-shift" "backprop" "adjoint"   
    ansatz_reps = 8 
    entanglement = 'linear'  # 'full' 'circular' 'pairwise' 'linear'
    ansatz = create_ansatz(Nq, entanglement, ansatz_reps, simulator, shots, diff_method)
    std_ansatz = 0.1
    std_coef = 0.1
    num_ansatz_params = (ansatz_reps+1) * Nq * 2

    if rank == 0:
        if opt_method == 'SR':
            lr_init = 0.03  
            lr_end = 0.01   
            regularization_value = 0.003  # 0.001
        elif opt_method == 'Adam':
            lr_init = 0.005  
            lr_end = 0.0005 
            beta1 = 0.9  # 0.9
            beta2 = 0.95  # 0.99
            weight_decay = 0.0  # 0.01
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

    if further_train == True:
        if rank == 0:
            save_dict = torch.load(f'data/qml-VMC-{name}-Nq={Nq}-{sample_size}-{shots}.pt')
            model_Transformer.load_state_dict(save_dict['model_Transformer'])
            model_phase.load_state_dict(save_dict['model_phase'])
            params = (
                save_dict['ansatz_params1'].to(device),
                save_dict['coef_params1'].to(device),
                save_dict['ansatz_params2'].to(device),
                save_dict['coef_params2'].to(device)
            )
        else:
            params = (None, None, None, None)
        params = comm.bcast(params, root=0)
        ansatz_params1, coef_params1, ansatz_params2, coef_params2 = params
    else:           
        if rank == 0:
            ansatz_params1 = torch.normal(0, std_ansatz, (num_ansatz_params, ), device=device)
            coef_params1 = torch.normal(0, std_coef, (Nq, ), device=device) 
            ansatz_params2 = torch.normal(0, std_ansatz, (num_ansatz_params, ), device=device)
            coef_params2 = torch.normal(0, std_coef, (Nq, ), device=device)
            params = (ansatz_params1, coef_params1, ansatz_params2, coef_params2)
        else:
            params = (None, None, None, None)
        params = comm.bcast(params, root=0)
        ansatz_params1, coef_params1, ansatz_params2, coef_params2 = params

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
        if opt_method == 'Adam':
            qc_optimizer = AdamWoptimizer(n_params_qc = 2*(len(ansatz_params1)+Nq), 
                                        beta1=beta1, beta2=beta2, weight_decay=weight_decay, epsilon=1e-8)  
        energies = []
    for t in range(epoch):
        if rank == 0:
            lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos(np.pi * t / epoch))  # cosine annealing
            amp_nn, grad_nn1, grad_nn2 = calcAmpGradNn(model_inference, model_phase, valid_states)

        tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2 = combined_expval_and_grad_mpi(ansatz, Nq, ansatz_reps, valid_states, 
                                                                                ansatz_params1, coef_params1, ansatz_params2, coef_params2)
        if rank == 0:
            amp_qc, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2 = calAmpGradQc_Tanh(tot_expval1, expvals1, grads1, 
                                                                                        tot_expval2, expvals2, grads2)
            amp = amp_nn * amp_qc
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)

        if mpi == True:
            freq_s = sampling_mpi(Nq, valid_indices, model_sampling, sample_size)
        if rank == 0:
            if opt_method == 'Adam':
                if mpi == True:
                    energy, Lreg, gradients = EnergyGradientFisher_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, grad_ansatz1, grad_coef1, 
                                                                grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer,
                                                                sample_size, opt_method, reg_coef=0)
                else:                            
                    energy, Lreg, gradients = EnergyGradientFisher_valid(Nq, valid_indices, model_sampling, Eloc,
                                                                            grad_nn1, grad_nn2, grad_ansatz1, grad_coef1,
                                                                            grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer, 
                                                                            sample_size, opt_method, reg_coef=0)                                                             

                ansatz_params1, coef_params1, ansatz_params2, coef_params2 = optAdam(Nq, model_Transformer, model_phase, 
                                                                                n_param_Transformer, n_param_phase, ansatz_params1, 
                                                                                coef_params1, ansatz_params2, coef_params2, gradients, 
                                                                                qc_optimizer, beta1, beta2, weight_decay, lr, t)  
            elif opt_method == 'SR':
                if mpi == True:
                    energy, gradients, fisher = EnergyGradientFisher_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, grad_ansatz1, grad_coef1, 
                                                                grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer,
                                                                sample_size, opt_method, reg_coef=0)
                else:
                    energy, gradients, fisher = EnergyGradientFisher_valid(Nq, valid_indices, model_sampling, Eloc,
                                                                            grad_nn1, grad_nn2, grad_ansatz1, grad_coef1,
                                                                            grad_ansatz2, grad_coef2, amp_nn, amp_qc, n_param_Transformer, 
                                                                            sample_size, opt_method, reg_coef=0) 

                ansatz_params1, coef_params1, ansatz_params2, coef_params2 = optSR(Nq, model_Transformer, model_phase, n_param_Transformer, n_param_phase, 
                                                                            ansatz_params1, coef_params1, ansatz_params2, coef_params2, 
                                                                            gradients, fisher, regularization_value, learning_rate=lr)
                
            energy = energy.cpu().item()
            print(f'Step={t}, Energy={energy}')
            energies.append(energy)

            del grad_nn1, grad_nn2, tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2
            del grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2
            del amp, Eloc, energy, gradients
            if opt_method == 'SR':
                del fisher


    if rank == 0:
        energies_seeds = []
    for i in range(n_seed):
        seed1 = i * 100 # i
        local_seed = seed1 + rank * 10000  
        np.random.seed(local_seed)
        torch.manual_seed(local_seed)

        tot_expval1, expvals1, grads1, tot_expval2, expvals2, grads2 = combined_expval_and_grad_mpi(ansatz, Nq, ansatz_reps, valid_states, 
                                                                                ansatz_params1, coef_params1, ansatz_params2, coef_params2)
        if rank == 0:
            amp_qc, grad_ansatz1, grad_coef1, grad_ansatz2, grad_coef2 = calAmpGradQc_Tanh(tot_expval1, expvals1, grads1, 
                                                                                        tot_expval2, expvals2, grads2)
            amp = amp_nn * amp_qc
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)
        if mpi == True:
            freq_s = sampling_mpi(Nq, valid_indices, model_sampling, sample_size)
        if rank == 0:
            if mpi == True:
                energy = EnergyEstimator_valid_mpi(freq_s, Eloc, amp_qc, sample_size)
            else:
                energy = EnergyEstimator_valid(Nq, valid_indices, model_sampling, Eloc, amp_qc, sample_size)
            print(f'Seed={seed1}, Energy={energy}')
            energies_seeds.append(energy.cpu().item())


    os.makedirs("data", exist_ok=True)
    if rank == 0:
        save_dict = {
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
            'opt_method': opt_method,
            'lr_init': lr_init,
            'lr_end': lr_end,
            'epoch': epoch,
            #'Eg': Eg,
            'energies': energies,
            'energies_seeds': energies_seeds,
            #'amp_nn': amp_nn,
            #'amp_qc': amp_qc,
        }
        if opt_method == 'SR':
            save_dict['regularization_value'] = regularization_value
        elif opt_method == 'Adam':
            save_dict['beta1'] = beta1
            save_dict['beta2'] = beta2
            save_dict['weight_decay'] = weight_decay

        save_dict['model_Transformer'] = model_Transformer.state_dict()
        save_dict['model_phase'] = model_phase.state_dict()
        save_dict['ansatz_params1'] = ansatz_params1
        save_dict['coef_params1'] = coef_params1
        save_dict['ansatz_params2'] = ansatz_params2
        save_dict['coef_params2'] = coef_params2

        torch.save(save_dict, f'data/qml-VMC-{name}-Nq={Nq}-[{n_embd},{n_head},{n_layer}]-{layer_dims}-reps={ansatz_reps}-{entanglement}-{sample_size}-{shots}.pt') 
    
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Total computation time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()