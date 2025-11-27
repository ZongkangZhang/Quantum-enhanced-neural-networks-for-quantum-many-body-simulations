from package_hybrid_ansatz import *

def main():
    # random seed
    seed = 123
    local_seed = seed + rank * 1 
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    comm.Barrier()
    start_time = MPI.Wtime()

    #hyperparameters
    Nq = 12 
    system = 'spin'
    name = 'Heisenberg'
    n_up = math.floor(Nq / 2)
    n_down = math.ceil(Nq / 2)
    further_train = False
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
            lr_end = 0.0005 
            beta1 = 0.9  #
            beta2 = 0.95  
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


    model_sampling = TransformerSampling(model_Transformer).to(device)
    # optimization
    if rank == 0:
        model_inference = TransformerInference(model_Transformer).to(device)

        energies = []
    for t in range(epoch):
        if rank == 0:
            lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos(np.pi * t / epoch)) # cosine

            amp_nn, grad_nn1, grad_nn2 = calcAmpGradNn(model_inference, model_phase, valid_states)

            amp = amp_nn 
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)

        if rank == 0:
            if opt_method == 'Adam':
                energy, Lreg, gradients = EnergyGradientFisher_nqs_valid(Nq, valid_indices, model_sampling, Eloc, grad_nn1, grad_nn2, amp_nn, 
                                                                         n_param_Transformer, sample_size, opt_method, reg_coef=0)
                optAdam_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, gradients, beta1, beta2, weight_decay, learning_rate=lr)

            elif opt_method == 'SR':
                energy, gradients, fisher = EnergyGradientFisher_nqs_valid(Nq, valid_indices, model_sampling, Eloc, grad_nn1, grad_nn2, amp_nn, 
                                                                          n_param_Transformer, sample_size, opt_method, reg_coef=0)
                optSR_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, gradients, fisher, regularization_value, learning_rate=lr)
                
            energy = energy.cpu().item()
            energies.append(energy)
            print(f'Step={t}, Energy={energy}')

            del amp_nn, grad_nn1, grad_nn2
            del amp, Eloc, gradients
            if opt_method == 'SR':
                del fisher


    os.makedirs("data", exist_ok=True)
    if rank == 0:
        save_dict = {
            'model_Transformer': model_Transformer.state_dict(),
            'model_phase': model_phase.state_dict(),
            'seed': seed,
            'Nq': Nq,
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
        if opt_method == 'Adam':
            save_dict['beta1'] = beta1
            save_dict['beta2'] = beta2
            save_dict['weight_decay'] = weight_decay

        #torch.save(save_dict, f'data/NQS-VMC-{name}-Nq={Nq}-pre_train.pt') 
        torch.save(save_dict, f'data/NQS-VMC-{name}-Nq={Nq}-[{n_embd},{n_head},{n_layer}]-{layer_dims}-{sample_size}.pt') 
    
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Total computation time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()