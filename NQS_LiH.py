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
    Nq = 12 
    system = 'molecule'
    name = 'LiH'
    n_up = 2  
    n_down = 2  
    further_train = False
    opt_method = 'Adam'
    sample_size = 10000  
    epoch = 3000
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
            reg_coef_init = 0.07  # regularizer
            reg_stage = 400
        basis = bin_basis(Nq)
        valid_indices, valid_states = prune(Nq, basis, system, n_up, n_down)
        valid_dim = len(valid_indices)

        # generate Hamiltonian
        distance = 1.5
        molecule = MolecularData(geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, distance))], 
                                basis = 'sto-3g', # minimal basis
                                multiplicity = 1,
                                charge = 0)
        molecule = run_pyscf(molecule)
        second_q_hamiltonian = molecule.get_molecular_hamiltonian(
                                    #occupied_indices = [0],  # core
                                    #active_indices = [1, 2, 5]  # 3,4 are virtual
                                    )
        jw_hamiltonian = jordan_wigner(second_q_hamiltonian)  
        hamiltonian_dict = dict(jw_hamiltonian.terms.items())
        ops_tensor, coeff_tensor, matXY, matYZ, occY = hamiltonian_to_tensors(hamiltonian_dict, Nq)

        # FeedForward neural network
        layer_dims = [32,32]
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
            save_dict = torch.load(f'data/NQS-VMC-{name}-Nq={Nq}-distance={distance}.pt')
            model_Transformer.load_state_dict(save_dict['model_Transformer'])
            model_phase.load_state_dict(save_dict['model_phase'])

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
            lr = lr_end + 0.5 * (lr_init - lr_end) * (1 + np.cos(np.pi * t / epoch)) # cosine

            #reg_coef = reg_coef_init if t < reg_stage else 0.0  
            reg_coef = reg_coef_init * (1 - t / reg_stage) if t < reg_stage else 0.0 # linear decay

            amp_nn, grad_nn1, grad_nn2 = calcAmpGradNn(model_inference, model_phase, valid_states)
            amp = amp_nn 
            Eloc = calcEloc_batch(Nq, valid_states, valid_indices, amp, coeff_tensor, matXY, matYZ, occY)

        freq_s = sampling_mpi(Nq, valid_indices, model_sampling, sample_size)
        if rank == 0:
            if opt_method == 'Adam':
                energy, Lreg, gradients = EnergyGradientFisher_nqs_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, amp_nn, 
                                                                             n_param_Transformer, sample_size, opt_method, reg_coef)

                optAdam_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, gradients, beta1, beta2, weight_decay, learning_rate=lr)

            elif opt_method == 'SR':
                energy, gradients, fisher = EnergyGradientFisher_nqs_valid_mpi(freq_s, Eloc, grad_nn1, grad_nn2, amp_nn, 
                                                                               n_param_Transformer, sample_size, opt_method, reg_coef)

                optSR_nqs(model_Transformer, model_phase, n_param_Transformer, n_param_phase, gradients, fisher, regularization_value, learning_rate=lr)
                

            energies.append(energy.item())
            if reg_coef_init == 0 or Lreg == None:
                print(f'Step={t}, Energy={energy}')
            elif reg_coef_init != 0:
                print(f'Step={t}, Energy={energy}, Lreg={Lreg}, Loss={energy+Lreg}')
                Lregs.append(Lreg.item())     

            del amp_nn, grad_nn1, grad_nn2
            del amp, Eloc, energy, gradients
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
            'reg_coef_init': reg_coef_init,
            'reg_stage': reg_stage,
            'epoch': epoch,
            'energies': energies  
        }
        if opt_method == 'SR':
            save_dict['regularization_value'] = regularization_value
        if opt_method == 'Adam':
            save_dict['beta1'] = beta1
            save_dict['beta2'] = beta2
            save_dict['weight_decay'] = weight_decay
        if reg_coef_init != 0:
            save_dict['Lregs'] = Lregs

        torch.save(save_dict, f'data/NQS-VMC-{name}-Nq={Nq}-distance={distance}-[{n_embd},{n_head},{n_layer}]-{layer_dims}.pt') 
    
    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        print(f"Total computation time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()