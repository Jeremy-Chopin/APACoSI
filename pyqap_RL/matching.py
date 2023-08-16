import os
from os.path import join, isdir
from os import mkdir, listdir
import pandas as pd
import numpy as np
import torch
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from scipy import ndimage

from .utils import load_file, load_knowledge, calculate_max_diagonal
from .QAP.QAP_utils import create_adjency_matrix
from .QAP.KvConstructor import KvConstructor
from .QAP.KeConstructor import KeConstructor

from .RL import subject
from .RL import subjects
from .RL import params
from .RL import train_bench
from .RL import test_bench
from .RL import RL_utils
from .RL import RL_env

def create_prior_knowledge(nb_classes, data_path, experiments_path, node_knowledge_constructor, edges_knowledge_constructor, parameters):

    gt_path = join(data_path, 'annotations', 'train')

    knowledge_path = join(experiments_path, 'knowledges')
    if isdir(knowledge_path) is False:
        mkdir(knowledge_path)
        
    nodes_knowledge_path = join(experiments_path, 'knowledges', 'nodes')
    if isdir(nodes_knowledge_path) is False:
        mkdir(nodes_knowledge_path)
        
    edges_knowledge_path = join(experiments_path, 'knowledges', 'edges')
    if isdir(edges_knowledge_path) is False:
        mkdir(edges_knowledge_path) 

    files = listdir(gt_path)

    files = pd.read_csv(join(data_path, 'train_splits.csv'), sep=';').values

    files_path = []

    for file in files:

        path_to_file = join(gt_path, file[1])

        anno = load_file(path_to_file)

        v = np.nonzero(np.unique(anno))[0]

        if len(v) == nb_classes:
            files_path.append(join(gt_path, file[1]))
        
    Ar = create_adjency_matrix(node_knowledge_constructor, parameters, files_path)

    Ae = create_adjency_matrix(edges_knowledge_constructor, parameters, files_path)

    for specifier in node_knowledge_constructor.specifier_list:
        Ar_specifier_path = join(nodes_knowledge_path, "{}.npy".format(specifier.name))
        np.save(Ar_specifier_path, Ar[specifier.name])

    for specifier in edges_knowledge_constructor.specifier_list:
        Ae_specifier_path = join(edges_knowledge_path, "{}.npy".format(specifier.name))
        np.save(Ae_specifier_path, Ae[specifier.name])

def create_K_matrices(nb_classes, alpha, modes, data_path, experiments_path, nodes_specifier, edges_specifier, params):

    knowledge_dir = join(experiments_path, 'knowledges')

    K_dir = join(experiments_path, 'K')
    if isdir(K_dir) is False:
        mkdir(K_dir)

    node_knowledge = load_knowledge(knowledge_dir, 'nodes', nodes_specifier)
    edge_knowledge = load_knowledge(knowledge_dir, 'edges', edges_specifier)

    # Processing
    for mode in modes:

        K_mode_dir = join(K_dir, mode)
        if isdir(K_mode_dir) is False:
            mkdir(K_mode_dir)

        probabilities_dir = join(data_path, 'segmentation', mode)

        if mode == 'train':
            files = pd.read_csv(join(data_path, '{}_splits.csv'.format(mode)), sep=';').values
        else:
            files = os.listdir(probabilities_dir)
        
        for file in files:
            
            if mode == 'train':
                file = file[1]
            
            id = file.split('.')[0]
            
            # Paths

            probabilities_map_path = join(probabilities_dir, '{}.npy'.format(id))
            
            # load data
            probabilites_map = np.load(probabilities_map_path)

            if mode == 'test':
                probabilites_map = torch.softmax(torch.from_numpy(probabilites_map), dim=0)
                probabilites_map = probabilites_map.numpy()

            if probabilites_map.shape[0]  != 9:
                probabilites_map = np.transpose(probabilites_map,  (2,0,1))
            
            image_cnn = np.argmax(probabilites_map, axis=0)

            assert probabilites_map.shape == (9, 512, 352)

            # Pre-traitements
            dims = len(image_cnn.shape)
            probabilites_map = np.transpose(probabilites_map, (1,2,0))
            probabilites_map = probabilites_map[:,:, 1:nb_classes + 1]
            params['Cs'] = calculate_max_diagonal(image_cnn)
            image_cnn = median_filter(image_cnn, 3)
            
            # Create K
            
            labelled_image = label(image_cnn)#label(image_cnn)
            regions = regionprops(labelled_image, image_cnn)
                        
            kv_constructor = KvConstructor(nodes_specifier, [1], node_knowledge)

            kv = kv_constructor.construct_Kv_initial(probabilites_map, labelled_image, regions, params, nb_classes)

            ke_constructor = KeConstructor(edges_specifier, [1], edge_knowledge)
                
            ke = ke_constructor.construct_Ke_initial(probabilites_map, labelled_image, regions, params, nb_classes)
            
            k = alpha * kv + (1-alpha) * ke

            k_path = join(K_mode_dir, '{}_k.npy'.format(file.split('.')[0]))
            np.save(k_path, k)

def training(nb_class, seed_sizes, data_path, experiments_path, tr_parameters):

    for seed_size in seed_sizes:

        print('\nTraining with a seed of size {}'.format(seed_size))

        DIR_GT_PATH = join(data_path, 'annotations', 'train')
        DIR_SEG_PATH = os.path.join(data_path, 'segmentation', 'train')

        DIR_K_PATH = os.path.join(experiments_path,'K', 'train')

        Q_table_path = os.path.join(experiments_path, 'Q_table_{}.csv'.format(seed_size))
        training_time_path = os.path.join(experiments_path, 'training_time_{}.csv'.format(seed_size))
        
        # Loading of the data

        subject_list = []

        for index in range(0, len(os.listdir(DIR_SEG_PATH))):
            file = os.listdir(DIR_SEG_PATH)[index].replace('.npy', '')
            
            seg_path = os.path.join(DIR_SEG_PATH, '{}.npy'.format(file))
            gt_path = os.path.join(DIR_GT_PATH, '{}.png'.format(file))
            matrice_K_path = os.path.join(DIR_K_PATH, '{}_k.npy'.format(file))

            prob_map = load_file(seg_path)

            if prob_map.shape[0] == 9:
                prob_map = np.transpose(prob_map, (1,2,0))
            segmentation = np.argmax(prob_map, axis=2)	
            prob_map = prob_map[:,:,1:nb_class+1]
        
            segmentation = ndimage.median_filter(segmentation, 3)
            segmentation = label(segmentation)

            groundtruth = load_file(gt_path).astype(np.uint8)

            K = np.load(matrice_K_path)

            if len(np.unique(groundtruth)) == nb_class +1 :
                sub = subject.Subject(index, name = file, groundtruth=groundtruth, probability_map=prob_map, segmentation=segmentation, K=K)
                subject_list.append(sub)

        subs = subjects.Subjects(subject_list)

        # Modele parameters

        noeuds_model = np.arange(nb_class)

        states, actions, q, r = RL_utils.create_actions_states_Q_R_matrices(noeuds_model, seed_size)

        actions_dict = RL_utils.create_states_actions_dict(r)

        d = RL_utils.create_new_states_matrix(actions, states, r)

        param = params.Params(
            actions=actions,
            states=states,
            actions_dict=actions_dict,
            Q = q,
            R = r,
            D = d,
            nb_episodes_per_image = 50
        )



        # loading environements
        env = RL_env.GymQapEnv(states, actions, d)

        # Training
        tr_bench = train_bench.TrainBench(subs, param, tr_parameters, env)

        tr_bench.train(Q_table_path, training_time_path)

def inference(nb_class, seed_sizes, data_path, experiments_path):
    nb_class = 8
    seed_sizes = [2, 3]

    for seed_size in seed_sizes:

        print('\nInference with a seed of size {}'.format(seed_size))

        DIR_GT_PATH = join(data_path, 'annotations', 'test')

        DIR_SEG_PATH = os.path.join(data_path, 'segmentation', 'test')

        DIR_K_PATH = os.path.join(experiments_path, 'K', 'test')

        Q_table_path = os.path.join(experiments_path, 'Q_table_{}.csv'.format(str(seed_size)))

        proposal_pat = os.path.join(experiments_path, 'Proposal')
        if os.path.isdir(proposal_pat) is False:
            os.mkdir(proposal_pat)

        results_path = os.path.join(proposal_pat, 'Seed_{}'.format(str(seed_size)))
        if os.path.isdir(results_path) is False:
            os.mkdir(results_path)

        results_path = os.path.join(results_path, 'Learned')
        if os.path.isdir(results_path) is False:
            os.mkdir(results_path)

        testing_time_path = os.path.join(results_path, 'testing_time.csv')

        # Loading of the data

        subject_list = []

        files = os.listdir(DIR_SEG_PATH)

        for index in range(0, len(files)):
            file = os.listdir(DIR_SEG_PATH)[index].replace('.npy', '')
                
            seg_path = os.path.join(DIR_SEG_PATH, '{}.npy'.format(file))
            gt_path = os.path.join(DIR_GT_PATH, '{}.png'.format(file))
            matrice_K_path = os.path.join(DIR_K_PATH, '{}_k.npy'.format(file))

            prob_map = load_file(seg_path)

            prob_map = torch.softmax(torch.from_numpy(prob_map), dim=0)
            prob_map = prob_map.numpy()

            if prob_map.shape[0] == 9:
                prob_map = np.transpose(prob_map, (1,2,0))

            segmentation = np.argmax(prob_map, axis=2)	
            prob_map = prob_map[:,:,1:nb_class+1]
        
            segmentation = ndimage.median_filter(segmentation, 3)
            lbl = label(segmentation)
            
            groundtruth = load_file(gt_path).astype(np.uint8)

            #affine = load_affine(gt_path)

            K = np.load(matrice_K_path)

            v = len(np.unique(groundtruth))
            v1 = len(np.unique(segmentation))

            if  v == nb_class +1 and v1 == nb_class +1:
                sub = subject.Subject(index, name = file , groundtruth=groundtruth, probability_map=prob_map, segmentation=segmentation, labels=lbl, K=K)
                subject_list.append(sub)

        subs = subjects.Subjects(subject_list)

        # Modele parameters

        noeuds_model = np.arange(nb_class)

        states, actions, q, r = RL_utils.create_actions_states_Q_R_matrices(noeuds_model, seed_size)

        actions_dict = RL_utils.create_states_actions_dict(r)

        d = RL_utils.create_new_states_matrix(actions, states, r)

        param = params.Params(
            actions=actions,
            states=states,
            actions_dict=actions_dict,
            Q = q,
            R = r,
            D = d,
            nb_episodes_per_image = 1
        )

        # loading environements
        env = RL_env.GymQapEnv(states, actions, d)

        # Training
        t_bench = test_bench.TestBench(subs, param, env, results_path)

        t_bench.load_Q_table(Q_table_path)

        t_bench.test(testing_time_path, nb_class, seed_size)