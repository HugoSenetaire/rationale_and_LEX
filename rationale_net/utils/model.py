import torch
import torch.nn as nn
import rationale_net.models.encoder as encoder
import rationale_net.models.imputer as imputer
import rationale_net.models.generator as generator
import rationale_net.models.tagger as tagger
import rationale_net.models.empty as empty
import rationale_net.utils.learn as learn
import os
import pdb

def get_model(args, embeddings, dataset_train):
    if args.imputer is None :
        if args.snapshot is None:
            if args.use_as_tagger == True:
                gen = empty.Empty()
                model = tagger.Tagger(embeddings, args)
            else:
                gen   = generator.Generator(embeddings, args)
                model = encoder.Encoder(embeddings, args)
        else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                gen_path = learn.get_gen_path(args.snapshot)
                if os.path.exists(gen_path):
                    gen   = torch.load(gen_path)
                model = torch.load(args.snapshot)
            except :
                print("Sorry, This snapshot doesn't exist."); exit()
    else :
        if args.imputer == 'markov_chain':
            current_imputer = imputer.MarkovChain(dataset_train)
        gen = generator.Generator(embeddings, args)
        model = encoder.EncoderMultipleImputation(embeddings, args, current_imputer)
    if args.num_gpus > 1:
        model = nn.DataParallel(model,
                                    device_ids=range(args.num_gpus))

        if not gen is None:
            gen = nn.DataParallel(gen,
                                    device_ids=range(args.num_gpus))
    return gen, model
