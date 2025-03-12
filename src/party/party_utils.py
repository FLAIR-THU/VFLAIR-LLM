import os

Trained_Models_Save_Path = "/home/DAIR/guzx/Git_FedProject/Trained_Models/"

def get_model_folder(args):
    model_folder = Trained_Models_Save_Path+f'{args.dataset}/{args.prefix}/{str(args.vfl_model_slice_num)}-slice/{args.split_info}/{args.defense_name}_{args.defense_param}/seed_{args.current_seed}/trained_model/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    return model_folder

def get_defense_model_folder(args):
    defense_model_folder = Trained_Models_Save_Path+f'{args.dataset}/{args.prefix}/{str(args.vfl_model_slice_num)}-slice/{args.split_info}/{args.defense_name}_{args.defense_param}/seed_{args.current_seed}/defense_model/'
    if not os.path.exists(defense_model_folder):
        os.makedirs(defense_model_folder)
    
    return defense_model_folder