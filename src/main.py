import os
import argparse
import train
import eval

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', type=str, default="train", help="train or test_eval.")
    parser.add_argument('--sge_task_id',  type=int, help="SGE TASK ID", default=0)
    parser.add_argument('--dataset_path', type=str, help="Dataset path.")    
    parser.add_argument('--dataset_name', type=str, help="Dataset name.")    
    parser.add_argument('--output_path',  type=str, help="Output path.", default='output')
    parser.add_argument('--dataset_size', type=int, help="Dataset size per drug.")
    parser.add_argument('--device_number',type=int, help="CUDA VISIBLE DEVICES.", default=0)
    parser.add_argument('--batch_size',type=int, help="Image batch size.", default=128)
    
    # only test eval arguments
    parser.add_argument('--dataset_group_id',type=int, help="Id of the dataset split.", default=1)
    parser.add_argument('--dataset_size_list', type=str, help="Dataset size list")
    parser.add_argument('--drug_label_fname', type=str, default="../datasets/msh_tanzania_blank_drugs.csv", help="List of drug and distractor names.")
    parser.add_argument('--num_seeds', type=int, help="Number of seeds used by dataset/train.")
    parser.add_argument('--prediction_path', type=str, help="Path for prediction files.")
    
    args = parser.parse_args()    
    
    # Set default params
    d_params = {"sge_task_id": args.sge_task_id,
                "dataset_path": args.dataset_path,                
                "dataset_name": args.dataset_name,                
                "output_path": args.output_path,                
                "dataset_size": args.dataset_size,
                "device_number": args.device_number,                
                "batch_size": args.batch_size,
                "dataset_group_id": args.dataset_group_id,
                "dataset_size_list": args.dataset_size_list,
                "drug_label_fname": args.drug_label_fname,
                "num_seeds": args.num_seeds,
                "prediction_path": args.prediction_path
               }
    
    
    if args.mode == "train":
        # Launch training
        train.train(**d_params)
  
    if args.mode == "test_eval":
        # Launch test eval
        eval.eval(**d_params)
       
