# -*- coding: utf-8 -*-
"""
    This module is intended to join all the pipeline in separated tasks
    to be executed individually or in a flow by using command-line options

    Example:
    Dataset embedding and processing:
        $ python taskflows.py -e -pS
"""

import argparse
import gc
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from gensim.models.word2vec import Word2Vec

import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg

PATHS = configs.Paths()
FILES = configs.Files()
DEVICE = FILES.get_device()


def select(dataset):
    #here the data of ffmpeg is only considered
    result = dataset.loc[dataset['project'] == "FFmpeg"]
    len_filter = result.func.str.len() < 1200
    result = result.loc[len_filter]
    #debugging step
    #print(len(result))
    #result = result.iloc[11001:]
    #print(len(result))
    # result = result.head(200)   # made a change here 

    return result


def create_task():
    context = configs.Create()
    raw = data.read(PATHS.raw, FILES.raw)
    filtered = data.apply_filter(raw, select)
    filtered = data.clean(filtered) #removes duplicates
    data.drop(filtered, ["commit_id", "project"]) #drops the commit_id and project columns
    slices = data.slice_frame(filtered, context.slice_size) #calls the function in datamanager.py it is dividing
    #find out about context?   so the size of slicing is 100
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]
    # Creates list of (slice_number, slice_dataframe) tuples
    #create CPG files
    cpg_files = []
    # Create CPG binary files
    for s, slice in slices:
        data.to_files(slice, PATHS.joern)
        #something is wrong here bcz no .c files are created?
        cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
        print("CPG file created:", cpg_file)
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        #deleting the joern files is not a good idea bcz it is not deleting the .c files?
        shutil.rmtree(PATHS.joern)   # delete joern files
    # Create CPG with graphs json files
    # json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    # for (s, slice), json_file in zip(slices, json_files):
    #     graphs = prepare.json_process(PATHS.cpg, json_file)
    json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
    print("JSON files returned:", json_files)
    print("DEBUG: joern_create returned ->", json_files)
    #not understoood what is happening here?
    for (s, slice), json_file in zip(slices, json_files):
        graphs = prepare.json_process(PATHS.cpg, json_file)
        # print("Graphs processed:", graphs)
        #not understoood what is happening here?
        if graphs is None:
            print(f"Dataset chunk {s} not processed.")
            continue
        #find out about these indexes?
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        dataset = data.inner_join_by_index(slice, dataset)
        print(f"Writing cpg dataset chunk {s}.")
        data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")
        del dataset
        gc.collect()










def embed_task():
    context = configs.Embed()

    # Initialize Word2Vec model
    w2vmodel = Word2Vec(**context.w2v_args)
    w2v_init = True

    # Get all .pkl files from CPG folder
    dataset_files = [f for f in os.listdir(PATHS.cpg) if f.endswith(".pkl")]
    # print(dataset_files)
    
    # Sort files by their numeric prefix (e.g., 0_cpg.pkl, 1_cpg.pkl, etc.)
    def get_file_number(filename):
        try:
            # Only handle '0_cpg.pkl' style filenames
            return int(filename.split('_')[0])
        except (ValueError, IndexError):
            return float('inf')  # Put problematic files at the end
            
    dataset_files.sort(key=get_file_number)
    # print(dataset_files)
    # flage = True
    for pkl_file in dataset_files:
        file_name = pkl_file.split(".")[0]

        # Load the dataset from CPG directory
        file_path = os.path.join(PATHS.cpg, pkl_file)
        cpg_dataset = pd.read_pickle(file_path)
        
        # Test code to inspect CPG dataset structure
        if 'input' in cpg_dataset.columns and len(cpg_dataset) > 0:
            sample_input = cpg_dataset['input'].iloc[0]
            print("\n--- CPG Dataset Test ---")
            print(f"Sample input type: {type(sample_input)}")
            if hasattr(sample_input, 'x'):
                print(f"Sample input x shape: {sample_input.x.shape}")
            else:
                print("No 'x' attribute in sample input")
                
            if hasattr(sample_input, 'edge_index'):
                print(f"Sample input edge_index shape: {sample_input.edge_index.shape}")
                if hasattr(sample_input.edge_index, 'shape') and len(sample_input.edge_index.shape) > 1 and sample_input.edge_index.shape[1] > 0:
                    print(f"First 10 edges (if any): {sample_input.edge_index[:, :10]}")
                else:
                    print("No edges found or edge_index is not in expected format")
            else:
                print("No 'edge_index' attribute in sample input")
            print("------------------------\n")
        else:
            print("\n--- CPG Dataset Test ---")
            print("No 'input' column in dataset or dataset is empty")
            print("Available columns:", cpg_dataset.columns.tolist())
            print("------------------------\n")
        # if flage:
        #     print(cpg_dataset['func'].iloc[0])
        #     print(cpg_dataset['Index'].iloc[0])
        #     print(cpg_dataset['cpg'].iloc[0])    
        #     print(cpg_dataset['target'].iloc[0])
            
            
        

        # --- Convert function dicts to code strings for tokenization ---
        cpg_dataset['code_str'] = cpg_dataset['func'].apply(
            lambda x: x.get('function', '') if isinstance(x, dict) else str(x)
        )
        print(cpg_dataset.columns)
        # if flage:
        #     print(cpg_dataset['func'].iloc[0])
        #     print(cpg_dataset['Index'].iloc[0])
        #     print(cpg_dataset['cpg'].iloc[0])    
        #     print(cpg_dataset['target'].iloc[0])
        #     print(cpg_dataset['code_str'].iloc[0])
        #     flage = False
        # Tokenize
        tokens_dataset = data.tokenize(
            cpg_dataset[['code_str']].rename(columns={'code_str': 'func'})
        )

        # Filter out empty token lists
        tokens_dataset = tokens_dataset[tokens_dataset['tokens'].map(len) > 0]

        print(f"{file_name}: {len(tokens_dataset)} tokenized functions")
        if len(tokens_dataset) == 0:
            print(f"Skipping Word2Vec and CPG embedding for {file_name}, no valid tokens.")
            continue

        # Save tokens
        data.write(tokens_dataset, PATHS.tokens, f"{file_name}_{FILES.tokens}")
        #stored at data/tokens
        # Word2Vec: build vocabulary & train
        if w2v_init:
            # First time: build full vocabulary
            print("Building initial vocabulary...")
            w2vmodel.build_vocab(corpus_iterable=tokens_dataset.tokens)
            w2v_init = False
        else:
            # Subsequent files: update vocabulary
            w2vmodel.build_vocab(corpus_iterable=tokens_dataset.tokens, update=True)
        
        # Train the model
        print(f"Training on {len(tokens_dataset)} functions...")
        w2vmodel.train(
            corpus_iterable=tokens_dataset.tokens,
            total_examples=len(tokens_dataset),
            epochs=w2vmodel.epochs
        )

        # Ensure cpg column contains dicts
        cpg_dataset['cpg'] = cpg_dataset['cpg'].apply(lambda x: x if isinstance(x, dict) else None)
        cpg_dataset = cpg_dataset.dropna(subset=['cpg'])
        # print(cpg_dataset['cpg'].iloc[0])
        print("\n\n\nNow nodes\n\n\n")
        # --- FIX: parse_to_nodes output must be a list ---
        cpg_dataset["nodes"] = cpg_dataset['cpg'].apply(
            lambda x: cpg.parse_to_nodes(x, context.nodes_dim) if isinstance(x, dict) else {}
        )
        # print(cpg_dataset['nodes'].iloc[0])
        # Remove rows with no nodes
        cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
        if len(cpg_dataset) == 0:
            print(f"No valid nodes in {file_name}, skipping input creation.")
            continue

        # Debug: Detailed node structure inspection
        print("\n=== Node Structure Debug ===")
        sample_nodes = cpg_dataset['nodes'].iloc[0]
        print(f"Sample nodes type: {type(sample_nodes)}, len: {len(sample_nodes) if hasattr(sample_nodes, '__len__') else 'N/A'}")
        
        if hasattr(sample_nodes, '__iter__') and not isinstance(sample_nodes, (str, dict)):
            # Print first 3 nodes' structure
            for i, node in enumerate(list(sample_nodes)[:3]):
                print(f"\nNode {i}:")
                node_dict = node if isinstance(node, dict) else node.__dict__
                print(f"  id: {node_dict.get('id')}")
                
                # Handle edges - could be dict or list
                edges = node_dict.get('edges')
                if edges is not None:
                    if isinstance(edges, dict):
                        print(f"  edges (dict): {list(edges.items())[:3]}...")
                    else:
                        print(f"  edges: {list(edges)[:3]}..." if edges else "  edges: []")
                else:
                    print("  edges: None")
                
                # Handle ast_children - could be dict or list
                ast_children = node_dict.get('ast_children')
                if ast_children is not None:
                    if isinstance(ast_children, dict):
                        print(f"  ast_children (dict): {list(ast_children.items())[:5]}...")
                    else:
                        print(f"  ast_children: {list(ast_children)[:5]}..." if ast_children else "  ast_children: []")
                else:
                    print("  ast_children: None")
                
                # Print all keys in the node
                print(f"  All keys: {list(node_dict.keys())}")
                
                # Print a sample of the node content (first 3 key-value pairs)
                print("  Sample content:")
                for k, v in list(node_dict.items())[:3]:
                    v_preview = str(v)[:50] + '...' if len(str(v)) > 50 else v
                    print(f"    {k}: {v_preview}")
        
        print("=" * 50 + "\n")

        # Prepare input for training
        cpg_dataset["input"] = cpg_dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes, row.target, context.nodes_dim, w2vmodel.wv, context.edge_type
            ),
            axis=1
        )

        # Debug: Print dataset info and sample
        print("\n=== Dataset Debug Info ===")
        print(f"Number of samples: {len(cpg_dataset)}")
        print(f"Columns: {cpg_dataset.columns.tolist()}")
        
        # Print sample input structure
        sample_input = cpg_dataset['input'].iloc[0]
        print("\nSample input structure:")
        print(f"Type: {type(sample_input)}")
        if hasattr(sample_input, 'x'):
            print(f"  - x shape: {sample_input.x.shape if hasattr(sample_input.x, 'shape') else 'N/A'}")
        if hasattr(sample_input, 'edge_index'):
            print(f"  - edge_index shape: {sample_input.edge_index.shape if hasattr(sample_input.edge_index, 'shape') else 'N/A'}")
        if hasattr(sample_input, 'y'):
            print(f"  - y: {sample_input.y}")
        
        # Print target distribution
        print("\nTarget distribution:")
        print(cpg_dataset['target'].value_counts())
        
        # Print first few rows
        print("\nFirst few rows:")
        for idx, row in cpg_dataset.head(2).iterrows():
            print(f"\nRow {idx}:")
            print(f"  Target: {row['target']}")
            print(f"  Input - x shape: {row['input'].x.shape if hasattr(row['input'].x, 'shape') else 'N/A'}")
            print(f"  Input - edge_index shape: {row['input'].edge_index.shape if hasattr(row['input'].edge_index, 'shape') else 'N/A'}")
        
        print("\nSaving input dataset...")
        # Drop intermediate nodes column to save memory
        data.drop(cpg_dataset, ["nodes"])
        
        print(f"Saving input dataset {file_name} with size {len(cpg_dataset)}.")
        data.write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")

        del cpg_dataset, tokens_dataset
        gc.collect()

    # Save Word2Vec model
    print("Saving w2vmodel.")
    w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")
    
    # Print dataset summary
    print_dataset_summary()


def print_dataset_summary():
    """Print a summary of all processed datasets"""
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    
    # Get list of all processed files
    input_files = list(Path(PATHS.input).glob("*_cpg_input.pkl"))
    
    if not input_files:
        print("No processed datasets found!")
        return
    
    summary = []
    
    for file in input_files:
        # Extract dataset name and type from filename
        dataset_name = file.stem.replace("_cpg_input", "")
        
        # Load the dataset
        try:
            # Load the input file directly using pandas
            import pandas as pd
            df = pd.read_pickle(file)
            
            # Calculate stats
            total = len(df)
            vuln = df['target'].sum()
            non_vuln = total - vuln
            
            # Get node and edge info from first sample
            sample = df.iloc[0]['input']
            node_shape = sample.x.shape if hasattr(sample, 'x') else "N/A"
            edge_shape = sample.edge_index.shape if hasattr(sample, 'edge_index') else "N/A"
            
            summary.append({
                'Dataset': dataset_name,
                'Samples': total,
                'Vulnerable': f"{vuln} ({vuln/total:.1%})",
                'Non-Vuln': f"{non_vuln} ({non_vuln/total:.1%})",
                'Node Shape': str(node_shape),
                'Edge Shape': str(edge_shape)
            })
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Print summary table
    if summary:
        import pandas as pd
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n" + pd.DataFrame(summary).to_string(index=False))
    
    print("\n" + "="*80 + "\n")


def process_task(stopping):
    context = configs.Process()
    devign = configs.Devign()
    model_path = PATHS.model + FILES.model
    model = process.Devign(path=model_path, device=DEVICE, model=devign.model, learning_rate=devign.learning_rate,
                           weight_decay=devign.weight_decay,
                           loss_lambda=devign.loss_lambda)
    train = process.Train(model, context.epochs)
    input_dataset = data.loads(PATHS.input)
    # split the dataset and pass to DataLoader with batch size
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            data.train_val_test_split(input_dataset, shuffle=context.shuffle)))
    train_loader_step = process.LoaderStep("Train", train_loader, DEVICE)
    val_loader_step = process.LoaderStep("Validation", val_loader, DEVICE)
    test_loader_step = process.LoaderStep("Test", test_loader, DEVICE)

    if stopping:
        early_stopping = process.EarlyStopping(model, patience=context.patience)
        train(train_loader_step, val_loader_step, early_stopping)
        model.load()
    else:
        train(train_loader_step, val_loader_step)
        model.save()

    process.predict(model, test_loader_step)


def main():
    """
    main function that executes tasks based on command-line options
    """
    parser: ArgumentParser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('-c', '--create', action='store_true')
    parser.add_argument('-e', '--embed', action='store_true')
    parser.add_argument('-p', '--process', action='store_true')
    parser.add_argument('-pS', '--process_stopping', action='store_true')

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)



if __name__ == "__main__":
    main()
