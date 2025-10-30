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
    
    # Check if this is a language-specific dataset
    if 'language' in raw.columns:
        print("🔄 Processing language-specific dataset")
        language = raw['language'].iloc[0] if len(raw) > 0 else 'c'
        print(f"📋 Detected language: {language}")
        
        # Apply basic filtering but keep language info
        filtered = data.apply_filter(raw, select)
        filtered = data.clean(filtered) #removes duplicates
        
        # Don't drop language column for multi-language support
        columns_to_drop = ["commit_id", "project"]
        existing_columns = [col for col in columns_to_drop if col in filtered.columns]
        if existing_columns:
            data.drop(filtered, existing_columns)
        
        # Use smart language-aware slicing
        print(f"📋 Dataset size: {len(filtered)} samples")
        slices = data.smart_language_aware_slice(filtered, context.slice_size)
        
        # Note: smart_language_aware_slice returns (batch_id, batch_df, language) tuples
        # No need for the lambda transformation - already returns proper format
        
        cpg_files = []
        
        # Create CPG binary files with language-specific frontend
        for slice_id, slice_df, slice_language in slices:
            print(f"\n🔄 Processing batch {slice_id}: {len(slice_df)} samples")
            if slice_language:
                print(f"📋 Language: {slice_language}")
            
            # Create files with appropriate extensions
            data.to_files(slice_df, PATHS.joern)
            
            # Use language-specific Joern frontend
            # Use detected language from slice, fallback to original detection if None
            frontend_language = slice_language if slice_language else language
            cpg_file = prepare.joern_parse(
                context.joern_cli_dir, 
                PATHS.joern, 
                PATHS.cpg, 
                f"{slice_id}_{FILES.cpg}",
                language=frontend_language
            )
            
            print(f"✅ CPG file created: {cpg_file}")
            cpg_files.append(cpg_file)
            print(f"📋 Dataset batch {slice_id} converted to CPG")
            
            # Clean up joern files
            if os.path.exists(PATHS.joern):
                shutil.rmtree(PATHS.joern)
        
        # Create JSON files from CPG
        print(f"\n🔄 Creating JSON files from {len(cpg_files)} CPG files")
        json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
        print(f"📋 JSON files created: {json_files}")
        
        # Process JSON files to create final datasets
        for (slice_id, slice_df, slice_language), json_file in zip(slices, json_files):
            if json_file is None:
                print(f"❌ No JSON file for batch {slice_id}")
                continue
                
            print(f"\n🔄 Processing JSON file: {json_file}")
            graphs = prepare.json_process(PATHS.cpg, json_file)
            
            if graphs is None or len(graphs) == 0:
                print(f"❌ No graphs generated from {json_file}")
                continue
            
            print(f"✅ Generated {len(graphs)} graphs from {json_file}")
            
            # Apply universal index fix for proper joining
            num_original = len(slice_df)
            graphs_to_use = graphs[:num_original]  # Take first N graphs to match original
            
            # Force sequential indices for proper joining
            for i, graph in enumerate(graphs_to_use):
                graph['Index'] = i
            
            # Create dataset with CPG
            dataset = data.create_with_index(graphs_to_use, ["Index", "cpg"])
            dataset.index = list(range(len(dataset)))  # Force sequential DataFrame index
            
            # Ensure slice_df has matching index
            slice_df.index = list(range(len(slice_df)))
            
            # Join with original slice data
            final_dataset = data.inner_join_by_index(slice_df, dataset)
            
            # Fallback join if inner_join fails
            if len(final_dataset) == 0:
                print(f"⚠️  Inner join failed, using direct concatenation")
                final_dataset = pd.concat([slice_df.reset_index(drop=True), 
                                         dataset.reset_index(drop=True)], axis=1)
            
            print(f"📋 Final dataset size: {len(final_dataset)}")
            print(f"📋 Columns: {list(final_dataset.columns)}")
            
            # Save the dataset
            output_file = f"{slice_id}_{FILES.cpg}.pkl"
            data.write(final_dataset, PATHS.cpg, output_file)
            print(f"✅ Saved: {output_file}")
            
            # Show sample
            if len(final_dataset) > 0:
                sample = final_dataset.iloc[0]
                target_label = "VULNERABLE" if sample.get('target') == 1 else "SAFE"
                lang = sample.get('language', slice_language or 'N/A')
                print(f"📋 Sample: [{target_label}] Language: {lang}")
            
            del dataset, final_dataset
            gc.collect()
            
    else:
        # Original pipeline for non-language-specific datasets
        print("🔄 Processing original dataset format")
        filtered = data.apply_filter(raw, select)
        filtered = data.clean(filtered)
        data.drop(filtered, ["commit_id", "project"])
        # Use smart slicing (will fallback to original for non-multilang datasets)
        slices = data.smart_language_aware_slice(filtered, context.slice_size)
        
        # Convert to old format for backward compatibility with original pipeline
        converted_slices = []
        for slice_id, slice_df, slice_language in slices:
            converted_slices.append((slice_id, slice_df.apply(lambda x: x)))
        slices = converted_slices
        
        cpg_files = []
        for s, slice in slices:
            data.to_files(slice, PATHS.joern)
            cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, f"{s}_{FILES.cpg}")
            print("CPG file created:", cpg_file)
            cpg_files.append(cpg_file)
            print(f"Dataset {s} to cpg.")
            shutil.rmtree(PATHS.joern)
        
        json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, cpg_files)
        print("JSON files returned:", json_files)
        
        for (s, slice), json_file in zip(slices, json_files):
            graphs = prepare.json_process(PATHS.cpg, json_file)
            if graphs is None:
                print(f"Dataset chunk {s} not processed.")
                continue
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


def run_test_pipeline():
    """
    Test pipeline function that processes the mini dataset with detailed logging
    Traces the complete pipeline: JSON → CPG → tokens → embedding → input PKL
    """
    print("=" * 80)
    print("RUNNING TEST PIPELINE ON MINI DATASET")
    print("=" * 80)
    
    # Create test output directory
    test_output_dir = Path("data/test_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Stage 1: Load test dataset
    print("\n🔄 STAGE 1: Loading test dataset...")
    test_dataset_path = "data/dataset_test_mini.json"
    
    try:
        import json
        with open(test_dataset_path, 'r') as f:
            test_data = json.load(f)
        
        print(f"✓ Loaded {len(test_data)} test samples")
        print(f"Sample data structure: {list(test_data[0].keys())}")
        print(f"First sample: {test_data[0]['func'][:100]}...")
        
        # Convert to DataFrame
        test_df = pd.DataFrame(test_data)
        print(f"✓ Created DataFrame with shape: {test_df.shape}")
        print(f"Target distribution: {test_df['target'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return
    
    # Stage 2: Create CPG files
    print("\n🔄 STAGE 2: Creating CPG files...")
    
    try:
        context = configs.Create()
        
        # Create .c files for Joern
        data.to_files(test_df, PATHS.joern)
        print(f"✓ Created .c files in {PATHS.joern}")
        
        # Check created files
        c_files = list(Path(PATHS.joern).glob("*.c"))
        print(f"✓ Created {len(c_files)} .c files")
        for f in c_files[:3]:  # Show first 3
            print(f"  - {f.name}")
        
        # Generate CPG
        cpg_file = prepare.joern_parse(context.joern_cli_dir, PATHS.joern, PATHS.cpg, "test_cpg")
        print(f"✓ Generated CPG file: {cpg_file}")
        
        # Clean up joern files
        shutil.rmtree(PATHS.joern)
        
    except Exception as e:
        print(f"❌ Error creating CPG: {e}")
        return
    
    # Stage 3: Process CPG to JSON
    print("\n🔄 STAGE 3: Processing CPG to JSON...")
    
    try:
        # Create JSON from CPG
        json_files = prepare.joern_create(context.joern_cli_dir, PATHS.cpg, PATHS.cpg, [cpg_file])
        print(f"✓ Created JSON files: {json_files}")
        
        if json_files and json_files[0]:
            # Process JSON to graphs
            graphs = prepare.json_process(PATHS.cpg, json_files[0])
            print(f"✓ Processed {len(graphs) if graphs else 0} graphs from JSON")
            
            if graphs:
                print(f"Sample graph keys: {list(graphs[0].keys()) if graphs else 'None'}")
                
                # Create dataset with CPG
                dataset = data.create_with_index(graphs, ["Index", "cpg"])
                dataset = data.inner_join_by_index(test_df, dataset)
                
                print(f"✓ Created CPG dataset with {len(dataset)} samples")
                print(f"Dataset columns: {list(dataset.columns)}")
                
                # Save CPG dataset
                cpg_output_path = test_output_dir / "test_cpg.pkl"
                data.write(dataset, str(test_output_dir), "test_cpg.pkl")
                print(f"✓ Saved CPG dataset to: {cpg_output_path}")
                
            else:
                print("❌ No graphs generated from JSON")
                return
        else:
            print("❌ No JSON files created")
            return
            
    except Exception as e:
        print(f"❌ Error processing CPG to JSON: {e}")
        return
    
    # Stage 4: Tokenization
    print("\n🔄 STAGE 4: Tokenizing functions...")
    
    try:
        # Convert function dicts to code strings
        dataset['code_str'] = dataset['func'].apply(
            lambda x: x.get('function', '') if isinstance(x, dict) else str(x)
        )
        
        print(f"✓ Converted functions to code strings")
        print(f"Sample code string: {dataset['code_str'].iloc[0][:100]}...")
        
        # Tokenize
        tokens_dataset = data.tokenize(
            dataset[['code_str']].rename(columns={'code_str': 'func'})
        )
        
        # Filter out empty token lists
        tokens_dataset = tokens_dataset[tokens_dataset['tokens'].map(len) > 0]
        
        print(f"✓ Tokenized {len(tokens_dataset)} functions")
        print(f"Sample tokens: {tokens_dataset['tokens'].iloc[0][:10]}...")
        
        # Save tokens
        tokens_output_path = test_output_dir / "test_tokens.pkl"
        data.write(tokens_dataset, str(test_output_dir), "test_tokens.pkl")
        print(f"✓ Saved tokens to: {tokens_output_path}")
        
    except Exception as e:
        print(f"❌ Error in tokenization: {e}")
        return
    
    # Stage 5: Word2Vec Training and Embedding
    print("\n🔄 STAGE 5: Word2Vec training and embedding...")
    
    try:
        context = configs.Embed()
        
        # Initialize Word2Vec model
        w2vmodel = Word2Vec(**context.w2v_args)
        
        # Build vocabulary and train
        print("Building vocabulary...")
        w2vmodel.build_vocab(corpus_iterable=tokens_dataset.tokens)
        print(f"✓ Built vocabulary with {len(w2vmodel.wv.key_to_index)} words")
        
        print("Training Word2Vec model...")
        w2vmodel.train(
            corpus_iterable=tokens_dataset.tokens,
            total_examples=len(tokens_dataset),
            epochs=w2vmodel.epochs
        )
        print(f"✓ Trained Word2Vec model")
        
        # Save Word2Vec model
        w2v_output_path = test_output_dir / "test_w2v.model"
        w2vmodel.save(str(w2v_output_path))
        print(f"✓ Saved Word2Vec model to: {w2v_output_path}")
        
        # Test some embeddings
        sample_tokens = tokens_dataset['tokens'].iloc[0][:5]
        print(f"Testing embeddings for tokens: {sample_tokens}")
        for token in sample_tokens:
            if token in w2vmodel.wv:
                embedding = w2vmodel.wv[token]
                print(f"  {token}: shape={embedding.shape}, mean={embedding.mean():.3f}")
            else:
                print(f"  {token}: NOT FOUND in vocabulary")
        
    except Exception as e:
        print(f"❌ Error in Word2Vec training: {e}")
        return
    
    # Stage 6: CPG Node Processing
    print("\n🔄 STAGE 6: Processing CPG nodes...")
    
    try:
        # Ensure cpg column contains dicts
        dataset['cpg'] = dataset['cpg'].apply(lambda x: x if isinstance(x, dict) else None)
        dataset = dataset.dropna(subset=['cpg'])
        print(f"✓ Filtered to {len(dataset)} samples with valid CPG")
        
        # Parse CPG to nodes
        print("Parsing CPG to nodes...")
        dataset["nodes"] = dataset['cpg'].apply(
            lambda x: cpg.parse_to_nodes(x, context.nodes_dim) if isinstance(x, dict) else {}
        )
        
        # Remove rows with no nodes
        dataset = dataset.loc[dataset.nodes.map(len) > 0]
        print(f"✓ Parsed nodes for {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample_nodes = dataset['nodes'].iloc[0]
            print(f"Sample nodes: type={type(sample_nodes)}, count={len(sample_nodes) if hasattr(sample_nodes, '__len__') else 'N/A'}")
            
            if hasattr(sample_nodes, '__iter__') and not isinstance(sample_nodes, (str, dict)):
                first_node = list(sample_nodes)[0]
                node_dict = first_node if isinstance(first_node, dict) else first_node.__dict__
                print(f"First node keys: {list(node_dict.keys())}")
        
    except Exception as e:
        print(f"❌ Error processing CPG nodes: {e}")
        return
    
    # Stage 7: Create Input Tensors
    print("\n🔄 STAGE 7: Creating input tensors...")
    
    try:
        print("Converting nodes to input tensors...")
        dataset["input"] = dataset.apply(
            lambda row: prepare.nodes_to_input(
                row.nodes, row.target, context.nodes_dim, w2vmodel.wv, context.edge_type
            ),
            axis=1
        )
        
        print(f"✓ Created input tensors for {len(dataset)} samples")
        
        # Analyze the created inputs
        print("\n📊 INPUT TENSOR ANALYSIS:")
        for idx, row in dataset.head(3).iterrows():
            sample_input = row['input']
            print(f"\nSample {idx} (target={row['target']}):")
            
            if hasattr(sample_input, 'x'):
                x_shape = sample_input.x.shape
                x_mean = sample_input.x.mean().item()
                x_std = sample_input.x.std().item()
                x_zero_ratio = (sample_input.x == 0).float().mean().item()
                
                print(f"  Features (x): shape={x_shape}, mean={x_mean:.3f}, std={x_std:.3f}, zero_ratio={x_zero_ratio:.2%}")
                
                # Check for zero features
                if x_zero_ratio > 0.9:
                    print(f"  ⚠️  WARNING: {x_zero_ratio:.1%} of features are zero!")
                elif x_zero_ratio > 0.5:
                    print(f"  ⚠️  CAUTION: {x_zero_ratio:.1%} of features are zero")
                else:
                    print(f"  ✓ Feature quality looks good")
            
            if hasattr(sample_input, 'edge_index'):
                edge_shape = sample_input.edge_index.shape
                print(f"  Edges: shape={edge_shape}")
            
            if hasattr(sample_input, 'y'):
                print(f"  Label: {sample_input.y.item()}")
        
        # Save final input dataset
        final_output_path = test_output_dir / "test_input.pkl"
        data.write(dataset[["input", "target"]], str(test_output_dir), "test_input.pkl")
        print(f"\n✓ Saved final input dataset to: {final_output_path}")
        
    except Exception as e:
        print(f"❌ Error creating input tensors: {e}")
        return
    
    # Stage 8: Final Summary
    print("\n" + "=" * 80)
    print("TEST PIPELINE SUMMARY")
    print("=" * 80)
    
    print(f"✅ Successfully processed {len(dataset)} samples")
    print(f"📁 Output directory: {test_output_dir}")
    print(f"📊 Target distribution: {dataset['target'].value_counts().to_dict()}")
    
    # Check for feature corruption
    zero_count = 0
    normal_count = 0
    
    for _, row in dataset.iterrows():
        sample_input = row['input']
        if hasattr(sample_input, 'x'):
            zero_ratio = (sample_input.x == 0).float().mean().item()
            if zero_ratio > 0.9:
                zero_count += 1
            else:
                normal_count += 1
    
    print(f"\n🔍 FEATURE QUALITY CHECK:")
    print(f"  Normal features: {normal_count}")
    print(f"  Zero features: {zero_count}")
    
    if zero_count > 0:
        print(f"  ⚠️  {zero_count}/{len(dataset)} samples have corrupted features!")
    else:
        print(f"  ✅ All samples have normal features")
    
    print(f"\n💡 Next steps:")
    print(f"  1. Check files in: {test_output_dir}")
    print(f"  2. Compare with corrupted production data")
    print(f"  3. Identify where corruption occurs in production pipeline")


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
    parser.add_argument('-t', '--test', action='store_true', help='Run test pipeline on mini dataset')

    args = parser.parse_args()

    if args.create:
        create_task()
    if args.embed:
        embed_task()
    if args.process:
        process_task(False)
    if args.process_stopping:
        process_task(True)
    if args.test:
        run_test_pipeline()



if __name__ == "__main__":
    main()
