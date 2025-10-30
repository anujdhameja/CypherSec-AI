#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replica Pipeline Testing System
===============================

This script creates a comprehensive end-to-end testing system for the vulnerability detection pipeline.
It tests all supported languages (C, C++, C#, Python, Java, PHP) using replica datasets and configurations.

Usage:
    python replica_pipeline_test.py --language all
    python replica_pipeline_test.py --language c
    python replica_pipeline_test.py --create-only
    python replica_pipeline_test.py --embed-only
    python replica_pipeline_test.py --predict-only
"""

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from gensim.models.word2vec import Word2Vec
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

# Import your existing modules
import configs
import src.data as data
import src.prepare as prepare
import src.process as process
import src.utils.functions.cpg as cpg


class ReplicaConfig:
    """Replica configuration manager that mirrors the original config structure"""
    
    def __init__(self):
        # Load replica configuration
        with open('replica_configs.json', 'r') as f:
            self.config = json.load(f)
    
    def Paths(self):
        """Return paths configuration"""
        class PathsObj:
            def __init__(self, paths_dict):
                for key, value in paths_dict.items():
                    setattr(self, key, value)
        return PathsObj(self.config['paths'])
    
    def Files(self):
        """Return files configuration"""
        class FilesObj:
            def __init__(self, files_dict):
                for key, value in files_dict.items():
                    setattr(self, key, value)
            
            def get_device(self):
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return FilesObj(self.config['files'])
    
    def Create(self):
        """Return create configuration"""
        class CreateObj:
            def __init__(self, create_dict):
                for key, value in create_dict.items():
                    setattr(self, key, value)
        return CreateObj(self.config['create'])
    
    def Embed(self):
        """Return embed configuration"""
        class EmbedObj:
            def __init__(self, embed_dict):
                for key, value in embed_dict.items():
                    setattr(self, key, value)
        return EmbedObj(self.config['embed'])
    
    def Process(self):
        """Return process configuration"""
        class ProcessObj:
            def __init__(self, process_dict):
                for key, value in process_dict.items():
                    setattr(self, key, value)
        return ProcessObj(self.config['process'])
    
    def Devign(self):
        """Return devign configuration"""
        class DevignObj:
            def __init__(self, devign_dict):
                for key, value in devign_dict.items():
                    setattr(self, key, value)
        return DevignObj(self.config['devign'])


class ReplicaPipelineTester:
    """Main replica pipeline testing class"""
    
    def __init__(self):
        self.replica_config = ReplicaConfig()
        self.PATHS = self.replica_config.Paths()
        self.FILES = self.replica_config.Files()
        self.DEVICE = self.FILES.get_device()
        
        # Flags configurable from CLI
        self.use_trained = False
        self.trained_model_path = os.path.join("models", "final_model.pth")
        
        # Supported languages
        self.LANGUAGES = ['c', 'cpp', 'csharp', 'python', 'java', 'php']
        
        # Create necessary directories
        self._create_directories()
        
        print("🚀 Replica Pipeline Tester Initialized")
        print(f"📱 Device: {self.DEVICE}")
        print(f"🗂️  Languages: {', '.join(self.LANGUAGES)}")
    
    def _create_directories(self):
        """Create all necessary directories for replica testing"""
        dirs_to_create = [
            self.PATHS.cpg,
            self.PATHS.joern,
            self.PATHS.raw,
            self.PATHS.input,
            self.PATHS.model,
            self.PATHS.tokens,
            self.PATHS.w2v
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")
    
    def load_language_dataset(self, language: str) -> pd.DataFrame:
        """Load dataset for a specific language"""
        dataset_path = f"data/replica_raw/replica_dataset_{language}.json"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data_list = json.load(f)
        
        df = pd.DataFrame(data_list)
        print(f"📊 Loaded {language.upper()} dataset: {len(df)} samples")
        print(f"   Vulnerable: {df['target'].sum()}, Safe: {len(df) - df['target'].sum()}")
        
        return df
    
    def combine_datasets(self, languages: List[str]) -> pd.DataFrame:
        """Combine multiple language datasets"""
        combined_data = []
        
        for lang in languages:
            try:
                lang_df = self.load_language_dataset(lang)
                lang_df['language'] = lang
                combined_data.append(lang_df)
                print(f"✓ Added {lang.upper()} dataset")
            except FileNotFoundError as e:
                print(f"⚠️  Warning: {e}")
                continue
        
        if not combined_data:
            raise ValueError("No datasets could be loaded!")
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"\n📈 Combined Dataset Summary:")
        print(f"   Total samples: {len(combined_df)}")
        print(f"   Languages: {combined_df['language'].value_counts().to_dict()}")
        print(f"   Vulnerability distribution: {combined_df['target'].value_counts().to_dict()}")
        
        return combined_df
    
    def replica_create_task(self, languages: List[str]):
        """Replica version of create_task with multi-language support"""
        print("\n" + "="*80)
        print("🔄 REPLICA CREATE TASK - CPG GENERATION")
        print("="*80)
        
        context = self.replica_config.Create()
        
        # Load and combine datasets
        dataset = self.combine_datasets(languages)
        
        # Apply basic filtering (similar to original select function)
        dataset = dataset.loc[dataset['project'] == "replica_test"]
        
        # Filter by function length (convert to string if needed)
        dataset['func_str'] = dataset['func'].apply(
            lambda x: x.get('function', '') if isinstance(x, dict) else str(x)
        )
        len_filter = dataset.func_str.str.len() < 2000  # Slightly higher limit for test data
        dataset = dataset.loc[len_filter]
        
        print(f"📊 After filtering: {len(dataset)} samples")
        
        # Clean duplicates
        dataset = data.clean(dataset)
        print(f"📊 After cleaning: {len(dataset)} samples")
        
        # Drop unnecessary columns
        columns_to_drop = ["commit_id", "project", "func_str"]
        for col in columns_to_drop:
            if col in dataset.columns:
                dataset = dataset.drop(columns=[col])
        
        # Create slices (smaller slices for testing)
        slices = data.slice_frame(dataset, context.slice_size)
        slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]
        
        print(f"📦 Created {len(slices)} data slices")
        
        # Generate CPG files
        cpg_files = []
        for s, slice_df in slices:
            print(f"\n🔄 Processing slice {s} ({len(slice_df)} samples)...")
            
            # Create .c files for Joern
            data.to_files(slice_df, self.PATHS.joern)
            
            # Check created files
            c_files = list(Path(self.PATHS.joern).glob("*.c"))
            print(f"   Created {len(c_files)} .c files")
            
            # Generate CPG
            cpg_file = prepare.joern_parse(
                context.joern_cli_dir, 
                self.PATHS.joern, 
                self.PATHS.cpg, 
                f"replica_{s}_{self.FILES.cpg}"
            )
            print(f"   ✓ Generated CPG: {cpg_file}")
            cpg_files.append(cpg_file)
            
            # Clean up joern files
            if os.path.exists(self.PATHS.joern):
                shutil.rmtree(self.PATHS.joern)
        
        # Process CPG to JSON and create final datasets
        json_files = prepare.joern_create(context.joern_cli_dir, self.PATHS.cpg, self.PATHS.cpg, cpg_files)
        print(f"\n📄 Generated {len(json_files)} JSON files")
        
        for (s, slice_df), json_file in zip(slices, json_files):
            if not json_file:
                print(f"⚠️  No JSON file for slice {s}")
                continue
            
            # Process JSON to graphs
            cpg_root = "data/cpg"
            main_json = "0_cpg.json"
            replica_json = "replica_0_replica_cpg.json"
            if json_file == replica_json:
                full_replica_path = os.path.join(cpg_root, replica_json)
                full_main_path = os.path.join(cpg_root, main_json)
                if not os.path.exists(full_replica_path) and os.path.exists(full_main_path):
                    shutil.copy(full_main_path, full_replica_path)
                graphs = prepare.json_process(cpg_root, replica_json)
            else:
                graphs = prepare.json_process(self.PATHS.cpg, json_file)
            if graphs is None or len(graphs) == 0:
                print(f"⚠️  No graphs generated for slice {s}")
                continue
            
            print(f"   ✓ Processed {len(graphs)} graphs from slice {s}")
            
            # Create dataset with CPG
            cpg_dataset = data.create_with_index(graphs, ["Index", "cpg"])
            cpg_dataset = data.inner_join_by_index(slice_df, cpg_dataset)
            
            # Save CPG dataset
            output_filename = f"replica_{s}_{self.FILES.cpg}.pkl"
            data.write(cpg_dataset, self.PATHS.cpg, output_filename)
            print(f"   💾 Saved: {output_filename}")
            
            del cpg_dataset
            gc.collect()
        
        print(f"\n✅ CREATE TASK COMPLETED")
        print(f"📁 CPG files saved to: {self.PATHS.cpg}")
    
    def replica_embed_task(self):
        """Replica version of embed_task with enhanced debugging"""
        print("\n" + "="*80)
        print("🔄 REPLICA EMBED TASK - FEATURE EXTRACTION")
        print("="*80)
        
        context = self.replica_config.Embed()
        
        # Initialize Word2Vec model
        w2vmodel = Word2Vec(**context.w2v_args)
        w2v_init = True
        
        # Get all .pkl files from CPG folder
        dataset_files = [f for f in os.listdir(self.PATHS.cpg) if f.endswith(".pkl")]
        
        if not dataset_files:
            raise FileNotFoundError(f"No CPG datasets found in {self.PATHS.cpg}")
        
        # Sort files by their numeric prefix
        def get_file_number(filename):
            try:
                parts = filename.split('_')
                return int(parts[1]) if len(parts) > 1 else float('inf')
            except (ValueError, IndexError):
                return float('inf')
        
        dataset_files.sort(key=get_file_number)
        print(f"📂 Found {len(dataset_files)} CPG datasets: {dataset_files}")
        
        for pkl_file in dataset_files:
            print(f"\n🔄 Processing: {pkl_file}")
            file_name = pkl_file.split(".")[0]
            
            # Load CPG dataset
            file_path = os.path.join(self.PATHS.cpg, pkl_file)
            cpg_dataset = pd.read_pickle(file_path)
            print(f"   📊 Loaded {len(cpg_dataset)} samples")
            print(f"   📋 Columns: {list(cpg_dataset.columns)}")
            
            # Convert function dicts to code strings for tokenization
            cpg_dataset['code_str'] = cpg_dataset['func'].apply(
                lambda x: x.get('function', '') if isinstance(x, dict) else str(x)
            )
            
            # Tokenize
            tokens_dataset = data.tokenize(
                cpg_dataset[['code_str']].rename(columns={'code_str': 'func'})
            )
            
            # Filter out empty token lists
            tokens_dataset = tokens_dataset[tokens_dataset['tokens'].map(len) > 0]
            print(f"   🔤 Tokenized {len(tokens_dataset)} functions")
            
            if len(tokens_dataset) == 0:
                print(f"   ⚠️  No valid tokens, skipping {file_name}")
                continue
            
            # Save tokens
            data.write(tokens_dataset, self.PATHS.tokens, f"{file_name}_{self.FILES.tokens}")
            
            # Word2Vec training
            if w2v_init:
                print("   🧠 Building initial vocabulary...")
                w2vmodel.build_vocab(corpus_iterable=tokens_dataset.tokens)
                w2v_init = False
            else:
                print("   🧠 Updating vocabulary...")
                w2vmodel.build_vocab(corpus_iterable=tokens_dataset.tokens, update=True)
            
            print(f"   🏋️  Training Word2Vec on {len(tokens_dataset)} functions...")
            w2vmodel.train(
                corpus_iterable=tokens_dataset.tokens,
                total_examples=len(tokens_dataset),
                epochs=w2vmodel.epochs
            )
            
            # Process CPG nodes
            print("   🌐 Processing CPG nodes...")
            cpg_dataset['cpg'] = cpg_dataset['cpg'].apply(lambda x: x if isinstance(x, dict) else None)
            cpg_dataset = cpg_dataset.dropna(subset=['cpg'])
            
            cpg_dataset["nodes"] = cpg_dataset['cpg'].apply(
                lambda x: cpg.parse_to_nodes(x, context.nodes_dim) if isinstance(x, dict) else {}
            )
            
            # Remove rows with no nodes
            cpg_dataset = cpg_dataset.loc[cpg_dataset.nodes.map(len) > 0]
            print(f"   🔗 Parsed nodes for {len(cpg_dataset)} samples")
            
            if len(cpg_dataset) == 0:
                print(f"   ⚠️  No valid nodes, skipping input creation for {file_name}")
                continue
            
            # Create input tensors
            print("   ⚡ Creating input tensors...")
            cpg_dataset["input"] = cpg_dataset.apply(
                lambda row: prepare.nodes_to_input(
                    row.nodes, row.target, context.nodes_dim, w2vmodel.wv, context.edge_type
                ),
                axis=1
            )
            
            # Analyze input quality
            self._analyze_input_quality(cpg_dataset, file_name)
            
            # Save final input dataset
            data.drop(cpg_dataset, ["nodes"])
            output_filename = f"{file_name}_{self.FILES.input}"
            data.write(cpg_dataset[["input", "target", "language"]], self.PATHS.input, output_filename)
            print(f"   💾 Saved input dataset: {output_filename}")
            
            del cpg_dataset, tokens_dataset
            gc.collect()
        
        # Save Word2Vec model
        w2v_path = f"{self.PATHS.w2v}/{self.FILES.w2v}"
        w2vmodel.save(w2v_path)
        print(f"\n🧠 Saved Word2Vec model: {w2v_path}")
        
        # Print summary
        self._print_replica_summary()
        
        print(f"\n✅ EMBED TASK COMPLETED")
    
    def _analyze_input_quality(self, dataset: pd.DataFrame, file_name: str):
        """Analyze the quality of input tensors"""
        print(f"\n   📊 INPUT QUALITY ANALYSIS - {file_name}")
        
        zero_count = 0
        normal_count = 0
        
        for idx, row in dataset.head(3).iterrows():
            sample_input = row['input']
            
            if hasattr(sample_input, 'x'):
                x_shape = sample_input.x.shape
                x_mean = sample_input.x.mean().item()
                x_std = sample_input.x.std().item()
                x_zero_ratio = (sample_input.x == 0).float().mean().item()
                
                print(f"      Sample {idx} (target={row['target']}):")
                print(f"        Features: shape={x_shape}, mean={x_mean:.3f}, std={x_std:.3f}")
                print(f"        Zero ratio: {x_zero_ratio:.2%}")
                
                if x_zero_ratio > 0.9:
                    zero_count += 1
                    print(f"        ⚠️  HIGH zero ratio!")
                else:
                    normal_count += 1
                    print(f"        ✓ Normal features")
            
            if hasattr(sample_input, 'edge_index'):
                edge_shape = sample_input.edge_index.shape
                print(f"        Edges: shape={edge_shape}")
        
        print(f"   📈 Quality Summary: {normal_count} normal, {zero_count} corrupted")
    
    def _print_replica_summary(self):
        """Print summary of all processed replica datasets"""
        print("\n" + "="*80)
        print("📊 REPLICA DATASET SUMMARY")
        print("="*80)
        
        input_files = list(Path(self.PATHS.input).glob("*_replica_input.pkl"))
        
        if not input_files:
            print("❌ No processed datasets found!")
            return
        
        summary_data = []
        
        for file in input_files:
            try:
                df = pd.read_pickle(file)
                
                total = len(df)
                vuln = df['target'].sum()
                non_vuln = total - vuln
                
                # Get sample info
                sample = df.iloc[0]['input']
                node_shape = sample.x.shape if hasattr(sample, 'x') else "N/A"
                edge_shape = sample.edge_index.shape if hasattr(sample, 'edge_index') else "N/A"
                
                # Language distribution
                lang_dist = df['language'].value_counts().to_dict() if 'language' in df.columns else {}
                
                summary_data.append({
                    'File': file.name,
                    'Samples': total,
                    'Vulnerable': f"{vuln} ({vuln/total:.1%})",
                    'Safe': f"{non_vuln} ({non_vuln/total:.1%})",
                    'Node Shape': str(node_shape),
                    'Edge Shape': str(edge_shape),
                    'Languages': str(lang_dist)
                })
                
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False))
        
        print("="*80)
    
    def replica_predict_task(self):
        """Test prediction on replica datasets"""
        print("\n" + "="*80)
        print("🔄 REPLICA PREDICTION TASK")
        print("="*80)
        
        # Load input datasets
        input_files = list(Path(self.PATHS.input).glob("*_replica_input.pkl"))
        
        if not input_files:
            raise FileNotFoundError(f"No input datasets found in {self.PATHS.input}")
        
        # Combine all input datasets
        all_datasets = []
        for file in input_files:
            df = pd.read_pickle(file)
            all_datasets.append(df)
            print(f"📂 Loaded: {file.name} ({len(df)} samples)")
        
        combined_dataset = pd.concat(all_datasets, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_dataset)} samples")
        
        # Balance dataset by target label for fair quick test
        if 'target' in combined_dataset.columns:
            counts = combined_dataset['target'].value_counts().to_dict()
            if 0 in counts and 1 in counts and counts[0] != counts[1]:
                n = min(counts[0], counts[1])
                balanced = pd.concat([
                    combined_dataset[combined_dataset['target'] == 0].sample(n=n, random_state=42),
                    combined_dataset[combined_dataset['target'] == 1].sample(n=n, random_state=42)
                ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
                print(f"✅ Balanced dataset: {len(balanced)} samples (n0=n1={n})")
                combined_dataset = balanced
            else:
                print("⚠️ Skipping balance: labels missing or already balanced")
        else:
            print("⚠️ 'target' column not found; cannot balance")
        
        # Create DataLoader
        context = self.replica_config.Process()
        
        # Simple train/test split for testing
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(combined_dataset, test_size=0.3, random_state=42)
        
        print(f"🔄 Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Create data loaders
        train_loader = DataLoader([row['input'] for _, row in train_data.iterrows()], 
                                batch_size=context.batch_size, shuffle=True)
        test_loader = DataLoader([row['input'] for _, row in test_data.iterrows()], 
                               batch_size=context.batch_size, shuffle=False)
        
        print(f"📦 Created data loaders")
        
        # Load or create model
        model_path = f"{self.PATHS.model}/{self.FILES.model}"
        
        # For testing, create a simple model
        from src.process.model import create_devign_model
        
        model = create_devign_model(
            input_dim=205,  # nodes_dim from config
            output_dim=2,   # binary classification
            model_type='simple',  # Use simpler model for testing
            hidden_dim=128,
            num_steps=4,
            dropout=0.3
        )
        
        # Optionally load trained weights into the simple model (best-effort)
        if getattr(self, 'use_trained', False):
            try:
                if os.path.exists(self.trained_model_path):
                    state = torch.load(self.trained_model_path, map_location=self.DEVICE)
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    print(f"🧠 Loaded trained weights from {self.trained_model_path} (strict=False)")
                    if missing:
                        print(f"   ⚠️ Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"   ⚠️ Unexpected keys: {len(unexpected)}")
                else:
                    print(f"⚠️ Trained model not found at {self.trained_model_path}; continuing with random init")
            except Exception as e:
                print(f"⚠️ Could not load trained weights: {e}; continuing with random init")
        
        model = model.to(self.DEVICE)
        print(f"🤖 Created model on {self.DEVICE}")
        
        # Test forward pass
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 3:  # Test only first 3 batches
                    break
                
                batch = batch.to(self.DEVICE)
                outputs = model(batch)
                predictions = F.softmax(outputs, dim=1)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend([row['target'] for _, row in test_data.iloc[i*context.batch_size:(i+1)*context.batch_size].iterrows()])
                
                print(f"   Batch {i+1}: {len(batch)} samples, output shape: {outputs.shape}")
        
        # Analyze predictions
        self._analyze_predictions(test_predictions, test_targets, test_data.head(len(test_predictions)))
        
        print(f"\n✅ PREDICTION TASK COMPLETED")
    
    def _analyze_predictions(self, predictions, targets, sample_data):
        """Analyze model predictions"""
        print(f"\n📊 PREDICTION ANALYSIS")
        print(f"   Samples analyzed: {len(predictions)}")
        
        import numpy as np
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Convert probabilities to binary predictions
        pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(pred_classes == targets)
        print(f"   Accuracy: {accuracy:.2%}")
        
        # Show sample predictions
        print(f"\n📋 SAMPLE PREDICTIONS:")
        for i in range(min(5, len(predictions))):
            prob_vuln = predictions[i][1]  # Probability of vulnerability
            pred_class = pred_classes[i]
            actual = targets[i]
            language = sample_data.iloc[i]['language'] if 'language' in sample_data.columns else 'unknown'
            
            status = "✓" if pred_class == actual else "✗"
            print(f"   {status} Sample {i+1} ({language.upper()}): "
                  f"Pred={pred_class} (prob={prob_vuln:.3f}), Actual={actual}")
        
        # Language-wise analysis
        if 'language' in sample_data.columns:
            print(f"\n🌐 LANGUAGE-WISE ANALYSIS:")
            for lang in sample_data['language'].unique():
                lang_mask = sample_data['language'] == lang
                lang_indices = np.where(lang_mask)[0]
                
                if len(lang_indices) > 0:
                    lang_preds = pred_classes[lang_indices]
                    lang_targets = targets[lang_indices]
                    lang_acc = np.mean(lang_preds == lang_targets)
                    
                    print(f"   {lang.upper()}: {len(lang_indices)} samples, accuracy: {lang_acc:.2%}")
    
    def run_full_pipeline(self, languages: List[str]):
        """Run the complete replica pipeline"""
        print("\n" + "🚀"*40)
        print("STARTING FULL REPLICA PIPELINE TEST")
        print("🚀"*40)
        
        try:
            # Step 1: Create CPG
            self.replica_create_task(languages)
            
            # Step 2: Embed and extract features
            self.replica_embed_task()
            
            # Step 3: Test predictions
            self.replica_predict_task()
            
            print("\n" + "✅"*40)
            print("REPLICA PIPELINE TEST COMPLETED SUCCESSFULLY!")
            print("✅"*40)
            
            # Final summary
            self._print_final_summary()
            
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_final_summary(self):
        """Print final test summary"""
        print(f"\n📋 FINAL TEST SUMMARY")
        print(f"="*50)
        
        # Check all output directories
        dirs_to_check = [
            (self.PATHS.cpg, "CPG files"),
            (self.PATHS.tokens, "Token files"),
            (self.PATHS.input, "Input files"),
            (self.PATHS.w2v, "Word2Vec model")
        ]
        
        for dir_path, description in dirs_to_check:
            if os.path.exists(dir_path):
                files = list(Path(dir_path).glob("*"))
                print(f"📁 {description}: {len(files)} files in {dir_path}")
            else:
                print(f"❌ {description}: Directory not found - {dir_path}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Review generated files in replica directories")
        print(f"   2. Compare with production pipeline outputs")
        print(f"   3. Test with larger datasets using same format")
        print(f"   4. Deploy to production with confidence!")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Replica Pipeline Testing System")
    parser.add_argument('--language', '-l', 
                       choices=['all'] + ['c', 'cpp', 'csharp', 'python', 'java', 'php'],
                       default='all',
                       help='Language(s) to test')
    parser.add_argument('--create-only', action='store_true',
                       help='Run only CPG creation step')
    parser.add_argument('--embed-only', action='store_true',
                       help='Run only embedding step')
    parser.add_argument('--predict-only', action='store_true',
                       help='Run only prediction step')
    parser.add_argument('--use-trained', action='store_true',
                       help='Use trained model weights (best-effort) during replica prediction')
    parser.add_argument('--model-path', type=str, default='models/final_model.pth',
                       help='Path to trained model weights (default: models/final_model.pth)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ReplicaPipelineTester()
    
    # Apply CLI flags
    tester.use_trained = args.use_trained
    tester.trained_model_path = args.model_path
    
    # Determine languages to test
    if args.language == 'all':
        languages = tester.LANGUAGES
    else:
        languages = [args.language]
    
    print(f"🎯 Testing languages: {', '.join(lang.upper() for lang in languages)}")
    
    try:
        if args.create_only:
            tester.replica_create_task(languages)
        elif args.embed_only:
            tester.replica_embed_task()
        elif args.predict_only:
            tester.replica_predict_task()
        else:
            tester.run_full_pipeline(languages)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()