# 🔍 PIPELINE FLOW VERIFICATION

## ✅ COMPLETE FLOW CONFIRMED FOR: `python main.py -c -e`

### **📁 DATA SOURCE VERIFIED:**
- **Original Dataset**: `data/raw/dataset.json` ✅ **EXISTS**
- **Configuration**: `configs.json` points to correct paths ✅
- **NOT using test data** - Will use full production dataset ✅

### **🔄 STEP-BY-STEP FLOW:**

#### **Step 1: Create Task (`-c` flag)**
**Function**: `create_task()` in `main.py`

1. **Load Raw Data**:
   ```python
   raw = data.read(PATHS.raw, FILES.raw)  # Loads data/raw/dataset.json
   ```

2. **Apply Filters**:
   ```python
   filtered = data.apply_filter(raw, select)  # Filters for FFmpeg project
   filtered = data.clean(filtered)  # Removes duplicates
   ```

3. **Create Slices**:
   ```python
   slices = data.slice_frame(filtered, context.slice_size)  # Slice size: 100
   ```

4. **Generate CPG Files**:
   ```python
   # For each slice:
   data.to_files(slice, PATHS.joern)  # Create .c files
   cpg_file = prepare.joern_parse(...)  # Generate CPG binary
   json_files = prepare.joern_create(...)  # Create JSON from CPG
   graphs = prepare.json_process(...)  # Process JSON to graphs
   ```

5. **Save CPG Data**:
   ```python
   data.write(dataset, PATHS.cpg, f"{s}_{FILES.cpg}.pkl")  # Save to data/cpg/
   ```

#### **Step 2: Embed Task (`-e` flag)**
**Function**: `embed_task()` in `main.py`

1. **Load CPG Files**:
   ```python
   dataset_files = [f for f in os.listdir(PATHS.cpg) if f.endswith(".pkl")]
   # Loads from data/cpg/ (created in step 1)
   ```

2. **Initialize Word2Vec**:
   ```python
   w2vmodel = Word2Vec(**context.w2v_args)  # Vector size: 100
   ```

3. **For Each CPG File**:
   ```python
   cpg_dataset = pd.read_pickle(file_path)  # Load CPG data
   
   # Convert to code strings
   cpg_dataset['code_str'] = cpg_dataset['func'].apply(...)
   
   # 🎯 TOKENIZE (uses fixed tokenizer)
   tokens_dataset = data.tokenize(cpg_dataset[['code_str']])
   
   # Train Word2Vec
   w2vmodel.build_vocab(tokens_dataset.tokens)
   w2vmodel.train(tokens_dataset.tokens, ...)
   
   # Parse CPG to nodes
   cpg_dataset["nodes"] = cpg_dataset['cpg'].apply(
       lambda x: cpg.parse_to_nodes(x, context.nodes_dim)
   )
   
   # 🎯 CREATE EMBEDDINGS (uses FIXED NodesEmbedding)
   cpg_dataset["input"] = cpg_dataset.apply(
       lambda row: prepare.nodes_to_input(
           row.nodes, row.target, context.nodes_dim, w2vmodel.wv, context.edge_type
       ), axis=1
   )
   
   # Save input data
   data.write(cpg_dataset[["input", "target"]], PATHS.input, f"{file_name}_{FILES.input}")
   ```

4. **Save Word2Vec Model**:
   ```python
   w2vmodel.save(f"{PATHS.w2v}/{FILES.w2v}")  # Save to data/w2v/w2v.model
   ```

### **🎯 CRITICAL FIX POINTS:**

#### **✅ Fix Applied in `embed_task()`:**

**Line ~tokens_dataset creation**:
```python
tokens_dataset = data.tokenize(cpg_dataset[['code_str']])
# This uses the CORRECT tokenizer() function ✅
```

**Line ~nodes_to_input creation**:
```python
cpg_dataset["input"] = cpg_dataset.apply(
    lambda row: prepare.nodes_to_input(...)  # Calls NodesEmbedding ✅
)
```

#### **✅ Fix Applied in `src/prepare/embeddings.py`:**

**NodesEmbedding.embed_nodes() Method 1 (~line 378)**:
```python
# OLD (BROKEN):
tokens = self._tok_re.findall(code_text)

# NEW (FIXED):
tokens = tokenizer(code_text)  # ✅ SAME tokenizer as training
```

**NodesEmbedding.embed_nodes() Method 2 (~line 461)**:
```python
# OLD (BROKEN):
tokens = self._tok_re.findall(code_text or "")

# NEW (FIXED):
tokens = tokenizer(code_text or "")  # ✅ SAME tokenizer as training
```

### **📊 EXPECTED RESULTS:**

#### **Before Fix (Previous Runs)**:
- **Zero Features**: 40% of graphs had 100% zero features
- **Cause**: `'buffer'` → `regex` → `['buffer']` → NOT FOUND → ZERO
- **Performance**: ~51% accuracy (random)

#### **After Fix (This Run)**:
- **Zero Features**: 0% expected (all valid embeddings)
- **Cause Fixed**: `'buffer'` → `tokenizer()` → `['VAR1']` → FOUND → VALID
- **Performance**: 60-70%+ accuracy expected

### **🔍 MONITORING POINTS:**

When you run `python main.py -c -e`, watch for:

1. **During `-c` (Create)**:
   ```
   ✓ Loading data/raw/dataset.json
   ✓ Filtered X samples for FFmpeg
   ✓ Created Y CPG files
   ✓ Saved to data/cpg/*.pkl
   ```

2. **During `-e` (Embed)**:
   ```
   ✓ Processing X_cpg.pkl files
   ✓ Tokenized Y functions  
   ✓ Built vocabulary with Z words
   ✓ Training Word2Vec...
   ✓ Creating node embeddings... (should show non-zero means/stds)
   ✓ Saved to data/input/*.pkl
   ```

3. **Success Indicators**:
   ```
   ✓ Node features shape: torch.Size([N, 100])
   ✓ Feature mean: 0.XXX (NON-ZERO)
   ✓ Feature std: 0.XXX (NON-ZERO)
   ✓ Zero ratio: <10% (MUCH LOWER than before)
   ```

### **✅ VERIFICATION COMPLETE:**

- ✅ **Original dataset.json** will be used (not test data)
- ✅ **Fixed tokenization** is in place for embedding phase
- ✅ **Complete pipeline** from raw data to input tensors
- ✅ **Expected improvement** from 51% to 60-70%+ accuracy

**Ready to run: `python main.py -c -e`** 🚀