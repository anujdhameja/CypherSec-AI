import json

# --- Method 1: Convert a JSON string to a Python dictionary ---
# (Using the data from your example, corrected to be valid JSON)

json_string = """
{
  "project": "FFmpeg",
  "commit_id": "973b1a6b9070e2bf17d17568cbaf4043ce931f51",
  "target": 0,
  "func": "static av_codec int vcodec_init(AVCodecContext *avctx)\\n\\nVDADecoderContext *ctx = avctx->priv_data;\\n\\n struct vda_context *vda_ctx = &ctx->vda_ctx;\\n OSStatus status;\\n int ret;\\n\\n\\n ctx->h264_initialized = 0;\\n\\n\\n /* init pix_fmts of codec */\\n if (!ff_h264_vda_decoder.pix_fmts) {\\n\\n if (kCoreFoundationVersionNumber < kCoreFoundationVersionNumber10_7)\\n ff_h264_vda_decoder.pix_fmts = vda_pixfmts_prior_10_7;\\n\\n else\\n ff_h264_vda_decoder.pix_fmts = vda_pixfmts;\\n }\\n\\n\\n /* init vda */\\n\\n memset(vda_ctx, 0, sizeof(struct vda_context));\\n\\n"
}
"""

print("--- Parsing JSON from a string ---")
# Use json.loads() to load from a string
# This converts the JSON text into a Python dictionary
data_object = json.loads(json_string)

# Now you can use it like a regular Python dictionary
print(f"Project: {data_object['project']}")
print(f"Commit ID: {data_object['commit_id']}")
print(f"Target: {data_object['target']}")
print("\n")


# --- Method 2: Convert a JSON file to a Python dictionary ---
# (Assuming you saved the valid JSON above into a file named 'data.json')

import os

print("--- Parsing JSON from a file ---")
try:
    file_path = r'C:\Devign\devign\data\raw\dataset.json'
    print(f"Trying to read file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    # Read the first 1000 bytes to check the content
    with open(file_path, 'r', encoding='utf-8') as f:
        first_1000 = f.read(1000)
        print("\nFirst 1000 characters of the file:")
        print("-" * 50)
        print(first_1000)
        print("-" * 50)
    
    # Now try to parse the full JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data_from_file = json.load(f)
    
    print("\nSuccessfully parsed JSON data!")
    print(f"Data type: {type(data_from_file)}")
    
    if isinstance(data_from_file, list):
        print(f"\nFound {len(data_from_file)} items in the JSON file")
        if len(data_from_file) > 0:
            print("\nFirst item type:", type(data_from_file[0]))
            print("\nFirst item structure (first 500 chars):")
            print("-" * 50)
            print(json.dumps(data_from_file[0], indent=2)[:500] + ("..." if len(str(data_from_file[0])) > 500 else ""))
            print("-" * 50)
    elif isinstance(data_from_file, dict):
        print("\nJSON is a dictionary with keys:", list(data_from_file.keys()))
    else:
        print("\nUnexpected JSON structure:", type(data_from_file))

except FileNotFoundError:
    print("Error: 'data.json' file not found.")
    print("Please create 'data.json' with the valid JSON content to run this part.")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    print("This usually means the JSON in the file is invalid (e.g., missing quotes on keys).")