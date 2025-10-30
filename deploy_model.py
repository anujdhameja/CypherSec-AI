#!/usr/bin/env python3
"""
Model Deployment Script for Vulnerability Detection
Provides REST API and batch processing capabilities
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from flask import Flask, request, jsonify
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append('src')
import configs
from src.process.model import create_devign_model
from src.prepar