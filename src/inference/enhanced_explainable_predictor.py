#!/usr/bin/env python3
"""
Enhanced Explainable Predictor with Line-Level Vulnerability Detection

This predictor uses the EnhancedAttentionDevignModel to provide:
- Multi-head attention analysis
- Line-level vulnerability detection
- Vulnerability-aware explanations
- Improved accuracy for dangerous code patterns

Key improvements over basic explainable predictor:
- Better detection of strcpy, printf, sprintf vulnerabilities
- Multi-head attention for diverse pattern recognition
- Line number mapping and tracking
- Enhanced risk assessment
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.process.enhanced_attention_model import EnhancedAttentionDevignModel
from src.process.vulnerability_pattern_detector import VulnerabilityPatternDetector
from torch_geometric.data import Data
from gensim.models import Word2Vec


class EnhancedExplainablePredictor:
    """
    Enhanced Explainable Vulnerability Predictor with Line-Level Detection
    
    This predictor provides detailed line-by-line vulnerability analysis
    using multi-head attention and vulnerability-aware detection.
    """
    
    def __init__(self, model_path='models/enhanced_attention_model.pth', 
                 w2v_path='data/w2v/w2v.model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.w2v_path = w2v_path
        
        print("="*80)
        print("ENHANCED EXPLAINABLE VULNERABILITY PREDICTOR INITIALIZATION")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model path: {os.path.abspath(model_path)}")
        print(f"Word2Vec path: {os.path.abspath(w2v_path)}")
        
        # Load enhanced attention model
        print(f"\n🔧 Loading enhanced attention model...")
        self.model = EnhancedAttentionDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            num_attention_heads=4,  # Multi-head attention
            dropout=0.2,
            pooling='mean_max'
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Enhanced attention model weights loaded successfully")
        else:
            print(f"❌ Enhanced model file not found: {model_path}")
            print(f"💡 Run upgrade_to_enhanced_attention.py first")
            raise FileNotFoundError(f"Enhanced model file not found: {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load Word2Vec model
        print(f"\n📚 Loading Word2Vec model...")
        if os.path.exists(w2v_path):
            self.w2v = Word2Vec.load(w2v_path)
            print(f"✅ Word2Vec loaded (vocabulary: {len(self.w2v.wv)} words)")
        else:
            print(f"❌ Word2Vec file not found: {w2v_path}")
            raise FileNotFoundError(f"Word2Vec file not found: {w2v_path}")
        
        print(f"\n🎯 Enhanced explainable predictor ready!")
        print("="*80 + "\n")
    
    def predict_with_line_level_analysis(self, graph_data: Data, node_labels=None, 
                                       node_to_line_mapping=None, source_code=None, top_k=15) -> dict:
        """
        Predict vulnerability with detailed line-level analysis
        
        Args:
            graph_data: PyTorch Geometric Data object
            node_labels: Optional list of node descriptions/code lines
            node_to_line_mapping: Optional dict mapping node_idx -> line_number
            source_code: Optional source code string for pattern detection
            top_k: Number of top attention nodes to analyze
        
        Returns:
            dict: Comprehensive prediction result with line-level analysis
        """
        # Ensure graph is on correct device
        graph_data = graph_data.to(self.device)
        
        # Add batch dimension if not present
        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get line-level vulnerability analysis (KEY IMPROVEMENT)
            vulnerability_analysis = self.model.get_line_level_vulnerabilities(
                graph_data, 
                node_to_line_mapping=node_to_line_mapping,
                threshold=0.05,  # Lower threshold for more comprehensive analysis
                top_k=top_k
            )
            
            # Get basic prediction info
            pred = vulnerability_analysis['prediction']['class']
            confidence = vulnerability_analysis['prediction']['confidence']
            is_vulnerable = vulnerability_analysis['prediction']['is_vulnerable']
            
            # Pattern-based analysis (NEW ENHANCEMENT)
            pattern_analysis = None
            if source_code:
                pattern_analysis = self._analyze_vulnerability_patterns(
                    source_code, vulnerability_analysis, node_to_line_mapping
                )
            
            # Enhanced explanation generation
            explanation = self._generate_enhanced_explanation(
                vulnerability_analysis, node_labels, is_vulnerable, confidence, pattern_analysis
            )
            
            # Enhanced recommendations
            recommendations = self._generate_enhanced_recommendations(
                vulnerability_analysis, is_vulnerable, pattern_analysis
            )
            
            # Risk assessment with line-level context
            risk_assessment = self._assess_line_level_risk(vulnerability_analysis, pattern_analysis)
        
        # Combine all analysis
        result = {
            # Basic prediction info
            'is_vulnerable': is_vulnerable,
            'confidence': confidence,
            'prediction_class': pred,
            
            # Line-level analysis (KEY FEATURE)
            'vulnerable_lines': vulnerability_analysis['vulnerable_lines'],
            'line_level_risks': self._categorize_line_risks(vulnerability_analysis['vulnerable_lines']),
            
            # Enhanced attention analysis
            'attention_stats': vulnerability_analysis['attention_stats'],
            'multi_head_analysis': self._analyze_attention_patterns(vulnerability_analysis),
            
            # Risk and explanations
            'risk_level': risk_assessment['overall_risk'],
            'risk_factors': risk_assessment['risk_factors'],
            'explanation': explanation,
            'recommendations': recommendations,
            
            # Graph info
            'graph_info': {
                'num_nodes': graph_data.x.size(0),
                'num_edges': graph_data.edge_index.size(1) if graph_data.edge_index.numel() > 0 else 0,
                'node_labels': node_labels,
                'node_to_line_mapping': node_to_line_mapping
            }
        }
        
        return result
    
    def _analyze_vulnerability_patterns(self, source_code, vulnerability_analysis, node_to_line_mapping):
        """Analyze vulnerability patterns using pattern detector (NEW ENHANCEMENT)"""
        
        # Get pattern-based annotations
        pattern_annotations = VulnerabilityPatternDetector.annotate_vulnerable_lines(source_code)
        
        # Compare with attention-based detection
        vulnerable_lines = vulnerability_analysis['vulnerable_lines']
        attention_lines = {line['line_number']: line['attention_score'] for line in vulnerable_lines}
        
        # Find overlaps and mismatches
        pattern_lines = set(pattern_annotations.keys())
        attention_line_nums = set(attention_lines.keys())
        
        overlap = pattern_lines & attention_line_nums
        pattern_only = pattern_lines - attention_line_nums
        attention_only = attention_line_nums - pattern_lines
        
        # Generate enhanced explanations for detected lines
        enhanced_explanations = {}
        for line_num in pattern_lines | attention_line_nums:
            attention_score = attention_lines.get(line_num, 0.0)
            pattern_score = pattern_annotations.get(line_num, 0.0)
            
            # Get line content
            lines = source_code.split('\n')
            line_content = lines[line_num] if line_num < len(lines) else ""
            
            # Generate explanation
            explanation = VulnerabilityPatternDetector.explain_vulnerability_pattern(
                line_content, attention_score
            )
            
            enhanced_explanations[line_num] = {
                'line_content': line_content.strip(),
                'pattern_score': pattern_score,
                'attention_score': attention_score,
                'explanation': explanation,
                'detection_method': self._get_detection_method(line_num, overlap, pattern_only, attention_only)
            }
        
        return {
            'pattern_annotations': pattern_annotations,
            'overlap_lines': overlap,
            'pattern_only_lines': pattern_only,
            'attention_only_lines': attention_only,
            'enhanced_explanations': enhanced_explanations,
            'agreement_rate': len(overlap) / len(pattern_lines | attention_line_nums) if pattern_lines | attention_line_nums else 0
        }
    
    def _get_detection_method(self, line_num, overlap, pattern_only, attention_only):
        """Determine how a line was detected"""
        if line_num in overlap:
            return "BOTH"  # Both pattern and attention detected
        elif line_num in pattern_only:
            return "PATTERN_ONLY"  # Only pattern matching detected
        elif line_num in attention_only:
            return "ATTENTION_ONLY"  # Only attention detected
        else:
            return "UNKNOWN"
    
    def _generate_enhanced_explanation(self, vulnerability_analysis, node_labels, is_vulnerable, confidence, pattern_analysis=None):
        """Generate enhanced human-readable explanation with line-level details"""
        
        explanation_parts = []
        
        # Main prediction
        if is_vulnerable:
            explanation_parts.append(f"🚨 The enhanced model predicts this code is VULNERABLE with {confidence:.1%} confidence.")
        else:
            explanation_parts.append(f"✅ The enhanced model predicts this code is SAFE with {confidence:.1%} confidence.")
        
        # Pattern analysis integration (NEW ENHANCEMENT)
        if pattern_analysis:
            overlap_lines = pattern_analysis['overlap_lines']
            pattern_only = pattern_analysis['pattern_only_lines']
            attention_only = pattern_analysis['attention_only_lines']
            agreement_rate = pattern_analysis['agreement_rate']
            
            explanation_parts.append(f"🔍 Pattern Analysis: {agreement_rate:.1%} agreement between attention and pattern detection")
            
            if overlap_lines:
                explanation_parts.append(f"✅ {len(overlap_lines)} lines detected by BOTH attention and pattern matching")
            
            if pattern_only:
                explanation_parts.append(f"📋 {len(pattern_only)} lines detected by pattern matching ONLY")
            
            if attention_only:
                explanation_parts.append(f"🧠 {len(attention_only)} lines detected by attention mechanism ONLY")
        
        # Line-level analysis
        vulnerable_lines = vulnerability_analysis['vulnerable_lines']
        stats = vulnerability_analysis['attention_stats']
        
        if len(vulnerable_lines) > 0:
            high_risk_lines = [line for line in vulnerable_lines if line['risk_level'] == 'HIGH']
            medium_risk_lines = [line for line in vulnerable_lines if line['risk_level'] == 'MEDIUM']
            
            if high_risk_lines:
                explanation_parts.append(f"🔴 Found {len(high_risk_lines)} HIGH RISK lines requiring immediate attention.")
                
                # Enhanced pattern-based identification
                if pattern_analysis:
                    for line in high_risk_lines[:3]:
                        line_num = line['line_number']
                        if line_num in pattern_analysis['enhanced_explanations']:
                            enhanced_exp = pattern_analysis['enhanced_explanations'][line_num]
                            explanation_parts.append(f"   • Line {line_num}: {enhanced_exp['explanation']}")
                
                # Fallback to original pattern detection
                elif node_labels:
                    dangerous_patterns = []
                    for line in high_risk_lines[:3]:  # Top 3 high risk lines
                        line_idx = line['line_number'] if isinstance(line['line_number'], int) else line['node_index']
                        if line_idx < len(node_labels):
                            code_text = node_labels[line_idx].lower()
                            if 'strcpy' in code_text:
                                dangerous_patterns.append(f"strcpy() without bounds checking (Line {line_idx})")
                            elif 'strcat' in code_text:
                                dangerous_patterns.append(f"strcat() without bounds checking (Line {line_idx})")
                            elif 'sprintf' in code_text:
                                dangerous_patterns.append(f"sprintf() without bounds checking (Line {line_idx})")
                            elif 'printf' in code_text and '%s' not in code_text:
                                dangerous_patterns.append(f"printf() format string vulnerability (Line {line_idx})")
                    
                    if dangerous_patterns:
                        explanation_parts.append("🎯 Detected dangerous patterns:")
                        for pattern in dangerous_patterns:
                            explanation_parts.append(f"   • {pattern}")
            
            if medium_risk_lines:
                explanation_parts.append(f"🟡 Found {len(medium_risk_lines)} MEDIUM RISK lines that should be reviewed.")
        
        # Attention distribution analysis
        if stats['num_high_attention'] > 0:
            explanation_parts.append(f"🔍 Multi-head attention focused on {stats['num_high_attention']} critical code areas.")
        
        if stats['max'] > 0.2:
            explanation_parts.append(f"⚠️  Peak attention score: {stats['max']:.3f} - indicates strong focus on specific vulnerability patterns.")
        
        return "\n".join(explanation_parts)
    
    def _generate_enhanced_recommendations(self, vulnerability_analysis, is_vulnerable, pattern_analysis=None):
        """Generate enhanced actionable recommendations"""
        
        recommendations = []
        vulnerable_lines = vulnerability_analysis['vulnerable_lines']
        stats = vulnerability_analysis['attention_stats']
        
        if is_vulnerable:
            high_risk_lines = [line for line in vulnerable_lines if line['risk_level'] == 'HIGH']
            
            if len(high_risk_lines) >= 3:
                recommendations.append("🚨 CRITICAL: Multiple high-risk vulnerabilities detected!")
                recommendations.append("🔥 IMMEDIATE ACTION: Stop deployment and conduct security review")
            elif len(high_risk_lines) >= 1:
                recommendations.append("⚠️  HIGH PRIORITY: Security vulnerabilities require immediate attention")
            
            # Specific recommendations based on line analysis
            recommendations.append("🎯 FOCUS AREAS:")
            for i, line in enumerate(high_risk_lines[:3]):  # Top 3 high risk
                recommendations.append(f"   #{i+1}: Line {line['line_number']} (attention: {line['attention_score']:.3f})")
            
            # Pattern-specific recommendations
            recommendations.append("🛡️  SECURITY FIXES:")
            recommendations.append("   • Replace strcpy() with strncpy() and add bounds checking")
            recommendations.append("   • Replace sprintf() with snprintf() with size limits")
            recommendations.append("   • Use printf(\"%s\", buffer) instead of printf(buffer)")
            recommendations.append("   • Add input validation and sanitization")
            
            if stats['max'] > 0.25:
                recommendations.append("🔍 DEEP ANALYSIS: Extremely high attention detected - manual code review essential")
        
        else:
            recommendations.append("✅ Code appears safe based on enhanced analysis")
            recommendations.append("📝 Continue following secure coding practices")
            
            if stats['max'] > 0.15:
                recommendations.append("🔍 Some areas show elevated attention - consider periodic review")
            
            if len(vulnerable_lines) > 0:
                recommendations.append("📊 Monitor flagged areas in future code changes")
        
        return recommendations
    
    def _assess_line_level_risk(self, vulnerability_analysis, pattern_analysis=None):
        """Assess overall risk based on line-level analysis"""
        
        vulnerable_lines = vulnerability_analysis['vulnerable_lines']
        stats = vulnerability_analysis['attention_stats']
        is_vulnerable = vulnerability_analysis['prediction']['is_vulnerable']
        
        risk_factors = []
        
        # Count risk levels
        high_risk_count = sum(1 for line in vulnerable_lines if line['risk_level'] == 'HIGH')
        medium_risk_count = sum(1 for line in vulnerable_lines if line['risk_level'] == 'MEDIUM')
        
        # Determine overall risk
        if is_vulnerable and high_risk_count >= 3:
            overall_risk = "CRITICAL"
            risk_factors.append(f"Multiple high-risk lines detected ({high_risk_count})")
        elif is_vulnerable and high_risk_count >= 1:
            overall_risk = "HIGH"
            risk_factors.append(f"High-risk vulnerability patterns found")
        elif is_vulnerable and medium_risk_count >= 2:
            overall_risk = "MEDIUM"
            risk_factors.append(f"Multiple medium-risk areas identified")
        elif stats['max'] > 0.2:
            overall_risk = "MEDIUM"
            risk_factors.append(f"Elevated attention on specific code areas")
        else:
            overall_risk = "LOW"
            risk_factors.append("No significant vulnerability patterns detected")
        
        # Additional risk factors
        if stats['num_high_attention'] > 2:
            risk_factors.append(f"Multiple areas of high attention ({stats['num_high_attention']})")
        
        if len(vulnerable_lines) > 10:
            risk_factors.append("Large number of flagged code lines")
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'high_risk_lines': high_risk_count,
            'medium_risk_lines': medium_risk_count
        }
    
    def _categorize_line_risks(self, vulnerable_lines):
        """Categorize lines by risk level for easy analysis"""
        
        categorized = {
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        for line in vulnerable_lines:
            risk_level = line['risk_level']
            categorized[risk_level].append(line)
        
        return categorized
    
    def _analyze_attention_patterns(self, vulnerability_analysis):
        """Analyze multi-head attention patterns"""
        
        stats = vulnerability_analysis['attention_stats']
        vulnerable_lines = vulnerability_analysis['vulnerable_lines']
        
        # Attention distribution analysis
        attention_distribution = {
            'concentrated': stats['max'] > 0.2,  # High peak attention
            'distributed': stats['std'] < 0.05,  # Low variance
            'focused_lines': len([line for line in vulnerable_lines if line['attention_score'] > 0.15])
        }
        
        # Pattern detection
        patterns = {
            'buffer_overflow_focus': any('strcpy' in str(line.get('code', '')) or 
                                       'strcat' in str(line.get('code', '')) 
                                       for line in vulnerable_lines[:3]),
            'format_string_focus': any('printf' in str(line.get('code', '')) 
                                     for line in vulnerable_lines[:3]),
            'memory_management_focus': any('malloc' in str(line.get('code', '')) or 
                                         'free' in str(line.get('code', ''))
                                         for line in vulnerable_lines[:3])
        }
        
        return {
            'attention_distribution': attention_distribution,
            'detected_patterns': patterns,
            'focus_quality': 'HIGH' if stats['max'] > 0.2 else 'MEDIUM' if stats['max'] > 0.1 else 'LOW'
        }
    
    def print_enhanced_explanation(self, result: dict, title: str = ""):
        """Print comprehensive enhanced explanation"""
        
        print("\n" + "="*90)
        print("🔍 ENHANCED EXPLAINABLE VULNERABILITY ANALYSIS")
        print("="*90)
        
        if title:
            print(f"📄 Analysis: {title}")
        
        # Basic prediction with enhanced confidence
        status = "🚨 VULNERABLE" if result['is_vulnerable'] else "✅ SAFE"
        print(f"\n🎯 PREDICTION: {status} ({result['confidence']:.1%} confidence)")
        
        # Enhanced risk level with factors
        risk_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        risk_emoji = risk_colors.get(result['risk_level'], "⚪")
        print(f"🎯 RISK LEVEL: {risk_emoji} {result['risk_level']}")
        
        if result['risk_factors']:
            print(f"📊 Risk Factors:")
            for factor in result['risk_factors']:
                print(f"   • {factor}")
        
        # Line-level vulnerability analysis (KEY FEATURE)
        if result['vulnerable_lines']:
            print(f"\n🎯 LINE-LEVEL VULNERABILITY ANALYSIS:")
            
            line_risks = result['line_level_risks']
            
            if line_risks['HIGH']:
                print(f"\n🔴 HIGH RISK LINES ({len(line_risks['HIGH'])}):")
                for line in line_risks['HIGH'][:5]:  # Top 5 high risk
                    print(f"   Line {line['line_number']}: Attention {line['attention_score']:.3f} "
                          f"(Rank #{line['rank']}, {line['relative_importance']:.1%} importance)")
            
            if line_risks['MEDIUM']:
                print(f"\n🟡 MEDIUM RISK LINES ({len(line_risks['MEDIUM'])}):")
                for line in line_risks['MEDIUM'][:3]:  # Top 3 medium risk
                    print(f"   Line {line['line_number']}: Attention {line['attention_score']:.3f} "
                          f"(Rank #{line['rank']})")
        
        # Multi-head attention analysis
        multi_head = result['multi_head_analysis']
        print(f"\n📈 MULTI-HEAD ATTENTION ANALYSIS:")
        print(f"   Focus Quality: {multi_head['focus_quality']}")
        print(f"   Attention Distribution: {'Concentrated' if multi_head['attention_distribution']['concentrated'] else 'Distributed'}")
        print(f"   Focused Lines: {multi_head['attention_distribution']['focused_lines']}")
        
        # Enhanced explanation
        print(f"\n💡 ENHANCED EXPLANATION:")
        for line in result['explanation'].split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Enhanced recommendations
        print(f"\n💡 ENHANCED RECOMMENDATIONS:")
        for rec in result['recommendations']:
            print(f"   {rec}")
        
        # Graph details
        graph_info = result['graph_info']
        print(f"\n📊 GRAPH DETAILS:")
        print(f"   Nodes: {graph_info['num_nodes']}")
        print(f"   Edges: {graph_info['num_edges']}")
        print(f"   Lines Analyzed: {len(result['vulnerable_lines'])}")
        
        print("="*90 + "\n")
    
    def visualize_line_level_attention(self, result: dict, save_path=None):
        """Create enhanced visualization with line-level details"""
        
        print("\n" + "="*70)
        print("🎨 ENHANCED ATTENTION VISUALIZATION")
        print("="*70)
        
        vulnerable_lines = result['vulnerable_lines']
        node_labels = result['graph_info']['node_labels']
        
        if not vulnerable_lines:
            print("No attention data available for visualization")
            return
        
        # Sort by attention score for visualization
        sorted_lines = sorted(vulnerable_lines, key=lambda x: x['attention_score'], reverse=True)
        
        max_attention = max(line['attention_score'] for line in sorted_lines)
        
        print(f"📊 Top {len(sorted_lines)} Lines by Attention Score:")
        print(f"{'Rank':<4} {'Line':<4} {'Risk':<6} {'Attention':<10} {'Bar':<25} {'Code'}")
        print("-" * 70)
        
        for i, line in enumerate(sorted_lines):
            # Create attention bar
            bar_length = int((line['attention_score'] / max_attention) * 20) if max_attention > 0 else 0
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            # Get code text if available
            line_idx = line['line_number'] if isinstance(line['line_number'], int) else line['node_index']
            if node_labels and line_idx < len(node_labels):
                code_text = node_labels[line_idx][:40] + "..." if len(node_labels[line_idx]) > 40 else node_labels[line_idx]
            else:
                code_text = f"Node_{line_idx}"
            
            # Risk level emoji
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[line['risk_level']]
            
            print(f"#{i+1:<3} {line_idx:<4} {risk_emoji}{line['risk_level']:<5} "
                  f"{line['attention_score']:.3f}     |{bar}| {code_text}")
        
        print("-" * 70)
        print("Legend: 🔴 High Risk  🟡 Medium Risk  🟢 Low Risk")
        print("="*70 + "\n")
        
        if save_path:
            # Save detailed visualization to file
            with open(save_path, 'w') as f:
                f.write("Enhanced Line-Level Attention Analysis\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"Prediction: {'VULNERABLE' if result['is_vulnerable'] else 'SAFE'}\n")
                f.write(f"Confidence: {result['confidence']:.2%}\n")
                f.write(f"Risk Level: {result['risk_level']}\n\n")
                
                f.write("Line-Level Analysis:\n")
                for line in sorted_lines:
                    f.write(f"Line {line['line_number']}: {line['attention_score']:.3f} ({line['risk_level']})\n")
                
            print(f"💾 Enhanced visualization saved to: {save_path}")


if __name__ == "__main__":
    # Test the enhanced explainable predictor
    print("Enhanced Explainable Predictor Test")
    print("="*50)
    
    try:
        predictor = EnhancedExplainablePredictor('models/enhanced_attention_model.pth')
        print("✅ Enhanced predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Run upgrade_to_enhanced_attention.py first")