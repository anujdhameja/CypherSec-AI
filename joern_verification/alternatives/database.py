"""
Database of alternative CPG/AST generation tools for various programming languages.
"""

from typing import Dict, List, Optional
from .models import (
    AlternativeTool, LanguageAlternatives, InstallationInstruction, 
    UsageExample, ToolComparison, InstallationMethod, ToolCapability
)


class AlternativeToolDatabase:
    """Database containing alternative tools for CPG/AST generation."""
    
    def __init__(self):
        """Initialize the alternative tool database."""
        self._tools: Dict[str, AlternativeTool] = {}
        self._language_alternatives: Dict[str, LanguageAlternatives] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with predefined alternative tools."""
        # Tree-sitter - Universal parser generator
        tree_sitter = AlternativeTool(
            name="Tree-sitter",
            description="A parser generator tool and incremental parsing library for building syntax trees",
            supported_languages=[
                "python", "javascript", "java", "c", "cpp", "csharp", "go", 
                "rust", "ruby", "php", "swift", "kotlin", "typescript"
            ],
            capabilities=[
                ToolCapability.AST_GENERATION,
                ToolCapability.SYNTAX_PARSING
            ],
            installation=[
                InstallationInstruction(
                    method=InstallationMethod.PIP,
                    command="pip install tree-sitter",
                    description="Install Tree-sitter Python bindings",
                    prerequisites=["Python 3.6+"],
                    post_install_steps=[
                        "Install language grammars: pip install tree-sitter-python tree-sitter-javascript etc."
                    ],
                    verification_command="python -c 'import tree_sitter; print(tree_sitter.__version__)'"
                ),
                InstallationInstruction(
                    method=InstallationMethod.NPM,
                    command="npm install tree-sitter tree-sitter-cli",
                    description="Install Tree-sitter for Node.js",
                    prerequisites=["Node.js 12+"],
                    verification_command="tree-sitter --version"
                )
            ],
            usage_examples=[
                UsageExample(
                    description="Parse Python code and generate AST",
                    command="python -c \"import tree_sitter_python; from tree_sitter import Language, Parser; parser = Parser(); parser.set_language(Language(tree_sitter_python.language(), 'python')); tree = parser.parse(b'def hello(): pass')\"",
                    input_example="def hello():\n    pass",
                    expected_output_format="Tree-sitter syntax tree with nodes and relationships",
                    notes="Requires tree-sitter-python grammar to be installed"
                )
            ],
            comparison_with_joern=[
                ToolComparison(
                    feature="AST Generation",
                    joern_support="full",
                    alternative_support="full",
                    notes="Tree-sitter provides detailed syntax trees"
                ),
                ToolComparison(
                    feature="Semantic Analysis",
                    joern_support="full",
                    alternative_support="none",
                    notes="Tree-sitter focuses on syntax, not semantics"
                ),
                ToolComparison(
                    feature="Language Support",
                    joern_support="partial",
                    alternative_support="full",
                    notes="Tree-sitter supports more languages through community grammars"
                )
            ],
            official_website="https://tree-sitter.github.io/tree-sitter/",
            documentation_url="https://tree-sitter.github.io/tree-sitter/",
            github_url="https://github.com/tree-sitter/tree-sitter",
            license="MIT",
            maturity_level="mature",
            pros=[
                "Supports many programming languages",
                "Fast incremental parsing",
                "Active community and development",
                "Language-agnostic approach"
            ],
            cons=[
                "Syntax-only parsing (no semantic analysis)",
                "Requires separate grammar installation for each language",
                "Different output format than Joern CPG"
            ],
            integration_complexity="medium"
        )
        
        # Python AST module
        python_ast = AlternativeTool(
            name="Python AST",
            description="Built-in Python module for parsing Python source code into Abstract Syntax Trees",
            supported_languages=["python"],
            capabilities=[
                ToolCapability.AST_GENERATION,
                ToolCapability.SYNTAX_PARSING
            ],
            installation=[
                InstallationInstruction(
                    method=InstallationMethod.MANUAL,
                    command="# Built into Python standard library",
                    description="No installation required - part of Python standard library",
                    prerequisites=["Python 2.6+"],
                    verification_command="python -c 'import ast; print(\"AST module available\")'"
                )
            ],
            usage_examples=[
                UsageExample(
                    description="Parse Python code and dump AST",
                    command="python -c \"import ast; print(ast.dump(ast.parse('def hello(): pass')))\"",
                    input_example="def hello():\n    pass",
                    expected_output_format="AST node representation with Python objects",
                    notes="Provides detailed Python-specific AST nodes"
                )
            ],
            comparison_with_joern=[
                ToolComparison(
                    feature="Python Support",
                    joern_support="full",
                    alternative_support="full",
                    notes="Both provide comprehensive Python parsing"
                ),
                ToolComparison(
                    feature="Multi-language Support",
                    joern_support="full",
                    alternative_support="none",
                    notes="Python AST only works with Python code"
                )
            ],
            official_website="https://docs.python.org/3/library/ast.html",
            documentation_url="https://docs.python.org/3/library/ast.html",
            license="Python Software Foundation License",
            maturity_level="mature",
            pros=[
                "Built into Python standard library",
                "No additional installation required",
                "Comprehensive Python AST support",
                "Well-documented and stable"
            ],
            cons=[
                "Python-only support",
                "No semantic analysis beyond syntax",
                "Different output format than Joern"
            ],
            integration_complexity="low"
        )
        
        # ANTLR
        antlr = AlternativeTool(
            name="ANTLR",
            description="ANother Tool for Language Recognition - parser generator for reading, processing, executing, or translating structured text",
            supported_languages=[
                "java", "python", "javascript", "c", "cpp", "csharp", "go",
                "php", "swift", "kotlin", "ruby"
            ],
            capabilities=[
                ToolCapability.AST_GENERATION,
                ToolCapability.SYNTAX_PARSING
            ],
            installation=[
                InstallationInstruction(
                    method=InstallationMethod.MANUAL,
                    command="wget https://www.antlr.org/download/antlr-4.x.x-complete.jar",
                    description="Download ANTLR JAR file",
                    prerequisites=["Java 8+"],
                    post_install_steps=[
                        "Set CLASSPATH to include antlr-4.x.x-complete.jar",
                        "Create alias: alias antlr4='java -jar /path/to/antlr-4.x.x-complete.jar'"
                    ],
                    verification_command="java -jar antlr-4.x.x-complete.jar"
                ),
                InstallationInstruction(
                    method=InstallationMethod.PIP,
                    command="pip install antlr4-python3-runtime",
                    description="Install ANTLR Python runtime",
                    prerequisites=["Python 3.6+"],
                    verification_command="python -c 'import antlr4; print(antlr4.__version__)'"
                )
            ],
            usage_examples=[
                UsageExample(
                    description="Generate parser for a language grammar",
                    command="antlr4 -Dlanguage=Python3 MyGrammar.g4",
                    input_example="grammar MyGrammar;\nprogram: statement+;\nstatement: ID '=' expr ';';\n...",
                    expected_output_format="Generated parser classes and lexer",
                    notes="Requires grammar file definition for target language"
                )
            ],
            comparison_with_joern=[
                ToolComparison(
                    feature="Grammar Flexibility",
                    joern_support="partial",
                    alternative_support="full",
                    notes="ANTLR allows custom grammar definitions"
                ),
                ToolComparison(
                    feature="Ready-to-use Parsers",
                    joern_support="full",
                    alternative_support="partial",
                    notes="ANTLR requires grammar development for new languages"
                )
            ],
            official_website="https://www.antlr.org/",
            documentation_url="https://github.com/antlr/antlr4/blob/master/doc/index.md",
            github_url="https://github.com/antlr/antlr4",
            license="BSD-3-Clause",
            maturity_level="mature",
            pros=[
                "Highly flexible grammar system",
                "Supports many target languages",
                "Strong community and ecosystem",
                "Excellent documentation"
            ],
            cons=[
                "Requires grammar development expertise",
                "Complex setup for new languages",
                "Higher learning curve"
            ],
            integration_complexity="high"
        )
        
        # Store tools in database
        self._tools = {
            "tree-sitter": tree_sitter,
            "python-ast": python_ast,
            "antlr": antlr
        }
        
        # Initialize language-specific alternatives
        self._initialize_language_alternatives()
    
    def _initialize_language_alternatives(self):
        """Initialize language-specific alternative recommendations."""
        
        # Python alternatives
        self._language_alternatives["python"] = LanguageAlternatives(
            language="python",
            primary_alternatives=[self._tools["tree-sitter"], self._tools["python-ast"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="python-ast",
            joern_support_status="full",
            notes="Python AST is recommended for Python-only projects, Tree-sitter for multi-language projects"
        )
        
        # JavaScript alternatives
        self._language_alternatives["javascript"] = LanguageAlternatives(
            language="javascript",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter provides excellent JavaScript support with TypeScript compatibility"
        )
        
        # Java alternatives
        self._language_alternatives["java"] = LanguageAlternatives(
            language="java",
            primary_alternatives=[self._tools["tree-sitter"], self._tools["antlr"]],
            secondary_alternatives=[],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Both Tree-sitter and ANTLR provide good Java support"
        )
        
        # C/C++ alternatives
        for lang in ["c", "cpp"]:
            self._language_alternatives[lang] = LanguageAlternatives(
                language=lang,
                primary_alternatives=[self._tools["tree-sitter"]],
                secondary_alternatives=[self._tools["antlr"]],
                recommended_tool="tree-sitter",
                joern_support_status="full",
                notes="Tree-sitter provides good C/C++ parsing capabilities"
            )
        
        # C# alternatives
        self._language_alternatives["csharp"] = LanguageAlternatives(
            language="csharp",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter has community-maintained C# grammar"
        )
        
        # Go alternatives
        self._language_alternatives["go"] = LanguageAlternatives(
            language="go",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter provides good Go language support"
        )
        
        # PHP alternatives
        self._language_alternatives["php"] = LanguageAlternatives(
            language="php",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter has maintained PHP grammar"
        )
        
        # Ruby alternatives
        self._language_alternatives["ruby"] = LanguageAlternatives(
            language="ruby",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter provides Ruby parsing support"
        )
        
        # Swift alternatives
        self._language_alternatives["swift"] = LanguageAlternatives(
            language="swift",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter has community Swift grammar"
        )
        
        # Kotlin alternatives
        self._language_alternatives["kotlin"] = LanguageAlternatives(
            language="kotlin",
            primary_alternatives=[self._tools["tree-sitter"]],
            secondary_alternatives=[self._tools["antlr"]],
            recommended_tool="tree-sitter",
            joern_support_status="full",
            notes="Tree-sitter provides Kotlin parsing capabilities"
        )
    
    def get_tool(self, tool_name: str) -> Optional[AlternativeTool]:
        """
        Get information about a specific alternative tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            AlternativeTool object or None if not found
        """
        return self._tools.get(tool_name.lower())
    
    def get_alternatives_for_language(self, language: str) -> Optional[LanguageAlternatives]:
        """
        Get alternative tools for a specific programming language.
        
        Args:
            language: Programming language identifier
            
        Returns:
            LanguageAlternatives object or None if not found
        """
        return self._language_alternatives.get(language.lower())
    
    def get_all_tools(self) -> Dict[str, AlternativeTool]:
        """
        Get all alternative tools in the database.
        
        Returns:
            Dictionary mapping tool names to AlternativeTool objects
        """
        return self._tools.copy()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of all languages that have alternative tools available.
        
        Returns:
            List of language identifiers
        """
        return list(self._language_alternatives.keys())
    
    def search_tools_by_capability(self, capability: ToolCapability) -> List[AlternativeTool]:
        """
        Search for tools that provide a specific capability.
        
        Args:
            capability: Desired tool capability
            
        Returns:
            List of tools that provide the capability
        """
        matching_tools = []
        for tool in self._tools.values():
            if capability in tool.capabilities:
                matching_tools.append(tool)
        return matching_tools
    
    def search_tools_by_language(self, language: str) -> List[AlternativeTool]:
        """
        Search for tools that support a specific language.
        
        Args:
            language: Programming language identifier
            
        Returns:
            List of tools that support the language
        """
        matching_tools = []
        for tool in self._tools.values():
            if language.lower() in [lang.lower() for lang in tool.supported_languages]:
                matching_tools.append(tool)
        return matching_tools
    
    def add_tool(self, tool_name: str, tool: AlternativeTool):
        """
        Add a new alternative tool to the database.
        
        Args:
            tool_name: Unique identifier for the tool
            tool: AlternativeTool object
        """
        self._tools[tool_name.lower()] = tool
    
    def update_language_alternatives(self, language: str, alternatives: LanguageAlternatives):
        """
        Update alternative recommendations for a specific language.
        
        Args:
            language: Programming language identifier
            alternatives: LanguageAlternatives object
        """
        self._language_alternatives[language.lower()] = alternatives