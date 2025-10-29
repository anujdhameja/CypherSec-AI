"""
Test content templates for different programming languages.
Each template contains representative code constructs for CPG generation testing.
"""

from typing import Dict

class TestTemplates:
    """Contains test code templates for various programming languages."""
    
    @staticmethod
    def get_c_template() -> str:
        """C language test template with functions, control flow, and variables."""
        return '''#include <stdio.h>
#include <stdlib.h>

int global_var = 42;

int add_numbers(int a, int b) {
    return a + b;
}

int main() {
    int local_var = 10;
    int result;
    
    if (local_var > 5) {
        result = add_numbers(local_var, global_var);
        printf("Result: %d\\n", result);
    } else {
        result = 0;
    }
    
    for (int i = 0; i < 3; i++) {
        printf("Loop iteration: %d\\n", i);
    }
    
    return 0;
}'''

    @staticmethod
    def get_cpp_template() -> str:
        """C++ language test template with classes, methods, and STL."""
        return '''#include <iostream>
#include <vector>
#include <string>

class Calculator {
private:
    std::string name;
    
public:
    Calculator(const std::string& calc_name) : name(calc_name) {}
    
    int add(int a, int b) {
        return a + b;
    }
    
    void display_result(int result) {
        std::cout << name << " result: " << result << std::endl;
    }
};

int main() {
    Calculator calc("TestCalc");
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    int sum = 0;
    for (const auto& num : numbers) {
        sum = calc.add(sum, num);
    }
    
    calc.display_result(sum);
    return 0;
}'''

    @staticmethod
    def get_csharp_template() -> str:
        """C# language test template with classes, properties, and exception handling."""
        return '''using System;
using System.Collections.Generic;

namespace TestNamespace
{
    public class Calculator
    {
        public string Name { get; set; }
        private List<int> history;
        
        public Calculator(string name)
        {
            Name = name;
            history = new List<int>();
        }
        
        public int Add(int a, int b)
        {
            try
            {
                int result = a + b;
                history.Add(result);
                return result;
            }
            catch (OverflowException ex)
            {
                Console.WriteLine($"Overflow error: {ex.Message}");
                return 0;
            }
        }
        
        public void DisplayHistory()
        {
            foreach (int result in history)
            {
                Console.WriteLine($"Previous result: {result}");
            }
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Calculator calc = new Calculator("TestCalc");
            int result = calc.Add(10, 20);
            calc.DisplayHistory();
            Console.WriteLine($"Final result: {result}");
        }
    }
}'''

    @staticmethod
    def get_python_template() -> str:
        """Python language test template with functions, classes, and imports."""
        return '''import os
import sys
from typing import List, Optional

class Calculator:
    """A simple calculator class for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.history: List[int] = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers and store result in history."""
        result = a + b
        self.history.append(result)
        return result
    
    def get_history(self) -> List[int]:
        """Return calculation history."""
        return self.history.copy()
    
    def display_result(self, result: int) -> None:
        """Display calculation result."""
        print(f"{self.name} result: {result}")

def main():
    """Main function for testing."""
    calc = Calculator("TestCalc")
    numbers = [1, 2, 3, 4, 5]
    
    total = 0
    for num in numbers:
        total = calc.add(total, num)
    
    calc.display_result(total)
    
    # Test conditional logic
    if total > 10:
        print("Result is greater than 10")
    else:
        print("Result is 10 or less")

if __name__ == "__main__":
    main()'''

    @staticmethod
    def get_java_template() -> str:
        """Java language test template with classes, methods, and collections."""
        return '''import java.util.*;

public class Calculator {
    private String name;
    private List<Integer> history;
    
    public Calculator(String name) {
        this.name = name;
        this.history = new ArrayList<>();
    }
    
    public int add(int a, int b) {
        try {
            int result = a + b;
            history.add(result);
            return result;
        } catch (ArithmeticException e) {
            System.out.println("Arithmetic error: " + e.getMessage());
            return 0;
        }
    }
    
    public void displayHistory() {
        for (Integer result : history) {
            System.out.println("Previous result: " + result);
        }
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator("TestCalc");
        int[] numbers = {1, 2, 3, 4, 5};
        
        int sum = 0;
        for (int num : numbers) {
            sum = calc.add(sum, num);
        }
        
        calc.displayHistory();
        System.out.println("Final result: " + sum);
        
        if (sum > 10) {
            System.out.println("Result is greater than 10");
        }
    }
}'''

    @staticmethod
    def get_php_template() -> str:
        """PHP language test template with functions, arrays, and web constructs."""
        return '''<?php

class Calculator {
    private $name;
    private $history;
    
    public function __construct($name) {
        $this->name = $name;
        $this->history = array();
    }
    
    public function add($a, $b) {
        try {
            $result = $a + $b;
            array_push($this->history, $result);
            return $result;
        } catch (Exception $e) {
            echo "Error: " . $e->getMessage() . "\\n";
            return 0;
        }
    }
    
    public function displayHistory() {
        foreach ($this->history as $result) {
            echo "Previous result: " . $result . "\\n";
        }
    }
    
    public function getName() {
        return $this->name;
    }
}

function main() {
    $calc = new Calculator("TestCalc");
    $numbers = array(1, 2, 3, 4, 5);
    
    $sum = 0;
    foreach ($numbers as $num) {
        $sum = $calc->add($sum, $num);
    }
    
    $calc->displayHistory();
    echo "Final result: " . $sum . "\\n";
    
    if ($sum > 10) {
        echo "Result is greater than 10\\n";
    } else {
        echo "Result is 10 or less\\n";
    }
}

main();

?>'''

    @staticmethod
    def get_javascript_template() -> str:
        """JavaScript language test template with functions, objects, and async patterns."""
        return '''class Calculator {
    constructor(name) {
        this.name = name;
        this.history = [];
    }
    
    add(a, b) {
        try {
            const result = a + b;
            this.history.push(result);
            return result;
        } catch (error) {
            console.log(`Error: ${error.message}`);
            return 0;
        }
    }
    
    displayHistory() {
        this.history.forEach((result, index) => {
            console.log(`Previous result ${index}: ${result}`);
        });
    }
    
    async processNumbers(numbers) {
        return new Promise((resolve) => {
            let sum = 0;
            numbers.forEach(num => {
                sum = this.add(sum, num);
            });
            resolve(sum);
        });
    }
}

async function main() {
    const calc = new Calculator("TestCalc");
    const numbers = [1, 2, 3, 4, 5];
    
    try {
        const result = await calc.processNumbers(numbers);
        calc.displayHistory();
        console.log(`Final result: ${result}`);
        
        if (result > 10) {
            console.log("Result is greater than 10");
        } else {
            console.log("Result is 10 or less");
        }
    } catch (error) {
        console.error("Processing error:", error);
    }
}

main();'''

    @staticmethod
    def get_kotlin_template() -> str:
        """Kotlin language test template with classes, null safety, and collections."""
        return '''import kotlin.collections.mutableListOf

class Calculator(private val name: String) {
    private val history: MutableList<Int> = mutableListOf()
    
    fun add(a: Int, b: Int): Int {
        return try {
            val result = a + b
            history.add(result)
            result
        } catch (e: ArithmeticException) {
            println("Arithmetic error: ${e.message}")
            0
        }
    }
    
    fun displayHistory() {
        history.forEachIndexed { index, result ->
            println("Previous result $index: $result")
        }
    }
    
    fun getName(): String = name
}

fun main() {
    val calc = Calculator("TestCalc")
    val numbers = listOf(1, 2, 3, 4, 5)
    
    var sum = 0
    numbers.forEach { num ->
        sum = calc.add(sum, num)
    }
    
    calc.displayHistory()
    println("Final result: $sum")
    
    when {
        sum > 10 -> println("Result is greater than 10")
        sum == 10 -> println("Result is exactly 10")
        else -> println("Result is less than 10")
    }
}'''

    @staticmethod
    def get_ruby_template() -> str:
        """Ruby language test template with classes and Ruby-specific constructs."""
        return '''class Calculator
  attr_reader :name, :history
  
  def initialize(name)
    @name = name
    @history = []
  end
  
  def add(a, b)
    begin
      result = a + b
      @history << result
      result
    rescue StandardError => e
      puts "Error: #{e.message}"
      0
    end
  end
  
  def display_history
    @history.each_with_index do |result, index|
      puts "Previous result #{index}: #{result}"
    end
  end
end

def main
  calc = Calculator.new("TestCalc")
  numbers = [1, 2, 3, 4, 5]
  
  sum = 0
  numbers.each do |num|
    sum = calc.add(sum, num)
  end
  
  calc.display_history
  puts "Final result: #{sum}"
  
  case sum
  when 0..10
    puts "Result is 10 or less"
  when 11..20
    puts "Result is between 11 and 20"
  else
    puts "Result is greater than 20"
  end
end

main if __FILE__ == $0'''

    @staticmethod
    def get_swift_template() -> str:
        """Swift language test template with classes and Swift-specific features."""
        return '''import Foundation

class Calculator {
    let name: String
    private var history: [Int] = []
    
    init(name: String) {
        self.name = name
    }
    
    func add(_ a: Int, _ b: Int) -> Int {
        let result = a + b
        history.append(result)
        return result
    }
    
    func displayHistory() {
        for (index, result) in history.enumerated() {
            print("Previous result \\(index): \\(result)")
        }
    }
    
    func processNumbers(_ numbers: [Int]) -> Int {
        var sum = 0
        for num in numbers {
            sum = add(sum, num)
        }
        return sum
    }
}

func main() {
    let calc = Calculator(name: "TestCalc")
    let numbers = [1, 2, 3, 4, 5]
    
    let result = calc.processNumbers(numbers)
    calc.displayHistory()
    print("Final result: \\(result)")
    
    switch result {
    case 0...10:
        print("Result is 10 or less")
    case 11...20:
        print("Result is between 11 and 20")
    default:
        print("Result is greater than 20")
    }
}

main()'''

    @staticmethod
    def get_go_template() -> str:
        """Go language test template with package definition and Go-specific constructs."""
        return '''package main

import (
    "fmt"
    "errors"
)

type Calculator struct {
    Name    string
    history []int
}

func NewCalculator(name string) *Calculator {
    return &Calculator{
        Name:    name,
        history: make([]int, 0),
    }
}

func (c *Calculator) Add(a, b int) (int, error) {
    if a < 0 || b < 0 {
        return 0, errors.New("negative numbers not supported")
    }
    
    result := a + b
    c.history = append(c.history, result)
    return result, nil
}

func (c *Calculator) DisplayHistory() {
    for i, result := range c.history {
        fmt.Printf("Previous result %d: %d\\n", i, result)
    }
}

func main() {
    calc := NewCalculator("TestCalc")
    numbers := []int{1, 2, 3, 4, 5}
    
    sum := 0
    for _, num := range numbers {
        result, err := calc.Add(sum, num)
        if err != nil {
            fmt.Printf("Error: %v\\n", err)
            continue
        }
        sum = result
    }
    
    calc.DisplayHistory()
    fmt.Printf("Final result: %d\\n", sum)
    
    switch {
    case sum <= 10:
        fmt.Println("Result is 10 or less")
    case sum <= 20:
        fmt.Println("Result is between 11 and 20")
    default:
        fmt.Println("Result is greater than 20")
    }
}'''

    @staticmethod
    def get_all_templates() -> Dict[str, str]:
        """Return all available test templates."""
        return {
            'c': TestTemplates.get_c_template(),
            'cpp': TestTemplates.get_cpp_template(),
            'csharp': TestTemplates.get_csharp_template(),
            'python': TestTemplates.get_python_template(),
            'java': TestTemplates.get_java_template(),
            'php': TestTemplates.get_php_template(),
            'javascript': TestTemplates.get_javascript_template(),
            'kotlin': TestTemplates.get_kotlin_template(),
            'ruby': TestTemplates.get_ruby_template(),
            'swift': TestTemplates.get_swift_template(),
            'go': TestTemplates.get_go_template()
        }

    @staticmethod
    def get_file_extensions() -> Dict[str, str]:
        """Return file extensions for each language."""
        return {
            'c': '.c',
            'cpp': '.cpp',
            'csharp': '.cs',
            'python': '.py',
            'java': '.java',
            'php': '.php',
            'javascript': '.js',
            'kotlin': '.kt',
            'ruby': '.rb',
            'swift': '.swift',
            'go': '.go'
        }