#!/usr/bin/env python3
"""
Sample Python file for Joern CPG generation testing.
This file contains representative Python constructs for verification.
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class User:
    """Sample user data class."""
    name: str
    email: str
    age: int
    active: bool = True


class UserManager:
    """Sample class demonstrating Python features."""
    
    def __init__(self):
        self.users: List[User] = []
        self._cache: Dict[str, User] = {}
    
    def add_user(self, name: str, email: str, age: int) -> bool:
        """Add a new user to the system."""
        try:
            if self.validate_email(email):
                user = User(name=name, email=email, age=age)
                self.users.append(user)
                self._cache[email] = user
                return True
            else:
                raise ValueError("Invalid email format")
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
    
    def find_user(self, email: str) -> Optional[User]:
        """Find user by email address."""
        if email in self._cache:
            return self._cache[email]
        
        for user in self.users:
            if user.email == email:
                self._cache[email] = user
                return user
        
        return None
    
    def get_active_users(self) -> List[User]:
        """Get all active users."""
        return [user for user in self.users if user.active]
    
    def validate_email(self, email: str) -> bool:
        """Simple email validation."""
        return "@" in email and "." in email.split("@")[1]
    
    def process_users_batch(self, batch_size: int = 10):
        """Process users in batches."""
        for i in range(0, len(self.users), batch_size):
            batch = self.users[i:i + batch_size]
            yield batch


def calculate_statistics(numbers: List[int]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}
    
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    
    return {
        "count": count,
        "sum": total,
        "mean": mean,
        "min": min(numbers),
        "max": max(numbers)
    }


def main():
    """Main function demonstrating usage."""
    # Create user manager
    manager = UserManager()
    
    # Sample data
    sample_users = [
        ("Alice Johnson", "alice@example.com", 28),
        ("Bob Smith", "bob@example.com", 35),
        ("Carol Davis", "carol@example.com", 42),
    ]
    
    # Add users
    for name, email, age in sample_users:
        success = manager.add_user(name, email, age)
        if success:
            print(f"Added user: {name}")
        else:
            print(f"Failed to add user: {name}")
    
    # Find and display users
    test_email = "alice@example.com"
    user = manager.find_user(test_email)
    if user:
        print(f"Found user: {user.name} ({user.email})")
    
    # Process active users
    active_users = manager.get_active_users()
    print(f"Active users: {len(active_users)}")
    
    # Calculate statistics
    ages = [user.age for user in active_users]
    stats = calculate_statistics(ages)
    print(f"Age statistics: {stats}")
    
    # Process in batches
    for batch in manager.process_users_batch(2):
        print(f"Processing batch of {len(batch)} users")


if __name__ == "__main__":
    main()