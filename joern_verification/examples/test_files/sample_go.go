// Sample Go file for Joern CPG generation testing.
// This file contains representative Go constructs for verification.

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// User represents a user in the system
type User struct {
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	Age       int       `json:"age"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
}

// NewUser creates a new user with validation
func NewUser(name, email string, age int) (*User, error) {
	if strings.TrimSpace(name) == "" {
		return nil, errors.New("name cannot be empty")
	}
	
	if !isValidEmail(email) {
		return nil, errors.New("invalid email format")
	}
	
	if age < 0 || age > 150 {
		return nil, errors.New("invalid age")
	}
	
	return &User{
		Name:      name,
		Email:     email,
		Age:       age,
		Active:    true,
		CreatedAt: time.Now(),
	}, nil
}

// GetDisplayName returns formatted display name
func (u *User) GetDisplayName() string {
	return fmt.Sprintf("%s (%s)", u.Name, u.Email)
}

// IsAdult checks if user is 18 or older
func (u *User) IsAdult() bool {
	return u.Age >= 18
}

// Deactivate sets user as inactive
func (u *User) Deactivate() {
	u.Active = false
}

// ToJSON converts user to JSON bytes
func (u *User) ToJSON() ([]byte, error) {
	return json.Marshal(u)
}

// UserManager manages a collection of users
type UserManager struct {
	users []User
	cache map[string]*User
	mutex sync.RWMutex
}

// NewUserManager creates a new user manager
func NewUserManager() *UserManager {
	return &UserManager{
		users: make([]User, 0),
		cache: make(map[string]*User),
	}
}

// AddUser adds a new user to the system
func (um *UserManager) AddUser(name, email string, age int) error {
	um.mutex.Lock()
	defer um.mutex.Unlock()
	
	// Check for duplicate email
	if _, exists := um.cache[email]; exists {
		return errors.New("user with this email already exists")
	}
	
	// Create new user
	user, err := NewUser(name, email, age)
	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}
	
	// Add to collections
	um.users = append(um.users, *user)
	um.cache[email] = &um.users[len(um.users)-1]
	
	return nil
}

// FindUser finds a user by email
func (um *UserManager) FindUser(email string) *User {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	if user, exists := um.cache[email]; exists {
		return user
	}
	
	return nil
}

// GetActiveUsers returns all active users
func (um *UserManager) GetActiveUsers() []User {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	var activeUsers []User
	for _, user := range um.users {
		if user.Active {
			activeUsers = append(activeUsers, user)
		}
	}
	
	return activeUsers
}

// GetUsersByAgeRange returns users within specified age range
func (um *UserManager) GetUsersByAgeRange(minAge, maxAge int) []User {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	var filteredUsers []User
	for _, user := range um.users {
		if user.Age >= minAge && user.Age <= maxAge {
			filteredUsers = append(filteredUsers, user)
		}
	}
	
	// Sort by age
	sort.Slice(filteredUsers, func(i, j int) bool {
		return filteredUsers[i].Age < filteredUsers[j].Age
	})
	
	return filteredUsers
}

// SearchUsers searches users by name pattern
func (um *UserManager) SearchUsers(pattern string) []User {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	regex, err := regexp.Compile("(?i)" + pattern)
	if err != nil {
		log.Printf("Invalid search pattern: %v", err)
		return []User{}
	}
	
	var matchingUsers []User
	for _, user := range um.users {
		if regex.MatchString(user.Name) {
			matchingUsers = append(matchingUsers, user)
		}
	}
	
	return matchingUsers
}

// ProcessUsersInBatches processes users in parallel batches
func (um *UserManager) ProcessUsersInBatches(batchSize int, processor func(User) (string, error)) []ProcessResult {
	um.mutex.RLock()
	users := make([]User, len(um.users))
	copy(users, um.users)
	um.mutex.RUnlock()
	
	var results []ProcessResult
	var wg sync.WaitGroup
	var mutex sync.Mutex
	
	for i := 0; i < len(users); i += batchSize {
		end := i + batchSize
		if end > len(users) {
			end = len(users)
		}
		
		batch := users[i:end]
		fmt.Printf("Processing batch of %d users\n", len(batch))
		
		wg.Add(1)
		go func(batch []User) {
			defer wg.Done()
			
			for _, user := range batch {
				result := ProcessResult{User: user}
				
				if output, err := processor(user); err != nil {
					result.Error = err.Error()
					result.Success = false
				} else {
					result.Output = output
					result.Success = true
				}
				
				mutex.Lock()
				results = append(results, result)
				mutex.Unlock()
			}
		}(batch)
	}
	
	wg.Wait()
	return results
}

// ProcessResult represents the result of processing a user
type ProcessResult struct {
	User    User   `json:"user"`
	Output  string `json:"output,omitempty"`
	Error   string `json:"error,omitempty"`
	Success bool   `json:"success"`
}

// Statistics represents user statistics
type Statistics struct {
	Count       int     `json:"count"`
	AverageAge  float64 `json:"average_age"`
	MinAge      int     `json:"min_age"`
	MaxAge      int     `json:"max_age"`
	ActiveCount int     `json:"active_count"`
	AdultCount  int     `json:"adult_count"`
}

// CalculateStatistics calculates user statistics
func (um *UserManager) CalculateStatistics() Statistics {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	if len(um.users) == 0 {
		return Statistics{}
	}
	
	stats := Statistics{
		Count:  len(um.users),
		MinAge: um.users[0].Age,
		MaxAge: um.users[0].Age,
	}
	
	totalAge := 0
	for _, user := range um.users {
		totalAge += user.Age
		
		if user.Age < stats.MinAge {
			stats.MinAge = user.Age
		}
		if user.Age > stats.MaxAge {
			stats.MaxAge = user.Age
		}
		
		if user.Active {
			stats.ActiveCount++
		}
		
		if user.IsAdult() {
			stats.AdultCount++
		}
	}
	
	stats.AverageAge = float64(totalAge) / float64(len(um.users))
	return stats
}

// ExportToJSON exports all users and statistics to JSON
func (um *UserManager) ExportToJSON() ([]byte, error) {
	um.mutex.RLock()
	defer um.mutex.RUnlock()
	
	export := struct {
		Users      []User     `json:"users"`
		Statistics Statistics `json:"statistics"`
		ExportedAt time.Time  `json:"exported_at"`
	}{
		Users:      um.users,
		Statistics: um.CalculateStatistics(),
		ExportedAt: time.Now(),
	}
	
	return json.MarshalIndent(export, "", "  ")
}

// isValidEmail validates email format using regex
func isValidEmail(email string) bool {
	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	return emailRegex.MatchString(email)
}

// sampleProcessor is a sample user processor function
func sampleProcessor(user User) (string, error) {
	// Simulate some processing time
	time.Sleep(time.Millisecond * 10)
	
	if user.Age < 18 {
		return "", errors.New("user is not an adult")
	}
	
	return fmt.Sprintf("Processed user: %s", user.Name), nil
}

// main function demonstrating usage
func main() {
	fmt.Println("Starting Joern Go verification test...")
	
	// Create user manager
	manager := NewUserManager()
	
	// Sample data
	sampleUsers := []struct {
		name  string
		email string
		age   int
	}{
		{"Alice Johnson", "alice@example.com", 28},
		{"Bob Smith", "bob@example.com", 35},
		{"Carol Davis", "carol@example.com", 42},
		{"David Wilson", "david@example.com", 31},
		{"Eve Brown", "eve@example.com", 16}, // Minor user for testing
	}
	
	// Add users
	fmt.Println("\nAdding users...")
	for _, userData := range sampleUsers {
		if err := manager.AddUser(userData.name, userData.email, userData.age); err != nil {
			log.Printf("Failed to add user %s: %v", userData.name, err)
		} else {
			fmt.Printf("Added user: %s\n", userData.name)
		}
	}
	
	// Find and display user
	testEmail := "alice@example.com"
	if user := manager.FindUser(testEmail); user != nil {
		fmt.Printf("\nFound user: %s\n", user.GetDisplayName())
	} else {
		fmt.Printf("User not found: %s\n", testEmail)
	}
	
	// Get active users
	activeUsers := manager.GetActiveUsers()
	fmt.Printf("\nActive users: %d\n", len(activeUsers))
	
	// Get users by age range
	youngUsers := manager.GetUsersByAgeRange(25, 35)
	fmt.Printf("Users aged 25-35: %d\n", len(youngUsers))
	
	// Search users
	searchResults := manager.SearchUsers("alice")
	fmt.Printf("Search results for 'alice': %d\n", len(searchResults))
	
	// Calculate statistics
	stats := manager.CalculateStatistics()
	fmt.Printf("\nStatistics:\n")
	fmt.Printf("  Count: %d\n", stats.Count)
	fmt.Printf("  Average Age: %.2f\n", stats.AverageAge)
	fmt.Printf("  Min Age: %d\n", stats.MinAge)
	fmt.Printf("  Max Age: %d\n", stats.MaxAge)
	fmt.Printf("  Active Count: %d\n", stats.ActiveCount)
	fmt.Printf("  Adult Count: %d\n", stats.AdultCount)
	
	// Process users in batches
	fmt.Println("\nProcessing users in batches...")
	results := manager.ProcessUsersInBatches(2, sampleProcessor)
	
	successCount := 0
	for _, result := range results {
		if result.Success {
			successCount++
			fmt.Printf("  ✓ %s\n", result.Output)
		} else {
			fmt.Printf("  ✗ %s: %s\n", result.User.Name, result.Error)
		}
	}
	
	fmt.Printf("Processed %d users, %d successful\n", len(results), successCount)
	
	// Export data
	if exportData, err := manager.ExportToJSON(); err != nil {
		log.Printf("Failed to export data: %v", err)
	} else {
		fmt.Printf("\nExported %d users to JSON (%d bytes)\n", len(activeUsers), len(exportData))
	}
	
	fmt.Println("\nJoern Go verification test completed successfully!")
}