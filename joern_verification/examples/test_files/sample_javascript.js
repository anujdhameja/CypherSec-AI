/**
 * Sample JavaScript file for Joern CPG generation testing.
 * This file contains representative JavaScript constructs for verification.
 */

'use strict';

/**
 * User class demonstrating ES6+ features.
 */
class User {
    constructor(name, email, age) {
        this.name = name;
        this.email = email;
        this.age = age;
        this.active = true;
        this.createdAt = new Date();
    }
    
    /**
     * Get user display name.
     */
    getDisplayName() {
        return `${this.name} (${this.email})`;
    }
    
    /**
     * Check if user is adult.
     */
    isAdult() {
        return this.age >= 18;
    }
    
    /**
     * Deactivate user account.
     */
    deactivate() {
        this.active = false;
    }
    
    /**
     * Convert user to JSON representation.
     */
    toJSON() {
        return {
            name: this.name,
            email: this.email,
            age: this.age,
            active: this.active,
            createdAt: this.createdAt.toISOString()
        };
    }
    
    /**
     * Static method to validate email format.
     */
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
}

/**
 * UserManager class demonstrating JavaScript patterns.
 */
class UserManager {
    constructor() {
        this.users = [];
        this.cache = new Map();
        this.eventListeners = new Map();
    }
    
    /**
     * Add event listener.
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    /**
     * Emit event to listeners.
     */
    emit(event, data) {
        const listeners = this.eventListeners.get(event) || [];
        listeners.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in event listener for ${event}:`, error);
            }
        });
    }
    
    /**
     * Add user to the system.
     */
    async addUser(name, email, age) {
        try {
            // Validate input
            if (!name || typeof name !== 'string') {
                throw new Error('Invalid name');
            }
            
            if (!User.validateEmail(email)) {
                throw new Error('Invalid email format');
            }
            
            if (!Number.isInteger(age) || age < 0 || age > 150) {
                throw new Error('Invalid age');
            }
            
            // Check for duplicate email
            if (this.cache.has(email)) {
                throw new Error('User with this email already exists');
            }
            
            // Create user
            const user = new User(name, email, age);
            
            // Simulate async operation
            await this.simulateAsyncOperation();
            
            // Add to collections
            this.users.push(user);
            this.cache.set(email, user);
            
            // Emit event
            this.emit('userAdded', user);
            
            return user;
        } catch (error) {
            console.error('Error adding user:', error.message);
            this.emit('error', { operation: 'addUser', error });
            throw error;
        }
    }
    
    /**
     * Find user by email.
     */
    findUser(email) {
        return this.cache.get(email) || null;
    }
    
    /**
     * Get all active users.
     */
    getActiveUsers() {
        return this.users.filter(user => user.active);
    }
    
    /**
     * Get users by age range.
     */
    getUsersByAgeRange(minAge, maxAge) {
        return this.users
            .filter(user => user.age >= minAge && user.age <= maxAge)
            .sort((a, b) => a.age - b.age);
    }
    
    /**
     * Search users by name pattern.
     */
    searchUsers(pattern) {
        const regex = new RegExp(pattern, 'i');
        return this.users.filter(user => regex.test(user.name));
    }
    
    /**
     * Process users with async operations.
     */
    async processUsersAsync(processor) {
        const results = [];
        
        for (const user of this.users) {
            try {
                const result = await processor(user);
                results.push({ user, result, success: true });
            } catch (error) {
                results.push({ user, error, success: false });
            }
        }
        
        return results;
    }
    
    /**
     * Process users in parallel batches.
     */
    async processUsersInBatches(batchSize = 5, processor) {
        const batches = [];
        
        for (let i = 0; i < this.users.length; i += batchSize) {
            const batch = this.users.slice(i, i + batchSize);
            batches.push(batch);
        }
        
        const results = [];
        
        for (const batch of batches) {
            console.log(`Processing batch of ${batch.length} users`);
            
            const batchPromises = batch.map(async (user) => {
                try {
                    const result = await processor(user);
                    return { user, result, success: true };
                } catch (error) {
                    return { user, error, success: false };
                }
            });
            
            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);
        }
        
        return results;
    }
    
    /**
     * Calculate user statistics.
     */
    calculateStatistics() {
        if (this.users.length === 0) {
            return {
                count: 0,
                averageAge: 0,
                minAge: 0,
                maxAge: 0,
                activeCount: 0,
                adultCount: 0
            };
        }
        
        const ages = this.users.map(user => user.age);
        const activeUsers = this.getActiveUsers();
        const adultUsers = this.users.filter(user => user.isAdult());
        
        return {
            count: this.users.length,
            averageAge: ages.reduce((sum, age) => sum + age, 0) / ages.length,
            minAge: Math.min(...ages),
            maxAge: Math.max(...ages),
            activeCount: activeUsers.length,
            adultCount: adultUsers.length
        };
    }
    
    /**
     * Export users to JSON.
     */
    exportToJSON() {
        return {
            users: this.users.map(user => user.toJSON()),
            statistics: this.calculateStatistics(),
            exportedAt: new Date().toISOString()
        };
    }
    
    /**
     * Simulate async operation (e.g., database call).
     */
    async simulateAsyncOperation() {
        return new Promise(resolve => {
            setTimeout(resolve, Math.random() * 100);
        });
    }
}

/**
 * Utility functions demonstrating functional programming patterns.
 */
const UserUtils = {
    /**
     * Create user factory function.
     */
    createUserFactory: (defaultAge = 18) => {
        return (name, email, age = defaultAge) => {
            return new User(name, email, age);
        };
    },
    
    /**
     * Compose functions for user processing.
     */
    compose: (...functions) => {
        return (value) => {
            return functions.reduceRight((acc, fn) => fn(acc), value);
        };
    },
    
    /**
     * Curry function for user filtering.
     */
    filterBy: (property) => (value) => (users) => {
        return users.filter(user => user[property] === value);
    },
    
    /**
     * Debounce function for search operations.
     */
    debounce: (func, delay) => {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(null, args), delay);
        };
    }
};

/**
 * Main function demonstrating usage.
 */
async function main() {
    try {
        console.log('Starting Joern JavaScript verification test...');
        
        // Create user manager
        const manager = new UserManager();
        
        // Set up event listeners
        manager.on('userAdded', (user) => {
            console.log(`User added: ${user.getDisplayName()}`);
        });
        
        manager.on('error', (errorData) => {
            console.error(`Operation ${errorData.operation} failed:`, errorData.error.message);
        });
        
        // Sample data
        const sampleUsers = [
            { name: 'Alice Johnson', email: 'alice@example.com', age: 28 },
            { name: 'Bob Smith', email: 'bob@example.com', age: 35 },
            { name: 'Carol Davis', email: 'carol@example.com', age: 42 },
            { name: 'David Wilson', email: 'david@example.com', age: 31 }
        ];
        
        // Add users asynchronously
        console.log('\nAdding users...');
        for (const userData of sampleUsers) {
            try {
                await manager.addUser(userData.name, userData.email, userData.age);
            } catch (error) {
                console.error(`Failed to add user ${userData.name}:`, error.message);
            }
        }
        
        // Find and display user
        const testEmail = 'alice@example.com';
        const user = manager.findUser(testEmail);
        if (user) {
            console.log(`\nFound user: ${user.getDisplayName()}`);
        } else {
            console.log(`User not found: ${testEmail}`);
        }
        
        // Get active users
        const activeUsers = manager.getActiveUsers();
        console.log(`\nActive users: ${activeUsers.length}`);
        
        // Get users by age range
        const youngUsers = manager.getUsersByAgeRange(25, 35);
        console.log(`Users aged 25-35: ${youngUsers.length}`);
        
        // Search users
        const searchResults = manager.searchUsers('alice');
        console.log(`Search results for 'alice': ${searchResults.length}`);
        
        // Calculate statistics
        const stats = manager.calculateStatistics();
        console.log('\nStatistics:', stats);
        
        // Process users asynchronously
        console.log('\nProcessing users asynchronously...');
        const processor = async (user) => {
            await manager.simulateAsyncOperation();
            return `Processed ${user.name}`;
        };
        
        const results = await manager.processUsersInBatches(2, processor);
        console.log(`Processed ${results.length} users`);
        
        // Export data
        const exportData = manager.exportToJSON();
        console.log(`\nExported ${exportData.users.length} users`);
        
        // Demonstrate utility functions
        const filterActiveUsers = UserUtils.filterBy('active')(true);
        const activeUsersFiltered = filterActiveUsers(manager.users);
        console.log(`Filtered active users: ${activeUsersFiltered.length}`);
        
        console.log('\nJoern JavaScript verification test completed successfully!');
        
    } catch (error) {
        console.error('Main function error:', error);
    }
}

// Run main function if this is the main module
if (typeof require !== 'undefined' && require.main === module) {
    main().catch(console.error);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { User, UserManager, UserUtils };
}