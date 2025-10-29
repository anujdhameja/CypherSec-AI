/**
 * Sample C file for Joern CPG generation testing.
 * This file contains representative C constructs for verification.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX_USERS 100
#define MAX_NAME_LENGTH 50
#define MAX_EMAIL_LENGTH 100

/**
 * User structure demonstrating C data structures.
 */
typedef struct {
    char name[MAX_NAME_LENGTH];
    char email[MAX_EMAIL_LENGTH];
    int age;
    bool active;
} User;

/**
 * User manager structure demonstrating C programming patterns.
 */
typedef struct {
    User users[MAX_USERS];
    int count;
} UserManager;

/**
 * Statistics structure for calculations.
 */
typedef struct {
    int count;
    int total_age;
    double average_age;
    int min_age;
    int max_age;
} Statistics;

/**
 * Initialize user manager.
 */
void init_user_manager(UserManager* manager) {
    if (manager == NULL) {
        return;
    }
    
    manager->count = 0;
    memset(manager->users, 0, sizeof(manager->users));
}

/**
 * Validate email address format.
 */
bool validate_email(const char* email) {
    if (email == NULL || strlen(email) == 0) {
        return false;
    }
    
    char* at_symbol = strchr(email, '@');
    if (at_symbol == NULL) {
        return false;
    }
    
    char* dot_symbol = strrchr(at_symbol, '.');
    if (dot_symbol == NULL || dot_symbol <= at_symbol) {
        return false;
    }
    
    return true;
}

/**
 * Add user to the manager.
 */
bool add_user(UserManager* manager, const char* name, const char* email, int age) {
    if (manager == NULL || name == NULL || email == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to add_user\n");
        return false;
    }
    
    if (manager->count >= MAX_USERS) {
        fprintf(stderr, "Error: Maximum number of users reached\n");
        return false;
    }
    
    if (!validate_email(email)) {
        fprintf(stderr, "Error: Invalid email format\n");
        return false;
    }
    
    if (age < 0 || age > 150) {
        fprintf(stderr, "Error: Invalid age\n");
        return false;
    }
    
    User* user = &manager->users[manager->count];
    
    strncpy(user->name, name, MAX_NAME_LENGTH - 1);
    user->name[MAX_NAME_LENGTH - 1] = '\0';
    
    strncpy(user->email, email, MAX_EMAIL_LENGTH - 1);
    user->email[MAX_EMAIL_LENGTH - 1] = '\0';
    
    user->age = age;
    user->active = true;
    
    manager->count++;
    return true;
}

/**
 * Find user by email address.
 */
User* find_user(UserManager* manager, const char* email) {
    if (manager == NULL || email == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < manager->count; i++) {
        if (strcmp(manager->users[i].email, email) == 0) {
            return &manager->users[i];
        }
    }
    
    return NULL;
}

/**
 * Get count of active users.
 */
int get_active_user_count(UserManager* manager) {
    if (manager == NULL) {
        return 0;
    }
    
    int active_count = 0;
    for (int i = 0; i < manager->count; i++) {
        if (manager->users[i].active) {
            active_count++;
        }
    }
    
    return active_count;
}

/**
 * Get users within age range.
 */
int get_users_by_age_range(UserManager* manager, int min_age, int max_age, User* result, int max_results) {
    if (manager == NULL || result == NULL || max_results <= 0) {
        return 0;
    }
    
    int found_count = 0;
    for (int i = 0; i < manager->count && found_count < max_results; i++) {
        if (manager->users[i].age >= min_age && manager->users[i].age <= max_age) {
            result[found_count] = manager->users[i];
            found_count++;
        }
    }
    
    return found_count;
}

/**
 * Calculate user statistics.
 */
Statistics calculate_statistics(UserManager* manager) {
    Statistics stats = {0, 0, 0.0, 0, 0};
    
    if (manager == NULL || manager->count == 0) {
        return stats;
    }
    
    stats.count = manager->count;
    stats.min_age = manager->users[0].age;
    stats.max_age = manager->users[0].age;
    
    for (int i = 0; i < manager->count; i++) {
        int age = manager->users[i].age;
        stats.total_age += age;
        
        if (age < stats.min_age) {
            stats.min_age = age;
        }
        if (age > stats.max_age) {
            stats.max_age = age;
        }
    }
    
    stats.average_age = (double)stats.total_age / stats.count;
    return stats;
}

/**
 * Process users in batches.
 */
void process_users_in_batches(UserManager* manager, int batch_size) {
    if (manager == NULL || batch_size <= 0) {
        return;
    }
    
    printf("Processing %d users in batches of %d:\n", manager->count, batch_size);
    
    for (int i = 0; i < manager->count; i += batch_size) {
        int batch_end = (i + batch_size < manager->count) ? i + batch_size : manager->count;
        int batch_count = batch_end - i;
        
        printf("Batch %d: Processing %d users\n", (i / batch_size) + 1, batch_count);
        
        for (int j = i; j < batch_end; j++) {
            printf("  - %s (%s, age %d)\n", 
                   manager->users[j].name, 
                   manager->users[j].email, 
                   manager->users[j].age);
        }
    }
}

/**
 * Print user information.
 */
void print_user(const User* user) {
    if (user == NULL) {
        printf("User: NULL\n");
        return;
    }
    
    printf("User: %s (%s), Age: %d, Active: %s\n", 
           user->name, user->email, user->age, user->active ? "Yes" : "No");
}

/**
 * Main function demonstrating usage.
 */
int main() {
    UserManager manager;
    init_user_manager(&manager);
    
    // Sample data
    const char* sample_users[][3] = {
        {"Alice Johnson", "alice@example.com", "28"},
        {"Bob Smith", "bob@example.com", "35"},
        {"Carol Davis", "carol@example.com", "42"},
        {"David Wilson", "david@example.com", "31"}
    };
    
    int num_samples = sizeof(sample_users) / sizeof(sample_users[0]);
    
    // Add users
    printf("Adding users:\n");
    for (int i = 0; i < num_samples; i++) {
        const char* name = sample_users[i][0];
        const char* email = sample_users[i][1];
        int age = atoi(sample_users[i][2]);
        
        bool success = add_user(&manager, name, email, age);
        if (success) {
            printf("Added user: %s\n", name);
        } else {
            printf("Failed to add user: %s\n", name);
        }
    }
    
    // Find and display user
    const char* test_email = "alice@example.com";
    User* user = find_user(&manager, test_email);
    if (user != NULL) {
        printf("\nFound user:\n");
        print_user(user);
    } else {
        printf("User not found: %s\n", test_email);
    }
    
    // Get active user count
    int active_count = get_active_user_count(&manager);
    printf("\nActive users: %d\n", active_count);
    
    // Get users by age range
    User young_users[MAX_USERS];
    int young_count = get_users_by_age_range(&manager, 25, 35, young_users, MAX_USERS);
    printf("Users aged 25-35: %d\n", young_count);
    
    // Calculate statistics
    Statistics stats = calculate_statistics(&manager);
    printf("\nStatistics:\n");
    printf("  Count: %d\n", stats.count);
    printf("  Total Age: %d\n", stats.total_age);
    printf("  Average Age: %.2f\n", stats.average_age);
    printf("  Min Age: %d\n", stats.min_age);
    printf("  Max Age: %d\n", stats.max_age);
    
    // Process in batches
    printf("\n");
    process_users_in_batches(&manager, 2);
    
    return 0;
}