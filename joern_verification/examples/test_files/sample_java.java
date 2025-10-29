import java.util.*;
import java.util.stream.Collectors;

/**
 * Sample Java file for Joern CPG generation testing.
 * This file contains representative Java constructs for verification.
 */
public class SampleJava {
    
    /**
     * User data class demonstrating Java features.
     */
    public static class User {
        private String name;
        private String email;
        private int age;
        private boolean active;
        
        public User(String name, String email, int age) {
            this.name = name;
            this.email = email;
            this.age = age;
            this.active = true;
        }
        
        // Getters and setters
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        
        public int getAge() { return age; }
        public void setAge(int age) { this.age = age; }
        
        public boolean isActive() { return active; }
        public void setActive(boolean active) { this.active = active; }
        
        @Override
        public String toString() {
            return String.format("User{name='%s', email='%s', age=%d, active=%s}", 
                               name, email, age, active);
        }
        
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            User user = (User) obj;
            return Objects.equals(email, user.email);
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(email);
        }
    }
    
    /**
     * User management class demonstrating Java collections and error handling.
     */
    public static class UserManager {
        private List<User> users;
        private Map<String, User> cache;
        
        public UserManager() {
            this.users = new ArrayList<>();
            this.cache = new HashMap<>();
        }
        
        public boolean addUser(String name, String email, int age) {
            try {
                if (validateEmail(email)) {
                    User user = new User(name, email, age);
                    users.add(user);
                    cache.put(email, user);
                    return true;
                } else {
                    throw new IllegalArgumentException("Invalid email format");
                }
            } catch (Exception e) {
                System.err.println("Error adding user: " + e.getMessage());
                return false;
            }
        }
        
        public Optional<User> findUser(String email) {
            if (cache.containsKey(email)) {
                return Optional.of(cache.get(email));
            }
            
            for (User user : users) {
                if (user.getEmail().equals(email)) {
                    cache.put(email, user);
                    return Optional.of(user);
                }
            }
            
            return Optional.empty();
        }
        
        public List<User> getActiveUsers() {
            return users.stream()
                       .filter(User::isActive)
                       .collect(Collectors.toList());
        }
        
        public List<User> getUsersByAgeRange(int minAge, int maxAge) {
            return users.stream()
                       .filter(user -> user.getAge() >= minAge && user.getAge() <= maxAge)
                       .sorted(Comparator.comparing(User::getAge))
                       .collect(Collectors.toList());
        }
        
        private boolean validateEmail(String email) {
            return email != null && 
                   email.contains("@") && 
                   email.indexOf("@") < email.lastIndexOf(".");
        }
        
        public void processUsersInBatches(int batchSize) {
            for (int i = 0; i < users.size(); i += batchSize) {
                int endIndex = Math.min(i + batchSize, users.size());
                List<User> batch = users.subList(i, endIndex);
                processBatch(batch);
            }
        }
        
        private void processBatch(List<User> batch) {
            System.out.println("Processing batch of " + batch.size() + " users");
            batch.forEach(user -> System.out.println("  - " + user.getName()));
        }
        
        public Map<String, Object> calculateStatistics() {
            Map<String, Object> stats = new HashMap<>();
            
            if (users.isEmpty()) {
                stats.put("count", 0);
                stats.put("averageAge", 0.0);
                return stats;
            }
            
            int totalAge = users.stream()
                               .mapToInt(User::getAge)
                               .sum();
            
            double averageAge = (double) totalAge / users.size();
            
            OptionalInt minAge = users.stream()
                                     .mapToInt(User::getAge)
                                     .min();
            
            OptionalInt maxAge = users.stream()
                                     .mapToInt(User::getAge)
                                     .max();
            
            stats.put("count", users.size());
            stats.put("averageAge", averageAge);
            stats.put("minAge", minAge.orElse(0));
            stats.put("maxAge", maxAge.orElse(0));
            
            return stats;
        }
    }
    
    /**
     * Main method demonstrating usage.
     */
    public static void main(String[] args) {
        UserManager manager = new UserManager();
        
        // Sample data
        String[][] sampleUsers = {
            {"Alice Johnson", "alice@example.com", "28"},
            {"Bob Smith", "bob@example.com", "35"},
            {"Carol Davis", "carol@example.com", "42"},
            {"David Wilson", "david@example.com", "31"}
        };
        
        // Add users
        for (String[] userData : sampleUsers) {
            String name = userData[0];
            String email = userData[1];
            int age = Integer.parseInt(userData[2]);
            
            boolean success = manager.addUser(name, email, age);
            if (success) {
                System.out.println("Added user: " + name);
            } else {
                System.out.println("Failed to add user: " + name);
            }
        }
        
        // Find and display user
        String testEmail = "alice@example.com";
        Optional<User> user = manager.findUser(testEmail);
        if (user.isPresent()) {
            System.out.println("Found user: " + user.get());
        } else {
            System.out.println("User not found: " + testEmail);
        }
        
        // Get active users
        List<User> activeUsers = manager.getActiveUsers();
        System.out.println("Active users: " + activeUsers.size());
        
        // Get users by age range
        List<User> youngUsers = manager.getUsersByAgeRange(25, 35);
        System.out.println("Users aged 25-35: " + youngUsers.size());
        
        // Calculate statistics
        Map<String, Object> stats = manager.calculateStatistics();
        System.out.println("Statistics: " + stats);
        
        // Process in batches
        manager.processUsersInBatches(2);
    }
}