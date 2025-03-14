import random
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  





class Player:
    def __init__(self, id, capacity, strategy):
        self.id = id
        self.capacity = capacity  # Integer MW
        self.strategy = strategy  # BiddingStrategy object
        self.profits = []
        
    def create_bid(self, market_state=None):
        # Return a list of (price, quantity) pairs
        return self.strategy.generate_bid(self.capacity, market_state)





class BiddingStrategy:
    def __init__(self, genes):
        """
        genes: Parameters that define the bidding strategy
        Could include:
        - base_markup: Base percentage above marginal cost
        - demand_sensitivity: How much to increase bid with higher demand
        - capacity_factor: How to adjust bids based on own capacity
        """
        self.genes = genes
        
    def generate_bid(self, capacity, market_state=None):
        """
        Generate a set of price-quantity pairs based on strategy
        market_state is a placeholder for future refinement (see comment block below)
        - E.g., demand forecasts, competitor behavior, season
        """
        # Example implementation
        base_price = self.genes['base_price']
        markup = self.genes['markup']
        slope = self.genes['slope']
        
        # Use market_state if available
        #if market_state and 'last_clearing_price' in market_state:
        #    # Adjust markup based on last clearing price
        #    last_price = market_state['last_clearing_price']
        #    markup = markup * (1.1 if last_price > base_price + markup else 0.9)

        # Generate stepped bids
        bids = []
        for i in range(capacity):
            # Price increases linearly with each additional MW
            price = base_price + markup + (i * slope)
            bids.append((price, 1))  # 1 MW per bid
            
        return bids






def run_auction(bids, demand):
    """
    bids: List of (player_id, price, quantity) tuples
    demand: Integer representing total MW needed
    
    Returns: (accepted_bids, clearing_price)
    """
    # Sort bids by price (ascending)
    sorted_bids = sorted(bids, key=lambda x: x[1])
    
    accepted_bids = []
    total_accepted = 0
    
    # Accept bids until demand is met
    for bid in sorted_bids:
        player_id, price, quantity = bid
        
        if total_accepted + quantity <= demand:
            # Accept full bid
            accepted_bids.append((player_id, price, quantity))
            total_accepted += quantity
        else:
            # Accept partial bid
            remaining = demand - total_accepted
            if remaining > 0:
                accepted_bids.append((player_id, price, remaining))
                total_accepted += remaining
            break
            
    # Determine clearing price (price of last accepted bid)
    clearing_price = accepted_bids[-1][1] if accepted_bids else 0
    
    return accepted_bids, clearing_price






class MarketSimulation:
    def __init__(self, players, demand):
        self.players = players
        self.demand = demand
        
    def run_single_round(self):
        # Collect bids from all players
        all_bids = []
        for player in self.players:
            player_bids = player.create_bid()
            for price, quantity in player_bids:
                all_bids.append((player.id, price, quantity))
        
        # Run the auction
        accepted_bids, clearing_price = run_auction(all_bids, self.demand)
        
        # Calculate and distribute profits
        self.distribute_profits(accepted_bids, clearing_price)
        
        return {
            'accepted_bids': accepted_bids,
            'clearing_price': clearing_price
        }
        
    def distribute_profits(self, accepted_bids, clearing_price):
        # Reset profits for this round
        player_profits = {player.id: 0 for player in self.players}
        
        # Calculate profits for each player
        for player_id, bid_price, quantity in accepted_bids:
            # In single clearing price auction, everyone gets paid the clearing price
            profit = (clearing_price - bid_price) * quantity
            player_profits[player_id] += profit
            
        # Update player profit history
        for player in self.players:
            player.profits.append(player_profits[player.id])







class GeneticAlgorithm:
    def __init__(self, population_size, gene_space):
        """
        population_size: Number of strategies to evolve
        gene_space: Dict of parameter ranges for strategies
        """
        self.population_size = population_size
        self.gene_space = gene_space
        self.population = self.initialize_population()
        
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Create random genes within the specified ranges
            genes = {}
            for param, (min_val, max_val) in self.gene_space.items():
                genes[param] = np.random.uniform(min_val, max_val)
            
            strategy = BiddingStrategy(genes)
            population.append(strategy)
            
        print(f"Genes created: {genes}")
        
        return population
    
    def evaluate_fitness(self, strategies, num_rounds=100):
        """
        Test strategies in the market simulation
        """
        # Create players with different strategies
        players = []
        for i, strategy in enumerate(strategies):
            # Each player has same capacity for evaluation
            players.append(Player(i, capacity=10, strategy=strategy))
        
        # Run simulation
        # demand is set here <<== demand curve interface
        simulation = MarketSimulation(players, demand=50)
        for _ in range(num_rounds):
            simulation.run_single_round()
        
        # Return fitness scores (average profit per round)
        return [sum(player.profits) / num_rounds for player in players]
    
    def select_parents(self, strategies, fitness_scores):
        # Select parents with probability proportional to fitness
        total_fitness = sum(fitness_scores) 
        
        # Handle the case where all fitness scores are zero
        if total_fitness <= 0:
            # Just select randomly if all scores are zero
            parent_indices = np.random.choice(
                range(len(strategies)), 
                size=2, 
                replace=False
            )
        else:
            selection_probs = [f/total_fitness for f in fitness_scores]
            
            # Select two parents
            parent_indices = np.random.choice(
                range(len(strategies)), 
                size=2, 
                p=selection_probs,
                replace=False
            )
        
        return [strategies[i] for i in parent_indices]
    
    def crossover(self, parent1, parent2):
        # Create a child strategy by combining genes from parents
        child_genes = {}
        for gene in parent1.genes:
            # 50% chance of inheriting from each parent
            if random.random() < 0.5:
                child_genes[gene] = parent1.genes[gene]
            else:
                child_genes[gene] = parent2.genes[gene]
                
        return BiddingStrategy(child_genes)
    
    def mutate(self, strategy, mutation_rate=0.1):
        # Randomly modify genes with probability mutation_rate
        new_genes = strategy.genes.copy()
        for gene in new_genes:
            if random.random() < mutation_rate:
                min_val, max_val = self.gene_space[gene]
                # Add random noise
                new_genes[gene] += random.uniform(-0.1, 0.1) * (max_val - min_val)
                # Keep within bounds
                new_genes[gene] = max(min_val, min(max_val, new_genes[gene]))
                
        return BiddingStrategy(new_genes)
    
    def evolve_generation(self):
        # Evaluate current population
        fitness_scores = self.evaluate_fitness(self.population)
        
        new_population = []
        
        # Keep the best strategy (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population.append(self.population[best_idx])
        
        # Create the rest through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Select parents
            parents = self.select_parents(self.population, fitness_scores)
            
            # Create child through crossover
            child = self.crossover(parents[0], parents[1])
            
            # Mutate
            child = self.mutate(child)
            
            new_population.append(child)
            
        self.population = new_population
        
        #print(f"Best fitness is {fitness_scores[best_idx]}")
    
        return fitness_scores[best_idx]  # Return best fitness
    




def main():
    # Define gene space for strategies
    gene_space = {
        'base_price': (10, 50),
        'markup': (0, 20),
        'slope': (0, 2)
    }
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size=50, gene_space=gene_space)
    
    # Run for many generations
    best_fitness_history = []
    
    for generation in range(100):
        best_fitness = ga.evolve_generation()
        best_fitness_history.append(best_fitness)
            
    # Get best strategy
    best_strategy = ga.population[0]  # After evolution, this is the best
    
    # Run a final simulation with the best strategy
    # against some baseline strategies
    
    # Visualize results
    plt.plot(best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Profit)')
    plt.title('Evolution of Bidding Strategy')
    plt.show()

    #print(f"best_fitness_history: {best_fitness_history} ")
    # print("\nBest Fitness History (Generation: Fitness):")
    # for gen, fitness in enumerate(best_fitness_history):
    #     print(f"Generation {gen}: {fitness}")
    
    # with open("fitness_history.csv", "w") as f:
    #     f.write("Generation,Fitness\n")
    #     for gen, fitness in enumerate(best_fitness_history):
    #         f.write(f"{gen},{fitness}\n")
    #print("Fitness history saved to fitness_history.csv")

    # Print summary statistics
    # print("\nFitness History Statistics:", ',', end='', flush=True)
    # print(f"Initial fitness: {best_fitness_history[0]}", ',', end='', flush=True)
    # print(f"Final fitness: {best_fitness_history[-1]}", ',', end='', flush=True)
    # print(f"Improvement: {best_fitness_history[-1] - best_fitness_history[0]}", ',', end='', flush=True)
    # print(f"Improvement percentage: {(best_fitness_history[-1] / best_fitness_history[0] - 1) * 100:.2f}%", ',', end='', flush=True)
    
if __name__ == "__main__":
    main()

