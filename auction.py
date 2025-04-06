import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, tools, algorithms
import warnings

# Suppress the specific RuntimeWarning about class creation
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="A class named .* has already been created")

# Clean up existing creator classes if they exist
if 'FitnessMax' in dir():
    del FitnessMax
if 'Individual' in dir():
    del Individual

# Define market parameters
class EnergyMarket:
    def __init__(self, num_players, player_capacities, demand):
        """
        Initialize the energy market simulation.
        
        Args:
            num_players (int): Number of market participants
            player_capacities (list): List of integer capacities for each player in MW
            demand (int): Total market demand in MW
        """
        self.num_players = num_players
        self.player_capacities = player_capacities
        self.demand = demand
        
        # Validate inputs
        assert len(player_capacities) == num_players, "Capacities list must match number of players"
        assert all(isinstance(cap, int) and cap > 0 for cap in player_capacities), "All capacities must be positive integers"
        assert demand > 0, "Demand must be positive"
        
        # Check if total capacity can meet demand
        total_capacity = sum(player_capacities)
        if total_capacity < demand:
            print(f"Warning: Total capacity ({total_capacity} MW) is less than demand ({demand} MW)")
    
    def run_auction(self, bids):
        """
        Run a single-clearing price auction.
        
        Args:
            bids (list): List of bid prices for each player ($/MW)
            
        Returns:
            tuple: (clearing_price, player_profits, total_cost, accepted_bids)
        """
        # Create bid objects with player index, capacity, and bid price
        bid_objects = []
        for i in range(self.num_players):
            bid_objects.append({
                'player': i,
                'capacity': self.player_capacities[i],
                'price': bids[i] #each price needs to be updated to array_
            })
        
        # Sort bids by price (ascending)
        sorted_bids = sorted(bid_objects, key=lambda x: x['price'])
        
        # Select bids until demand is met
        accepted_bids = []
        remaining_demand = self.demand
        clearing_price = 0
        
        for bid in sorted_bids:
            if remaining_demand <= 0:
                break
                
            # Accept all or part of the bid
            accepted_capacity = min(bid['capacity'], remaining_demand)
            remaining_demand -= accepted_capacity
            
            accepted_bids.append({
                'player': bid['player'],
                'capacity': accepted_capacity,
                'price': bid['price']
            })
            
            # The last accepted bid sets the clearing price
            clearing_price = bid['price']
        
        # If we couldn't meet demand, use the highest bid price
        if remaining_demand > 0:
            clearing_price = max(bids)
            
        # Calculate profits for each player
        player_profits = [0] * self.num_players     #equivalent to zeros(self.num_players) in MATLAB
        total_cost = 0
        
        for accepted_bid in accepted_bids:
            player = accepted_bid['player']
            capacity = accepted_bid['capacity']
            
            # All accepted bids are paid at the clearing price
            revenue = capacity * clearing_price
            
            # Assuming marginal cost is 0 for simplicity
            # In a real model, you would subtract costs here
            profit = revenue
            
            player_profits[player] = profit
            total_cost += revenue
            
        return clearing_price, player_profits, total_cost, accepted_bids

# Set up genetic algorithm for optimizing bidding strategies
class BiddingOptimizer:
    def __init__(self, market, player_index, min_bid, max_bid, population_size=50, 
                 num_generations=100, opponent_strategies=None):
        """
        Initialize the genetic algorithm optimizer for a specific player.
        
        Args:
            market (EnergyMarket): The energy market simulation
            player_index (int): Index of the player being optimized
            min_bid (float): Minimum bid price
            max_bid (float): Maximum bid price
            population_size (int): Size of GA population
            num_generations (int): Number of generations to evolve
            opponent_strategies (list): Fixed strategies for other players
        """
        self.market = market
        self.player_index = player_index
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.population_size = population_size
        self.num_generations = num_generations
        
        # If opponent strategies are not provided, use random pricing
        if opponent_strategies is None:
            self.opponent_strategies = [random.uniform(min_bid, max_bid) 
                                       for _ in range(market.num_players)]
        else:
            self.opponent_strategies = opponent_strategies
        
        # Import creator here to localize its use
        from deap import creator
        
        # Check if the classes already exist and only create them if they don't
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Define gene (bid price)       #needs to be updated to array_
        self.toolbox.register("attr_float", random.uniform, min_bid, max_bid)
        
        # Initialize individual and population
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, n=1)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual, n=population_size)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_strategy)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=(max_bid-min_bid)/10, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_strategy(self, individual):
        """
        Fitness function - evaluates a bidding strategy by simulating the market.
        
        Args:
            individual (list): A single bid price       #needs to be updated to array_
            
        Returns:
            tuple: (profit,) - Note the comma to make it a tuple for DEAP
        """
        # Create full bid list with this player's bid and other players' strategies
        bids = self.opponent_strategies.copy()
        bids[self.player_index] = individual[0]
        
        # Run auction and get this player's profit
        _, player_profits, _, _ = self.market.run_auction(bids)
        profit = player_profits[self.player_index]
        
        return (profit,)
    
    def optimize(self):
        """
        Run the genetic algorithm to find optimal bidding strategy.
        
        Returns:
            tuple: (best_bid, best_profit, stats)
        """
        # Create initial population
        pop = self.toolbox.population()
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run the algorithm
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.6, mutpb=0.2, 
                                          ngen=self.num_generations, stats=stats, 
                                          halloffame=hof, verbose=False)
        
        # Return the best individual
        best_strategy = hof[0]
        best_bid = best_strategy[0]
        best_profit = best_strategy.fitness.values[0]
        
        return best_bid, best_profit, logbook

# Run a multi-player iterative optimization
def optimize_all_players(market, min_bid, max_bid, iterations):
    """
    Iteratively optimize all players' strategies.
    
    Args:
        market (EnergyMarket): The energy market simulation
        min_bid (float): Minimum bid price
        max_bid (float): Maximum bid price
        iterations (int): Number of optimization rounds
        
    Returns:
        tuple: (final_strategies, profit_history)
    """
    # Initialize with random strategies
    strategies = [random.uniform(min_bid, max_bid) for _ in range(market.num_players)]
        #each element of strategies needs to be updated to array_
    profit_history = []
    
    # Run iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration+1}/{iterations}")
        iteration_profits = []
        
        # Optimize each player's strategy
        # This holds all opponents constant while optimizing the current player
        for player in range(market.num_players):
            # Create opponent strategies (current best for all other players)
            opponent_strategies = strategies.copy()
            
            # Optimize this player
            optimizer = BiddingOptimizer(market, player, min_bid, max_bid, 
                                        opponent_strategies=opponent_strategies)
            best_bid, best_profit, _ = optimizer.optimize() #best_bid needs to be updated to array_
            
            # Update this player's strategy
            strategies[player] = best_bid
            iteration_profits.append(best_profit)
            
            print(f"  Player {player+1} optimal bid: ${best_bid:.2f}, profit: ${best_profit:.2f}")
        
        # Record profits for this iteration
        profit_history.append(iteration_profits)
        
        # Run a market with current strategies to show clearing price
        clearing_price, _, total_cost, _ = market.run_auction(strategies)
        print(f"  Market clearing price: ${clearing_price:.2f}, total cost: ${total_cost:.2f}")
    
    return strategies, profit_history

# Visualization functions
def plot_optimization_results(market, final_strategies, profit_history):
    """
    Visualize the results of the optimization process.
    
    Args:
        market (EnergyMarket): The energy market simulation
        final_strategies (list): Final optimized bidding strategies
        profit_history (list): History of profits across iterations
    """
    # Plot 1: Final bidding strategies
    plt.figure(figsize=(10, 6))
    players = [f"Player {i+1}\n({market.player_capacities[i]} MW)" for i in range(market.num_players)]
    plt.bar(players, final_strategies)
    plt.title("Optimal Bidding Strategies")
    plt.ylabel("Bid Price ($/MW)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add capacity information
    for i, strategy in enumerate(final_strategies):
        plt.text(i, strategy + 0.5, f"${strategy:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("optimal_bids.png")
    plt.close()
    
    # Plot 2: Profit evolution
    plt.figure(figsize=(10, 6))
    iterations = range(1, len(profit_history) + 1)
    
    for player in range(market.num_players):
        player_profits = [profit_history[i][player] for i in range(len(profit_history))]
        plt.plot(iterations, player_profits, marker='o', label=f"Player {player+1}")
    
    plt.title("Profit Evolution Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Profit ($)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("profit_evolution.png")
    plt.close()

# Example usage
def run_simulation():
    # Market parameters
    num_players = 5
    player_capacities = [100, 150, 110, 125, 175]  # MW
    demand = 500  # MW
    
    # Bid constraints
    min_bid = 10  # $/MW
    max_bid = 500  # $/MW
    
    # Create market
    market = EnergyMarket(num_players, player_capacities, demand)
    
    # Run optimization
    final_strategies, profit_history = optimize_all_players(
        market, min_bid, max_bid, iterations=100)
    
    # Plot results
    plot_optimization_results(market, final_strategies, profit_history)
    
    # Run final market with optimized strategies
    clearing_price, player_profits, total_cost, accepted_bids = market.run_auction(final_strategies)
    
    print("\nFinal Market Results:")
    print(f"Clearing Price: ${clearing_price:.2f}/MW")
    print(f"Total Market Cost: ${total_cost:.2f}")
    print("\nPlayer Results:")
    
    for i in range(num_players):
        print(f"Player {i+1} (Capacity: {market.player_capacities[i]} MW):")
        print(f"  Bid: ${final_strategies[i]:.2f}/MW")
        print(f"  Profit: ${player_profits[i]:.2f}")
    
    print("\nAccepted Bids:")
    for bid in accepted_bids:
        player = bid['player']
        print(f"Player {player+1}: {bid['capacity']} MW at ${bid['price']:.2f}/MW")

if __name__ == "__main__":
    run_simulation()