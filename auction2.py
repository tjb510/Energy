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
    def __init__(self, num_players, player_capacities, demand, production_costs=None):
        """
        Initialize the energy market simulation.
        
        Args:
            num_players (int): Number of market participants
            player_capacities (list): List of integer capacities for each player in MW
            demand (int): Total market demand in MW
            production_costs (list of lists): Production costs for each 10MW chunk for each player
        """
        self.num_players = num_players
        
        # Ensure all capacities are integer multiples of 10
        self.player_capacities = [self._round_to_10(cap) for cap in player_capacities]
        
        # Calculate number of 10MW chunks for each player
        self.player_chunks = [capacity // 10 for capacity in self.player_capacities]
        
        # Ensure demand is a multiple of 10
        self.demand = self._round_to_10(demand)
        
        # Set production costs
        if production_costs is None:
            # Default to zero costs if not provided
            self.production_costs = [[0 for _ in range(chunks)] for chunks in self.player_chunks]
        else:
            # Validate and set provided costs
            assert len(production_costs) == num_players, "Production costs must match number of players"
            for i, costs in enumerate(production_costs):
                assert len(costs) == self.player_chunks[i], f"Player {i} has {self.player_chunks[i]} chunks but {len(costs)} costs provided"
            self.production_costs = production_costs
        
        # Validate inputs
        assert len(player_capacities) == num_players, "Capacities list must match number of players"
        assert all(isinstance(cap, int) and cap > 0 for cap in self.player_capacities), "All capacities must be positive integers"
        assert self.demand > 0, "Demand must be positive"
        
        # Check if total capacity can meet demand
        total_capacity = sum(self.player_capacities)
        if total_capacity < self.demand:
            print(f"Warning: Total capacity ({total_capacity} MW) is less than demand ({self.demand} MW)")
    
    def _round_to_10(self, value):
        """Round a value to the nearest multiple of 10"""
        return int(round(value / 10.0)) * 10
    
    def run_auction(self, all_player_bids):
        """
        Run a single-clearing price auction with chunked bids.
        
        Args:
            all_player_bids (list): List of bid arrays for each player, where each array contains
                                    bid prices for each 10MW chunk of capacity
            
        Returns:
            tuple: (clearing_price, player_profits, total_cost, accepted_bids, player_accepted_capacities)
        """
        # Create bid objects with player index, chunk index, capacity (always 10MW), bid price, and production cost
        bid_objects = []
        for player_idx, player_bids in enumerate(all_player_bids):
            for chunk_idx, price in enumerate(player_bids):
                cost = self.production_costs[player_idx][chunk_idx]
                bid_objects.append({
                    'player': player_idx,
                    'chunk': chunk_idx,
                    'capacity': 10,  # Each chunk is 10MW
                    'price': price,
                    'cost': cost
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
                
            # Each chunk is 10MW, so we either take all or none
            accepted_capacity = 10  # Fixed at 10MW per chunk
            remaining_demand -= accepted_capacity
            
            accepted_bids.append({
                'player': bid['player'],
                'chunk': bid['chunk'],
                'capacity': accepted_capacity,
                'price': bid['price'],
                'cost': bid['cost']
            })
            
            # The last accepted bid sets the clearing price
            clearing_price = bid['price']
        
        # If we couldn't meet demand, use the highest bid price of any chunk
        if remaining_demand > 0:
            all_prices = [price for player_bids in all_player_bids for price in player_bids]
            clearing_price = max(all_prices)
            
        # Calculate profits for each player
        player_profits = [0] * self.num_players
        total_cost = 0
        player_accepted_capacities = [0] * self.num_players
        
        for accepted_bid in accepted_bids:
            player = accepted_bid['player']
            capacity = accepted_bid['capacity']
            production_cost = accepted_bid['cost']
            
            # All accepted bids are paid at the clearing price
            revenue = capacity * clearing_price
            
            # Now we consider production costs
            profit = revenue - (capacity * production_cost)
            
            player_profits[player] += profit
            player_accepted_capacities[player] += capacity
            total_cost += revenue
            
        return clearing_price, player_profits, total_cost, accepted_bids, player_accepted_capacities

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
        self.num_chunks = self.market.player_chunks[player_index]
        self.costs = self.market.production_costs[player_index]
        
        # If opponent strategies are not provided, initialize with random strategies
        if opponent_strategies is None:
            self.opponent_strategies = []
            for p in range(market.num_players):
                num_chunks = market.player_chunks[p]
                player_bids = [random.uniform(min_bid, max_bid) for _ in range(num_chunks)]
                self.opponent_strategies.append(player_bids)
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
        
        # Define gene initialization based on chunk cost (start bidding above cost)
        def init_bid_for_chunk(chunk_idx):
            cost = self.costs[chunk_idx]
            # Initialize with a price above cost, but below max_bid
            min_price = max(self.min_bid, cost * 1.05)  # At least 5% above cost
            max_price = min(self.max_bid, cost * 3)     # Up to 3x the cost
            return random.uniform(min_price, max_price)
        
        # Custom initialization of each gene/chunk based on production cost
        def init_individual():
            return [init_bid_for_chunk(chunk) for chunk in range(self.num_chunks)]
        
        # Register custom initialization
        self.toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=population_size)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_strategy)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        
        # Custom mutation that respects costs
        def custom_mutate(individual, mu, sigma, indpb):
            for i in range(len(individual)):
                if random.random() < indpb:
                    # Apply Gaussian mutation but ensure price stays above cost
                    cost = self.costs[i]
                    individual[i] += random.gauss(mu, sigma)
                    individual[i] = max(cost * 1.01, min(individual[i], self.max_bid))
            return individual,
        
        self.toolbox.register("mutate", custom_mutate, mu=0, sigma=(max_bid-min_bid)/10, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_strategy(self, individual):
        """
        Fitness function - evaluates a bidding strategy by simulating the market.
        
        Args:
            individual (list): Array of bid prices for each 10MW chunk
            
        Returns:
            tuple: (profit,) - Note the comma to make it a tuple for DEAP
        """
        # Create full bid list with this player's bid and other players' strategies
        all_bids = self.opponent_strategies.copy()
        all_bids[self.player_index] = individual
        
        # Run auction and get this player's profit
        _, player_profits, _, _, _ = self.market.run_auction(all_bids)
        profit = player_profits[self.player_index]
        
        return (profit,)
    
    def optimize(self):
        """
        Run the genetic algorithm to find optimal bidding strategy.
        
        Returns:
            tuple: (best_bids, best_profit, stats)
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
        best_bids = list(best_strategy)  # Convert from deap Individual to regular list
        best_profit = best_strategy.fitness.values[0]
        
        return best_bids, best_profit, logbook

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
    # Initialize with strategies based on costs
    strategies = []
    for p in range(market.num_players):
        num_chunks = market.player_chunks[p]
        player_costs = market.production_costs[p]
        
        # Initialize with cost-plus pricing (costs + random markup)
        player_bids = [cost * (1.1 + 0.5 * random.random()) for cost in player_costs]
        strategies.append(player_bids)
    
    profit_history = []
    
    # Run iterations
    for iteration in range(iterations):
        print(f"Iteration {iteration+1}/{iterations}")
        iteration_profits = []
        
        # Optimize each player's strategy
        for player in range(market.num_players):
            # Create opponent strategies (current best for all other players)
            opponent_strategies = strategies.copy()
            
            # Optimize this player
            optimizer = BiddingOptimizer(market, player, min_bid, max_bid, 
                                        opponent_strategies=opponent_strategies)
            best_bids, best_profit, _ = optimizer.optimize()
            
            # Update this player's strategy
            strategies[player] = best_bids
            iteration_profits.append(best_profit)
            
            chunk_bid_str = ", ".join([f"${bid:.2f}" for bid in best_bids])
            print(f"  Player {player+1} optimal bids: [{chunk_bid_str}], profit: ${best_profit:.2f}")
        
        # Record profits for this iteration
        profit_history.append(iteration_profits)
        
        # Run a market with current strategies to show clearing price
        clearing_price, _, total_cost, _, accepted_capacities = market.run_auction(strategies)
        print(f"  Market clearing price: ${clearing_price:.2f}, total cost: ${total_cost:.2f}")
        print(f"  Accepted capacities: {accepted_capacities} MW")
    
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
    # Plot 1: Final bidding strategies as supply curves with costs
    plt.figure(figsize=(12, 8))
    
    for player in range(market.num_players):
        x_values = []
        y_values = []
        cost_x = []
        cost_y = []
        chunk_size = 10  # MW
        
        # Build the supply curve for each player
        for chunk in range(market.player_chunks[player]):
            capacity_point = chunk * chunk_size
            x_values.append(capacity_point)
            x_values.append(capacity_point + chunk_size)
            y_values.append(final_strategies[player][chunk])
            y_values.append(final_strategies[player][chunk])
            
            # Also plot production costs
            cost_x.append(capacity_point)
            cost_x.append(capacity_point + chunk_size)
            cost_y.append(market.production_costs[player][chunk])
            cost_y.append(market.production_costs[player][chunk])
        
        # Plot bid supply curve
        plt.step(x_values, y_values, where='post', 
                 label=f"Player {player+1} Bids ({market.player_capacities[player]} MW)", 
                 linewidth=2)
        
        # Plot cost curve (dashed line)
        plt.step(cost_x, cost_y, where='post', 
                 label=f"Player {player+1} Costs", 
                 linewidth=1, linestyle='--', alpha=0.7)
    
    plt.title("Supply Curves with Production Costs")
    plt.xlabel("Capacity (MW)")
    plt.ylabel("Price/Cost ($/MW)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("supply_curves_with_costs.png")
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
    
    # Plot 3: Market supply curve with accepted bids and clearing price
    plt.figure(figsize=(12, 8))
    
    # Create a list of all bid objects
    all_bids = []
    for player in range(market.num_players):
        for chunk in range(market.player_chunks[player]):
            all_bids.append({
                'player': player,
                'chunk': chunk,
                'capacity': 10,
                'price': final_strategies[player][chunk],
                'cost': market.production_costs[player][chunk]
            })
    
    # Sort bids by price
    sorted_bids = sorted(all_bids, key=lambda x: x['price'])
    
    # Create market supply curve
    x_values = [0]
    y_values = [sorted_bids[0]['price']]
    
    # Create cost curve
    cost_x = [0]
    cost_y = [sorted_bids[0]['cost']]
    
    cumulative_capacity = 0
    
    for bid in sorted_bids:
        cumulative_capacity += bid['capacity']
        x_values.append(cumulative_capacity)
        y_values.append(bid['price'])
        cost_x.append(cumulative_capacity)
        cost_y.append(bid['cost'])
    
    # Draw the supply curve
    plt.step(x_values, y_values, where='post', color='blue', linewidth=3, label="Market Supply Curve")
    
    # Draw the cost curve
    plt.step(cost_x, cost_y, where='post', color='green', linewidth=2, 
             linestyle='--', label="Market Cost Curve", alpha=0.7)
    
    # Run a market simulation to get clearing price
    clearing_price, _, _, _, _ = market.run_auction([strat for strat in final_strategies])
    
    # Add demand line
    plt.axvline(x=market.demand, color='red', linestyle='--', linewidth=2, 
                label=f"Demand ({market.demand} MW)")
    
    # Add clearing price line
    plt.axhline(y=clearing_price, color='orange', linestyle='-.', linewidth=2,
                label=f"Clearing Price (${clearing_price:.2f}/MW)")
    
    plt.title("Market Supply Curve with Costs")
    plt.xlabel("Cumulative Capacity (MW)")
    plt.ylabel("Price ($/MW)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("market_supply_curve.png")
    plt.close()
    
    # Plot 4: Markup over costs
    plt.figure(figsize=(12, 8))
    
    for player in range(market.num_players):
        x_values = list(range(market.player_chunks[player]))
        costs = market.production_costs[player]
        bids = final_strategies[player]
        
        # Calculate markup percentage for each chunk
        markups = [(bids[i] / costs[i] - 1) * 100 if costs[i] > 0 else 0 for i in range(len(costs))]
        
        plt.bar([x + 0.1*player for x in x_values], markups, 
                width=0.1, label=f"Player {player+1}")
    
    plt.title("Bid Markup Over Production Costs")
    plt.xlabel("Capacity Chunk (10MW each)")
    plt.ylabel("Markup (%)")
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig("markup_over_costs.png")
    plt.close()
    
    # Plot 5: NEW - Markup vs Cost Scatterplot (to verify relationship between cost and markup)
    plt.figure(figsize=(12, 8))
    
    all_costs = []
    all_markups = []
    player_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for player in range(market.num_players):
        costs = market.production_costs[player]
        bids = final_strategies[player]
        
        # Calculate markup percentage for each chunk
        markups = [(bids[i] / costs[i] - 1) * 100 if costs[i] > 0 else 0 for i in range(len(costs))]
        
        # Scatter plot of cost vs markup
        plt.scatter(costs, markups, 
                   label=f"Player {player+1}", 
                   color=player_colors[player % len(player_colors)],
                   alpha=0.7, s=100)
        
        # Add arrows connecting points in order of chunks
        for i in range(len(costs) - 1):
            plt.annotate("", 
                        xy=(costs[i+1], markups[i+1]), 
                        xytext=(costs[i], markups[i]),
                        arrowprops=dict(arrowstyle="->", color=player_colors[player % len(player_colors)], alpha=0.5))
            
        # Collect data for trend line
        all_costs.extend(costs)
        all_markups.extend(markups)
    
    # Add a trend line
    if len(all_costs) > 1:  # Need at least 2 points for a trend line
        z = np.polyfit(all_costs, all_markups, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(all_costs), max(all_costs), 100)
        plt.plot(x_trend, p(x_trend), "k--", alpha=0.8, label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")
    
    plt.title("Relationship Between Production Cost and Markup Percentage")
    plt.xlabel("Production Cost ($/MW)")
    plt.ylabel("Markup Percentage (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add text annotation explaining the hypothesis
    plt.figtext(0.5, 0.01, 
               "Hypothesis: Higher cost chunks should have lower markup percentages\n"
               "If true, we should see a downward trend in this plot.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for the text at the bottom
    plt.savefig("cost_vs_markup.png")
    plt.close()
    
    # Plot 6: NEW - Markup vs Relative Cost Position
    plt.figure(figsize=(12, 8))
    
    for player in range(market.num_players):
        costs = market.production_costs[player]
        bids = final_strategies[player]
        
        # Normalize costs to 0-1 range within each player
        if max(costs) > min(costs):  # Avoid division by zero
            norm_costs = [(cost - min(costs)) / (max(costs) - min(costs)) for cost in costs]
        else:
            norm_costs = [0.5] * len(costs)  # If all costs are the same
            
        # Calculate markup percentage for each chunk
        markups = [(bids[i] / costs[i] - 1) * 100 if costs[i] > 0 else 0 for i in range(len(costs))]
        
        # Scatter plot of normalized cost vs markup
        plt.scatter(norm_costs, markups, 
                   label=f"Player {player+1}", 
                   color=player_colors[player % len(player_colors)],
                   alpha=0.7, s=100)
                   
        # Fit line for this player's data
        if len(costs) > 1:  # Need at least 2 points
            z = np.polyfit(norm_costs, markups, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 10)
            plt.plot(x_trend, p(x_trend), linestyle="--", 
                    color=player_colors[player % len(player_colors)], alpha=0.5)
        
        # Add player's cost range in the legend
        plt.legend(title="                       Cost Range")._legend_box.align = "left"
        leg = plt.gca().get_legend()
        leg.get_title().set_position((-40, 0))  # Adjust title position
    
    plt.title("Markup Percentage vs. Relative Cost Position Within Each Player's Portfolio")
    plt.xlabel("Normalized Cost Position (0 = Lowest, 1 = Highest)")
    plt.ylabel("Markup Percentage (%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
               "This plot shows how markup varies across each player's cost range.\n"
               "A negative slope indicates higher markups on lower-cost chunks.",
               ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("relative_cost_vs_markup.png")
    plt.close()

# Example usage with production costs
def run_simulation():
    # Market parameters
    num_players = 5
    # All capacities will be rounded to multiples of 10
    player_capacities = [100, 150, 110, 120, 180]  # MW
    demand = 500  # MW
    
    # Bid constraints
    min_bid = 10  # $/MW
    max_bid = 500  # $/MW
    
    # Define production costs for each player's 10MW chunks
    # Each player has different cost structure
    production_costs = [
        # Player 1: Low base cost, increases sharply (e.g., efficient combined cycle plant + peaker)
        [20, 22, 25, 30, 45, 60, 80, 100, 150, 200],
        
        # Player 2: Moderate and fairly flat costs (e.g., coal plant)
        [30, 32, 34, 36, 38, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80],
        
        # Player 3: Start low but increase steadily (e.g., older gas plant)
        [25, 28, 32, 38, 45, 55, 70, 90, 120, 150, 180],
        
        # Player 4: High but flat costs (e.g., oil plant)
        [50, 52, 54, 56, 58, 60, 62, 65, 70, 75, 80, 85],
        
        # Player 5: Very low base cost, then significant increases (e.g., nuclear + gas backup)
        [10, 12, 15, 18, 22, 30, 45, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    ]
    
    # Create market with production costs
    market = EnergyMarket(num_players, player_capacities, demand, production_costs)
    
    # Print the adjusted capacities, chunks, and costs
    print("Market Setup:")
    for i, capacity in enumerate(market.player_capacities):
        print(f"Player {i+1}: {capacity} MW, {market.player_chunks[i]} chunks of 10MW each")
        print(f"  Production costs per chunk: {['$'+str(cost) for cost in market.production_costs[i]]}")
    print(f"Adjusted demand: {market.demand} MW")
    
    # Run optimization with fewer iterations for demo
    final_strategies, profit_history = optimize_all_players(
        market, min_bid, max_bid, iterations=50)
    
    # Plot results
    plot_optimization_results(market, final_strategies, profit_history)
    
    # Run final market with optimized strategies
    clearing_price, player_profits, total_cost, accepted_bids, accepted_capacities = market.run_auction(final_strategies)
    
    print("\nFinal Market Results:")
    print(f"Clearing Price: ${clearing_price:.2f}/MW")
    print(f"Total Market Cost: ${total_cost:.2f}")
    print("\nPlayer Results:")
    
    # Analyze relationship between cost and markup
    all_costs = []
    all_markups = []
    accepted_costs = []
    accepted_markups = []
    
    # Set of accepted chunks for quick lookup
    accepted_chunks = set()
    for bid in accepted_bids:
        accepted_chunks.add((bid['player'], bid['chunk']))
    
    total_profit = 0
    for i in range(num_players):
        print(f"Player {i+1} (Capacity: {market.player_capacities[i]} MW):")
        
        # Display bids, costs, and markups side by side
        print("  Chunk | Bid Price | Cost | Markup | Status")
        print("  -------------------------------------------")
        for chunk in range(market.player_chunks[i]):
            bid = final_strategies[i][chunk]
            cost = market.production_costs[i][chunk]
            markup = ((bid/cost) - 1) * 100 if cost > 0 else 0
            
            # Collect data for correlation analysis
            all_costs.append(cost)
            all_markups.append(markup)
            
            # Track if the bid was accepted
            status = "Accepted" if (i, chunk) in accepted_chunks else "Rejected"
            if status == "Accepted":
                accepted_costs.append(cost)
                accepted_markups.append(markup)
            
            print(f"  {chunk+1:2d}    | ${bid:6.2f}   | ${cost:4.2f} | {markup:6.1f}% | {status}")
            
        print(f"  Accepted Capacity: {accepted_capacities[i]} MW")
        print(f"  Profit: ${player_profits[i]:.2f}")
        total_profit += player_profits[i]
    
    print(f"\nTotal Market Profit: ${total_profit:.2f}")
    print(f"Market Efficiency: {total_profit/total_cost*100:.1f}%")
    
    # Calculate correlation between cost and markup
    from scipy import stats
    if len(all_costs) > 1:
        correlation, p_value = stats.pearsonr(all_costs, all_markups)
        print(f"\nCorrelation Analysis:")
        print(f"Correlation between cost and markup: {correlation:.3f} (p-value: {p_value:.3f})")
        if correlation < 0:
            print("The negative correlation confirms the hypothesis: players bid with higher markups on lower-cost chunks.")
        else:
            print("The correlation is positive or zero, suggesting players are not bidding with higher markups on lower-cost chunks.")
    
    # Calculate average markup for accepted vs rejected bids
    if accepted_markups:
        avg_accepted_markup = sum(accepted_markups) / len(accepted_markups)
        rejected_markups = [m for i, m in enumerate(all_markups) if (all_costs[i] not in accepted_costs)]
        avg_rejected_markup = sum(rejected_markups) / len(rejected_markups) if rejected_markups else 0
        
        print(f"\nMarkup Analysis:")
        print(f"Average markup for accepted bids: {avg_accepted_markup:.1f}%")
        print(f"Average markup for rejected bids: {avg_rejected_markup:.1f}%")
    
    print("\nAccepted Bids (by ascending price):")
    for bid in accepted_bids:
        player = bid['player']
        chunk = bid['chunk']
        profit = (clearing_price - bid['cost']) * bid['capacity']
        print(f"Player {player+1}, Chunk {chunk+1}: {bid['capacity']} MW at ${bid['price']:.2f}/MW " +
              f"(Cost: ${bid['cost']:.2f}/MW, Profit: ${profit:.2f})")

if __name__ == "__main__":
    run_simulation()