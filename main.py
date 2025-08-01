import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd


@dataclass
class MarketState:
    prices: List[float]
    quantities: List[float]
    profits: List[float]
    market_share: List[float]


class Firm:
    def __init__(self, firm_id: int, initial_price: float,
                 learning_rate: float = 0.1):
        self.firm_id = firm_id
        self.price = initial_price
        self.learning_rate = learning_rate
        self.price_history = [initial_price]
        self.profit_history = []
        self.quantity_history = []

        # Q-learning parameters for algorithmic pricing
        self.q_table = {}  # State-action value table
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9  # Discount factor
        self.last_state = None
        self.last_action = None

        # Price adjustment parameters
        self.min_price = 5.0
        self.max_price = 25.0
        self.price_step = 0.5

    def get_state(self, competitor_prices: List[float],
                  own_profit: float) -> str:
        """Convert market conditions to a discrete state for Q-learning"""
        avg_competitor_price = np.mean(competitor_prices)
        price_diff = self.price - avg_competitor_price
        profit_level = "high" if own_profit > 50 else "medium" if own_profit > 20 else "low"

        if price_diff > 2:
            position = "above"
        elif price_diff < -2:
            position = "below"
        else:
            position = "equal"

        return f"{position}_{profit_level}"

    def get_possible_actions(self) -> List[str]:
        """Define possible pricing actions"""
        return ["increase", "decrease", "maintain"]

    def choose_action(self, state: str) -> str:
        """Choose action using epsilon-greedy strategy"""
        actions = self.get_possible_actions()

        # Initialize Q-values for new states
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in actions}

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return max(actions, key=lambda a: self.q_table[state][a])

    def update_q_value(self, state: str, action: str, reward: float,
                       next_state: str):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.get_possible_actions()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in
                                        self.get_possible_actions()}

        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        # Q-learning update rule
        self.q_table[state][action] = current_q + self.learning_rate * (
                reward + self.gamma * max_next_q - current_q
        )

    def set_price(self, competitor_prices: List[float], own_profit: float,
                  last_profit: float):
        """Set price using algorithmic decision making"""
        current_state = self.get_state(competitor_prices, own_profit)
        action = self.choose_action(current_state)

        # Calculate reward based on profit change
        reward = own_profit - last_profit if last_profit is not None else 0

        # Update Q-value from previous action
        if self.last_state and self.last_action:
            self.update_q_value(self.last_state, self.last_action, reward,
                                current_state)

        # Execute chosen action
        if action == "increase":
            self.price = min(self.max_price, self.price + self.price_step)
        elif action == "decrease":
            self.price = max(self.min_price, self.price - self.price_step)
        # "maintain" keeps price unchanged

        self.price_history.append(self.price)
        self.last_state = current_state
        self.last_action = action


class OligopolyMarket:
    def __init__(self, n_firms: int = 3, market_size: float = 1000):
        self.n_firms = n_firms
        self.market_size = market_size
        self.firms = []

        # Initialize firms with slightly different starting prices
        for i in range(n_firms):
            initial_price = 10 + random.uniform(-2, 2)
            self.firms.append(Firm(i, initial_price))

        self.history = []

    def calculate_demand(self, prices: List[float]) -> List[float]:
        """Calculate demand for each firm using logit model with outside option"""
        # Price sensitivity parameter
        beta = 0.3

        # Calculate utilities for each firm (higher for lower prices)
        firm_utilities = [-beta * p for p in prices]

        # Add random preference shocks to firm utilities
        firm_utilities = [u + random.gauss(0, 0.1) for u in firm_utilities]

        # Outside option utility (not buying at all)
        # This represents consumer surplus from not purchasing
        outside_utility = 0.0  # Normalized to 0

        # All utilities including outside option
        all_utilities = firm_utilities + [outside_utility]

        # Logit probabilities including outside option
        exp_utilities = [np.exp(u) for u in all_utilities]
        sum_exp = sum(exp_utilities)

        # Calculate choice probabilities
        choice_probabilities = [exp_u / sum_exp for exp_u in exp_utilities]

        # Firm probabilities (excluding outside option)
        firm_probabilities = choice_probabilities[:-1]
        outside_probability = choice_probabilities[-1]

        # Total market participation (1 - outside option probability)
        market_participation = 1 - outside_probability

        # Quantities: firm probability * participating customers
        quantities = [prob * self.market_size * market_participation for prob in
                      firm_probabilities]

        return quantities

    def calculate_profits(self, prices: List[float], quantities: List[float]) -> \
    List[float]:
        """Calculate profits assuming constant marginal cost"""
        marginal_cost = 5.0  # Same for all firms
        profits = [(p - marginal_cost) * q for p, q in zip(prices, quantities)]
        return profits

    def simulate_period(self) -> MarketState:
        """Simulate one time period"""
        prices = [firm.price for firm in self.firms]
        quantities = self.calculate_demand(prices)
        profits = self.calculate_profits(prices, quantities)
        market_shares = [q / sum(quantities) for q in quantities]

        # Update firm histories
        for i, firm in enumerate(self.firms):
            firm.profit_history.append(profits[i])
            firm.quantity_history.append(quantities[i])

        # Firms update their prices for next period
        for i, firm in enumerate(self.firms):
            competitor_prices = [p for j, p in enumerate(prices) if j != i]
            last_profit = firm.profit_history[-2] if len(
                firm.profit_history) > 1 else None
            firm.set_price(competitor_prices, profits[i], last_profit)

        return MarketState(prices, quantities, profits, market_shares)

    def run_simulation(self, n_periods: int = 200) -> List[MarketState]:
        """Run the full simulation"""
        for period in range(n_periods):
            state = self.simulate_period()
            self.history.append(state)

            # Gradually reduce exploration (epsilon decay)
            for firm in self.firms:
                firm.epsilon = max(0.01, firm.epsilon * 0.995)

        return self.history


def calculate_monopoly_price(market_size: float = 1000, beta: float = 0.3,
                             marginal_cost: float = 5.0) -> float:
    """Calculate theoretical monopoly price that maximizes total profit with outside option"""
    best_price = marginal_cost
    best_profit = 0

    # Search for profit-maximizing price
    for price in np.arange(marginal_cost + 0.1, 30.0, 0.1):
        # Single firm monopoly demand with outside option
        firm_utility = -beta * price
        outside_utility = 0.0

        # Logit choice probabilities
        exp_firm = np.exp(firm_utility)
        exp_outside = np.exp(outside_utility)
        sum_exp = exp_firm + exp_outside

        # Probability of choosing the firm (vs outside option)
        choice_prob = exp_firm / sum_exp
        quantity = choice_prob * market_size
        profit = (price - marginal_cost) * quantity

        if profit > best_profit:
            best_profit = profit
            best_price = price

    return best_price


def analyze_collusion(market_history: List[MarketState]) -> dict:
    """Analyze the degree of tacit collusion in the market"""
    n_periods = len(market_history)
    n_firms = len(market_history[0].prices)

    # Calculate price convergence
    final_periods = market_history[-50:]  # Last 50 periods
    price_variance = []

    for state in final_periods:
        price_variance.append(np.var(state.prices))

    avg_price_variance = np.mean(price_variance)

    # Calculate average prices over time
    avg_prices_over_time = [np.mean(state.prices) for state in market_history]

    # Calculate proper benchmarks
    monopoly_price = calculate_monopoly_price()
    competitive_price = 5.5  # Slightly above marginal cost (5.0) for sustainable competition

    final_avg_price = np.mean(avg_prices_over_time[-50:])

    # Collusion index (0 = competitive, 1 = monopoly)
    if monopoly_price > competitive_price:
        collusion_index = (final_avg_price - competitive_price) / (
                    monopoly_price - competitive_price)
        collusion_index = max(0, min(1, collusion_index))
    else:
        collusion_index = 0

    # If prices exceed monopoly price, calculate "super-collusion" index
    super_collusion = 0
    if final_avg_price > monopoly_price:
        super_collusion = (final_avg_price - monopoly_price) / monopoly_price

    return {
        'avg_price_variance': avg_price_variance,
        'final_average_price': final_avg_price,
        'collusion_index': collusion_index,
        'super_collusion_index': super_collusion,
        'monopoly_benchmark': monopoly_price,
        'competitive_benchmark': competitive_price
    }


def plot_simulation_results(market: OligopolyMarket, analysis: dict):
    """Create visualization of simulation results"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,
                                                             figsize=(15, 18))

    periods = range(len(market.history))

    # Plot 1: Price evolution
    for i, firm in enumerate(market.firms):
        ax1.plot(periods, firm.price_history[:-1], label=f'Firm {i + 1}',
                 alpha=0.8)

    ax1.axhline(y=analysis['monopoly_benchmark'], color='red', linestyle='--',
                label='Monopoly Price', alpha=0.7)
    ax1.axhline(y=analysis['competitive_benchmark'], color='green',
                linestyle='--',
                label='Competitive Price', alpha=0.7)
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Price')
    ax1.set_title('Price Evolution Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Profit evolution
    for i, firm in enumerate(market.firms):
        ax2.plot(periods, firm.profit_history, label=f'Firm {i + 1}', alpha=0.8)

    ax2.set_xlabel('Period')
    ax2.set_ylabel('Profit')
    ax2.set_title('Profit Evolution Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Price variance (measure of coordination)
    price_variance = [np.var(state.prices) for state in market.history]
    ax3.plot(periods, price_variance, color='purple', linewidth=2)
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Price Variance')
    ax3.set_title('Price Variance (Lower = More Coordination)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Market concentration
    avg_prices = [np.mean(state.prices) for state in market.history]
    ax4.plot(periods, avg_prices, color='orange', linewidth=2,
             label='Average Market Price')
    ax4.axhline(y=analysis['monopoly_benchmark'], color='red', linestyle='--',
                label='Monopoly Price', alpha=0.7)
    ax4.axhline(y=analysis['competitive_benchmark'], color='green',
                linestyle='--',
                label='Competitive Price', alpha=0.7)
    ax4.set_xlabel('Period')
    ax4.set_ylabel('Average Price')
    ax4.set_title('Average Market Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Total market quantity (shows customer participation)
    total_quantities = [sum(state.quantities) for state in market.history]
    ax5.plot(periods, total_quantities, color='brown', linewidth=2)
    ax5.set_xlabel('Period')
    ax5.set_ylabel('Total Quantity Sold')
    ax5.set_title('Market Participation (Total Sales)')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Market participation rate
    participation_rates = [sum(state.quantities) / market.market_size for state
                           in market.history]
    ax6.plot(periods, participation_rates, color='teal', linewidth=2)
    ax6.set_xlabel('Period')
    ax6.set_ylabel('Participation Rate')
    ax6.set_title('Customer Participation Rate')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Run the simulation
def main():
    print("Oligopoly Algorithmic Pricing Simulation")
    print("=" * 50)

    # Create market with 3 firms
    market = OligopolyMarket(n_firms=3, market_size=1000)

    print("Running simulation...")
    history = market.run_simulation(n_periods=300)

    # Analyze results
    analysis = analyze_collusion(history)

    print("\nSimulation Results:")
    print(f"Final Average Price: ${analysis['final_average_price']:.2f}")
    print(f"Competitive Benchmark: ${analysis['competitive_benchmark']:.2f}")
    print(f"Monopoly Benchmark: ${analysis['monopoly_benchmark']:.2f}")
    print(
        f"Collusion Index: {analysis['collusion_index']:.3f} (0=competitive, 1=monopoly)")
    # Calculate final participation rate
    final_periods = market.history[-10:]
    participation_rates = [sum(state.quantities) / market.market_size for state
                           in final_periods]
    final_participation = np.mean(
        participation_rates) if participation_rates else 0

    print(f"Final Market Participation Rate: {final_participation:.1%}")
    print(f"Customers Not Buying: {(1 - final_participation) * 100:.1f}%")
    if analysis['super_collusion_index'] > 0:
        print(
            f"Super-Collusion Index: {analysis['super_collusion_index']:.3f} (prices above monopoly level)")
    print(
        f"Price Variance (final periods): {analysis['avg_price_variance']:.3f}")

    # Show final Q-tables (learning outcomes)
    print("\nLearned Pricing Strategies (Q-tables):")
    for i, firm in enumerate(market.firms):
        print(f"\nFirm {i + 1} Q-table (top 5 states):")
        sorted_states = sorted(firm.q_table.items(),
                               key=lambda x: max(x[1].values()), reverse=True)[
                        :5]
        for state, actions in sorted_states:
            best_action = max(actions, key=actions.get)
            print(
                f"  {state}: {best_action} (Q-value: {actions[best_action]:.2f})")

    # Create visualizations
    plot_simulation_results(market, analysis)

    return market, analysis


if __name__ == "__main__":
    market, analysis = main()