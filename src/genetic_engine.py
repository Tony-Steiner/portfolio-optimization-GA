import numpy as np
import pandas as pd
import random


class GeneticOptimizer:
    def __init__(self, prices_df, risk_free_rate=0.04):
        """
        prices_df: DataFrame with ALL potential assets (columns = Tickers, index = Date).
        risk_free_rate: Annualized rate (e.g., 0.04 for 4%) used for Sharpe Ratio.
        """
        if prices_df.empty:
            raise ValueError(
                "Error: Input DataFrame is empty. Cannot optimize nothing."
            )

        self.prices = prices_df
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + self.risk_free_rate) ** (1 / 252) - 1  # Convert to daily

        # We calculate returns ONCE here to speed up the loop
        # dropna() removes the first row (t=0) which becomes NaN after log calculation
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Convert to numpy for 10x speed boost during repetitive loops
        self.returns_matrix = self.log_returns.values
        self.tickers = list(self.log_returns.columns)

    def fitness_function(self, portfolio_indices):
        """
        Calculates the Sharpe Ratio using raw numpy arrays (much faster than Pandas).
        portfolio_indices: List of integer indices corresponding to the columns.
        """
        # 1. Slice the matrix (Get only columns for the selected portfolio)
        selected_returns = self.returns_matrix[:, portfolio_indices]

        # 2. Calculate Portfolio Return (Equal Weight = Mean across columns)
        # axis=1 means average across the row (daily return of the basket)
        portfolio_daily_ret = np.mean(selected_returns, axis=1)

        # 3. Calculate Metrics
        daily_mean = np.mean(portfolio_daily_ret)
        daily_std = np.std(portfolio_daily_ret)

        if daily_std == 0:
            return -9999

        sharpe = (daily_mean - self.daily_rf) / daily_std
        return sharpe * np.sqrt(252)

    def run_optimization(
        self,
        population_size=50,
        generations=30,
        portfolio_size=30,
        mutation_rate=0.1,
        seed=927,
    ):
        """
        The Main Loop: Evolution happens here.
        seed: Set this to an integer (e.g., 42) to get reproducible results.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Safety Check: Can we actually build a portfolio this big?
        n_assets = len(self.tickers)
        if portfolio_size > n_assets:
            print(
                f"WARNING: Requested {portfolio_size} stocks but only {n_assets} available."
            )
            print(f"Adjusting portfolio_size to {n_assets}.")
            portfolio_size = n_assets

        print(
            f"--- Starting Evolution: {generations} Gen, {population_size} Pop, {portfolio_size} Assets ---"
        )

        # 1. Create Initial Population (Random Indices)
        # We work with INDICES (integers) instead of strings for speed
        all_indices = list(range(n_assets))
        population = []

        for _ in range(population_size):
            ind = random.sample(all_indices, portfolio_size)
            population.append(ind)

        # 2. Evolution Loop
        for gen in range(generations):
            scores = []

            # A. Evaluate Fitness
            for individual in population:
                score = self.fitness_function(individual)
                scores.append((score, individual))

            # Sort: Highest Sharpe Ratio first
            scores.sort(key=lambda x: x[0], reverse=True)

            if gen % 5 == 0:
                print(f"Gen {gen}: Best Sharpe = {scores[0][0]:.4f}")

            # B. Selection (Keep top 20%)
            cutoff = int(len(scores) * 0.2)
            survivors = scores[:cutoff]
            parents = [x[1] for x in survivors]

            # C. Breeding
            next_generation = list(parents)

            while len(next_generation) < population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                split = portfolio_size // 2
                child = parent1[:split] + parent2[split:]

                # Fix Duplicates
                child = list(set(child))
                while len(child) < portfolio_size:
                    new_pick = random.choice(all_indices)
                    if new_pick not in child:
                        child.append(new_pick)

                # D. Mutation
                if random.random() < mutation_rate:
                    remove_idx = random.randint(0, portfolio_size - 1)
                    new_gene = random.choice(all_indices)
                    while new_gene in child:
                        new_gene = random.choice(all_indices)
                    child[remove_idx] = new_gene

                next_generation.append(child)

            population = next_generation

        # 3. Final Result
        final_scores = [(self.fitness_function(ind), ind) for ind in population]
        final_scores.sort(key=lambda x: x[0], reverse=True)

        best_sharpe, best_indices = final_scores[0]

        # Convert indices back to Ticker Strings
        best_portfolio_tickers = [self.tickers[i] for i in best_indices]

        print(f"--- Evolution Complete. Top Sharpe: {best_sharpe:.4f} ---")
        return best_portfolio_tickers
