"""Example target for a Dream-RSI campaign: make prime_sum fast, keep it exact."""


def prime_sum(n: int) -> int:
    """Sum of all primes strictly below n."""
    total = 0
    for k in range(2, n):
        if all(k % d for d in range(2, k)):
            total += k
    return total
