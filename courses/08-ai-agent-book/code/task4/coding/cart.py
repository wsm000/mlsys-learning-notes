def total_after_discount(prices, discount):
    """Return total; discount is a fraction between zero and one."""
    if not (0 <= discount <= 1):
        raise ValueError('discount must be between 0 and 1')
    return sum(prices) * (1 - discount)
