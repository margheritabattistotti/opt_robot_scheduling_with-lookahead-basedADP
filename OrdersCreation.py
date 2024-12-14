import random

orders = [{"objectA": 1, "objectB": 2, "objectC": 1, "objectD": 0, "objectE": 0},
          {"objectA": 3, "objectB": 0, "objectC": 0, "objectD": 1, "objectE": 2},
          {"objectA": 0, "objectB": 3, "objectC": 0, "objectD": 0, "objectE": 2},
          {"objectA": 0, "objectB": 0, "objectC": 3, "objectD": 3, "objectE": 0},
          {"objectA": 2, "objectB": 0, "objectC": 0, "objectD": 2, "objectE": 1},
          {"objectA": 2, "objectB": 0, "objectC": 1, "objectD": 1, "objectE": 1},
          {"objectA": 0, "objectB": 1, "objectC": 2, "objectD": 1, "objectE": 2},
          {"objectA": 4, "objectB": 0, "objectC": 1, "objectD": 0, "objectE": 0}]


def create_all_orders(n):
    # INPUT
    # n: number of orders to be created
    all_orders = [random.choice(orders) for _ in range(n)]
    return all_orders
