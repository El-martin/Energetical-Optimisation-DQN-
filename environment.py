import numpy as np


class Machine:
    """ A machine characterized by
    -the item it takes as an input
    -the finished product is outputs
    -its yield... (see init)
    Designed to be a part of a Factory """

    def __init__(self, name, prod_delay, in_product, max_intake, out_product, machine_yield):

        self.name = name
        self.prod_delay = prod_delay

        self.in_product = in_product
        self.max_intake = max_intake
        self.out_product = out_product
        self.machine_yield = machine_yield

        self.occupied = False
        self.current_qty = 0
        self.clock = 0

    def start_production(self, stock):
        if not self.occupied and self.in_product in stock.keys():
            self.occupied = True
            self.clock = 0
            self.current_qty = min(stock[self.in_product], self.max_intake)
            stock[self.in_product] -= self.current_qty

        return stock

    def is_done(self):
        if self.clock >= self.prod_delay:
            return True
        return False

    def complete_production(self):
        if self.is_done():
            produced = self.machine_yield * self.current_qty
            self.current_qty = 0
            self.clock = 0
            self.occupied = False

            return {self.out_product: produced}

        return {self.out_product: 0}


class Factory:
    """ An entity centralizing machines, holding a stock of goods that can be transormed
    or sent to the factories it is connected to. """

    def __init__(self, name, links_in, links_out, machine_chain, storage_size, init_stock, hold_cost):

        self.name = name
        self.links_in = links_in
        self.links_out = links_out

        self.machine_chain = machine_chain

        self.storage = storage_size
        self.init_stock = init_stock
        self.stock = init_stock
        self.hold_cost = hold_cost

        self.clock = 0

    def reset(self):
        self.clock = 0
        self.stock = self.init_stock

    def space_left(self):
        occupied = sum(self.stock.values())
        return self.storage - occupied

    def add_stock(self, name, qty):
        self.stock[name] += min(self.space_left(), qty)

    def send_stock(self, name, qty):
        self.stock[name] -= min(self.stock[name], qty)

    def produce(self):
        for machine in self.machine_chain:
            self.stock = machine.start_production(self.stock)

    def send_production(self, prod_name, target, send_qty=0):
        target.add_stock(prod_name, send_qty)
        self.send_stock(prod_name, send_qty)

    def step(self, time_step=1):
        self.clock += time_step
