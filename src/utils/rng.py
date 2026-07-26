import random
from typing import Optional

def make_rng(seed: Optional[int] = None) -> random.Random:
   
    rng = random.Random(seed)
    if seed is not None:
        rng.seed(seed)
    return rng


   
   
    """
    Create and return a random number generator (RNG) instance.

   
        seed (Optional[int]): An optional seed for the RNG. If provided, the RNG
                                will produce a deterministic sequence of numbers. 
                              If None, the rng will be initialized randomly.
                              
    Returns: random.Random.  its an instance of Python's built-in random number generator.
        
        
    """