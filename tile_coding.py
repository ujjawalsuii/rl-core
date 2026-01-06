import numpy as np
import math

class TileCoder:
    """
    Implements Tile Coding (Sutton & Barto, Sec 9.5).
    Maps continuous state spaces to sparse binary feature vectors.
    """
    def __init__(self, iht_size=4096, num_tilings=8):
        self.iht_size = iht_size
        self.num_tilings = num_tilings
        # Random offsets for each tiling to prevent symmetry
        self.offsets = [np.random.uniform(0, 1, size=2) for _ in range(num_tilings)]

    def get_features(self, state, scale=10.0):
        """
        Returns the indices of active tiles for a given continuous state.
        state: tuple/list of (x, y) coordinates
        """
        active_tiles = []
        x, y = state
        
        for i in range(self.num_tilings):
            # Scale state and add offset
            scaled_x = (x * scale) + self.offsets[i][0]
            scaled_y = (y * scale) + self.offsets[i][1]
            
            # Discretize to integer coordinates
            tile_x = int(math.floor(scaled_x))
            tile_y = int(math.floor(scaled_y))
            
            # Hash to get a unique index within the IHT size
            # Using a simple large prime hash for 'semi-random' distribution
            h = hash((tile_x, tile_y, i)) % self.iht_size
            active_tiles.append(h)
            
        return np.array(active_tiles)
