class WindowNeighbor:
    """The window class for finding the
    neighbor pixels around the center"""
    
    def __init__(self, width, center, image):
        # center is a list of [row, col, Y_intensity]
        self.center = [center[0], center[1], image[center][0]]
        self.width = width
        self.neighbors = None
        self.find_neighbors(image)
        self.mean = None
        self.var = None

    def find_neighbors(self, image):
        self.neighbors = []
        ix_r_min = max(0, self.center[0] - self.width)
        ix_r_max = min(image.shape[0], self.center[0] + self.width + 1)
        ix_c_min = max(0, self.center[1] - self.width)
        ix_c_max = min(image.shape[1], self.center[1] + self.width + 1)
        for r in range(ix_r_min, ix_r_max):
            for c in range(ix_c_min, ix_c_max):
                if r == self.center[0] and c == self.center[1]:
                    continue
                self.neighbors.append([r, c, image[r, c, 0]])

    def __str__(self):
        return 'windows c=(%d, %d, %f) size: %d' % (
            self.center[0], self.center[1],
            self.center[2], len(self.neighbors)
        )
